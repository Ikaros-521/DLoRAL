import os
import sys
import time
import tempfile
import shutil
import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

# 添加项目路径到系统路径
sys.path.append(os.getcwd())

# 导入DLoRAL模型
from src.DLoRAL_model import Generator_eval
from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from ram.models.ram_lora import ram
from ram import inference_ram as inference

# 设置PIL最大图像像素
PIL.Image.MAX_IMAGE_PIXELS = 933120000

# 全局变量
model = None
DAPE = None
weight_dtype = torch.float32

# 变换器
tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
])

ram_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def initialize_models():
    """初始化模型"""
    global model, DAPE, weight_dtype
    
    print("正在初始化模型...")
    
    # 模型配置
    class ModelConfig:
        def __init__(self):
            self.pretrained_model_path = "stabilityai/stable-diffusion-2-1-base"
            self.pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1-base"
            self.ram_path = "preset/models/ram_swin_large_14m.pth"
            self.ram_ft_path = "preset/models/DAPE.pth"
            self.pretrained_path = "preset/models/checkpoints/model.pkl"
            self.process_size = 512
            self.upscale = 4
            self.align_method = "adain"
            self.vae_decoder_tiled_size = 224
            self.vae_encoder_tiled_size = 4096
            self.latent_tiled_size = 96
            self.latent_tiled_overlap = 32
            self.mixed_precision = "fp16"
            self.merge_and_unload_lora = False
            self.stages = 1
            self.load_cfr = True
            self.prompt = ""
            self.save_prompts = False
    
    args = ModelConfig()
    
    # 初始化DLoRAL模型
    model = Generator_eval(args)
    model.set_eval()
    
    # 初始化RAM模型
    DAPE = ram(pretrained=args.ram_path,
               pretrained_condition=args.ram_ft_path,
               image_size=384,
               vit='swin_l')
    DAPE.eval()
    DAPE.to("cuda")
    
    # 设置权重类型
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # 设置模型权重类型
    DAPE = DAPE.to(dtype=weight_dtype)
    model.vae = model.vae.to(dtype=weight_dtype)
    model.unet = model.unet.to(dtype=weight_dtype)
    model.cfr_main_net = model.cfr_main_net.to(dtype=weight_dtype)
    
    # 设置adapter
    if args.stages == 0:
        model.unet.set_adapter(['default_encoder_consistency', 'default_decoder_consistency', 'default_others_consistency'])
    else:
        model.unet.set_adapter(['default_encoder_quality', 'default_decoder_quality',
                                'default_others_quality',
                                'default_encoder_consistency', 'default_decoder_consistency',
                                'default_others_consistency'])
    
    print("模型初始化完成！")

def get_validation_prompt(image, prompt=""):
    """获取验证提示词"""
    global DAPE, weight_dtype
    
    validation_prompt = ""
    lq = tensor_transforms(image).unsqueeze(0).to("cuda")
    lq = ram_transforms(lq).to(dtype=weight_dtype)
    captions = inference(lq, DAPE)
    validation_prompt = f"{captions[0]}, {prompt},"
    
    return validation_prompt

def extract_frames(video_path, output_dir):
    """提取视频帧"""
    video_capture = cv2.VideoCapture(video_path)
    
    frame_number = 0
    success, frame = video_capture.read()
    frame_images = []
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 循环处理帧
    while success:
        frame_filename = f"frame_{frame_number:04d}.png"
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, frame)
        frame_images.append(frame_path)
        
        success, frame = video_capture.read()
        frame_number += 1
    
    video_capture.release()
    print(f"从 {video_path} 提取了 {len(frame_images)} 帧")
    
    return frame_images

def compute_frame_difference_mask(frames):
    """计算帧差异掩码"""
    ambi_matrix = frames.var(dim=0)
    threshold = ambi_matrix.mean().item()
    mask_id = torch.where(ambi_matrix >= threshold, ambi_matrix, torch.zeros_like(ambi_matrix))
    frame_mask = torch.where(mask_id == 0, mask_id, torch.ones_like(mask_id))
    return frame_mask

def process_video_super_resolution(video_path, prompt="", align_method="adain", process_size=512, upscale=4):
    """处理视频超分辨率"""
    global model, weight_dtype
    
    if model is None:
        return None, "模型未初始化，请先初始化模型"
    
    try:
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            frames_dir = os.path.join(temp_dir, "frames")
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            # 提取帧
            frame_images = extract_frames(video_path, frames_dir)
            
            if len(frame_images) == 0:
                return None, "无法从视频中提取帧"
            
            # 处理参数
            frame_num = 2
            frame_overlap = 1
            
            # 初始化批次
            input_image_batch = []
            input_image_gray_batch = []
            bname_batch = []
            exist_prompt = 0
            validation_prompt = ""
            
            # 处理每一帧
            for image_name in frame_images:
                input_image = Image.open(image_name).convert('RGB')
                input_image_gray = input_image.convert('L')
                ori_width, ori_height = input_image.size
                rscale = upscale
                resize_flag = False
                
                # 如果图像小于所需尺寸，进行缩放
                if ori_width < process_size // rscale or ori_height < process_size // rscale:
                    scale = (process_size // rscale) / min(ori_width, ori_height)
                    input_image = input_image.resize((int(scale * ori_width), int(scale * ori_height)))
                    input_image_gray = input_image_gray.resize((int(scale * ori_width), int(scale * ori_height)))
                    resize_flag = True
                
                # 按超分倍数放大图像尺寸
                input_image = input_image.resize((input_image.size[0] * rscale, input_image.size[1] * rscale))
                input_image_gray = input_image_gray.resize((input_image_gray.size[0] * rscale, input_image_gray.size[1] * rscale))
                
                # 调整图像尺寸确保是8的倍数
                new_width = input_image.width - input_image.width % 8
                new_height = input_image.height - input_image.height % 8
                input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
                input_image_gray = input_image_gray.resize((new_width, new_height), Image.LANCZOS)
                
                bname = os.path.basename(image_name)
                bname_batch.append(bname)
                
                # 生成提示词
                if exist_prompt == 0:
                    validation_prompt = get_validation_prompt(input_image, prompt)
                    exist_prompt = 1
                
                input_image_batch.append(input_image)
                input_image_gray_batch.append(input_image_gray)
            
            # 处理帧批次
            processed_frames = []
            
            for input_image_index in range(0, len(input_image_batch), (frame_num - frame_overlap)):
                if input_image_index + frame_num - 1 >= len(input_image_batch):
                    end = len(input_image_batch) - input_image_index
                    start = 0
                else:
                    start = 0
                    end = frame_num
                
                # 收集要处理的帧批次
                input_frames = []
                input_frames_gray = []
                
                for input_frame_index in range(start, end):
                    real_idx = input_image_index + input_frame_index
                    if real_idx < 0 or real_idx >= len(input_image_batch):
                        continue
                    
                    current_frame = transforms.functional.to_tensor(input_image_batch[real_idx])
                    current_frame_gray = transforms.functional.to_tensor(input_image_gray_batch[real_idx])
                    current_frame_gray = torch.nn.functional.interpolate(current_frame_gray.unsqueeze(0), scale_factor=0.125).squeeze(0)
                    input_frames.append(current_frame)
                    input_frames_gray.append(current_frame_gray)
                
                if len(input_frames) < 2:
                    break
                
                input_image_final = torch.stack(input_frames, dim=0)
                input_image_gray_final = torch.stack(input_frames_gray, dim=0)
                
                # 计算不确定性图
                uncertainty_map = []
                for image_index in range(input_image_final.shape[0]):
                    if image_index != 0:
                        cur_img = input_image_gray_final[image_index]
                        prev_img = input_image_gray_final[image_index - 1]
                        compute_frame = torch.stack([cur_img, prev_img])
                        uncertainty_map_each = compute_frame_difference_mask(input_image_gray_final)
                        uncertainty_map.append(uncertainty_map_each)
                
                if len(uncertainty_map) == 0:
                    continue
                
                uncertainty_map = torch.stack(uncertainty_map)
                
                # 模型推理
                with torch.no_grad():
                    c_t = input_image_final.unsqueeze(0).cuda() * 2 - 1
                    c_t = c_t.to(dtype=weight_dtype)
                    output_image, _, _, _, _ = model(stages=1, c_t=c_t, 
                                                   uncertainty_map=uncertainty_map.unsqueeze(0).cuda(), 
                                                   prompt=validation_prompt, 
                                                   weight_dtype=weight_dtype)
                
                frame_t = output_image[0]
                frame_t = (frame_t.cpu() * 0.5 + 0.5)
                output_pil = transforms.ToPILImage()(frame_t)
                
                # 颜色校正
                src_idx = input_image_index + start + 1
                if src_idx < 0 or src_idx >= len(input_image_batch):
                    src_idx = max(0, min(src_idx, len(input_image_batch) - 1))
                
                source_pil = input_image_batch[src_idx]
                
                if align_method == 'adain':
                    output_pil = adain_color_fix(target=output_pil, source=source_pil)
                elif align_method == 'wavelet':
                    output_pil = wavelet_color_fix(target=output_pil, source=source_pil)
                
                # 如果之前进行了缩放，现在恢复原始尺寸
                if resize_flag:
                    new_w = int(upscale * ori_width)
                    new_h = int(upscale * ori_height)
                    output_pil = output_pil.resize((new_w, new_h), Image.BICUBIC)
                
                processed_frames.append(output_pil)
            
            # 保存处理后的帧
            output_frames = []
            for i, frame in enumerate(processed_frames):
                frame_path = os.path.join(output_dir, f"processed_frame_{i:04d}.png")
                frame.save(frame_path)
                output_frames.append(frame_path)
            
            # 创建输出视频
            if len(output_frames) > 0:
                # 读取第一帧获取尺寸
                first_frame = cv2.imread(output_frames[0])
                height, width, layers = first_frame.shape
                
                # 创建视频写入器
                output_video_path = os.path.join(temp_dir, "output_video.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))
                
                # 写入帧
                for frame_path in output_frames:
                    frame = cv2.imread(frame_path)
                    video_writer.write(frame)
                
                video_writer.release()
                
                # 复制到最终输出路径
                final_output_path = os.path.join("results", f"output_{int(time.time())}.mp4")
                os.makedirs("results", exist_ok=True)
                shutil.copy2(output_video_path, final_output_path)
                
                return final_output_path, f"处理完成！共处理了 {len(processed_frames)} 帧"
            else:
                return None, "没有成功处理任何帧"
                
    except Exception as e:
        return None, f"处理过程中出现错误: {str(e)}"

def create_web_interface():
    """创建Web界面"""
    
    # 自定义CSS样式
    css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .title {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 20px;
        color: #2c3e50;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #7f8c8d;
        margin-bottom: 30px;
    }
    """
    
    with gr.Blocks(css=css, title="DLoRAL 视频超分辨率") as demo:
        gr.HTML("""
        <div class="title">🎬 DLoRAL 视频超分辨率</div>
        <div class="subtitle">一步扩散：细节丰富且时间一致的视频超分辨率</div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 输入设置")
                
                # 模型初始化按钮
                init_button = gr.Button("🚀 初始化模型", variant="primary", size="lg")
                init_status = gr.Textbox(label="模型状态", value="模型未初始化", interactive=False)
                
                # 输入视频
                input_video = gr.Video(label="上传视频文件", type="filepath")
                
                # 参数设置
                with gr.Accordion("⚙️ 高级参数", open=False):
                    prompt = gr.Textbox(label="提示词", value="", placeholder="输入额外的提示词（可选）")
                    align_method = gr.Dropdown(
                        choices=["adain", "wavelet", "nofix"], 
                        value="adain", 
                        label="颜色校正方法"
                    )
                    process_size = gr.Slider(
                        minimum=256, maximum=1024, value=512, step=64,
                        label="处理尺寸"
                    )
                    upscale = gr.Slider(
                        minimum=2, maximum=8, value=4, step=1,
                        label="超分倍数"
                    )
                
                # 处理按钮
                process_button = gr.Button("🎯 开始处理", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                gr.Markdown("### 📥 输出结果")
                
                # 输出视频
                output_video = gr.Video(label="处理后的视频")
                
                # 处理状态
                status_text = gr.Textbox(label="处理状态", interactive=False)
                
                # 处理信息
                info_text = gr.Textbox(label="处理信息", interactive=False, lines=3)
        
        # 示例
        gr.Markdown("### 📋 使用说明")
        gr.Markdown("""
        1. **初始化模型**: 首次使用或重启后需要先初始化模型
        2. **上传视频**: 支持MP4格式的视频文件
        3. **设置参数**: 可选择调整处理参数
        4. **开始处理**: 点击处理按钮开始视频超分辨率
        5. **下载结果**: 处理完成后可下载增强后的视频
        
        **注意事项**:
        - 处理时间取决于视频长度和分辨率
        - 建议视频分辨率不超过1080p
        - 确保有足够的GPU内存
        """)
        
        # 事件处理
        def init_models():
            try:
                initialize_models()
                return "✅ 模型初始化成功！"
            except Exception as e:
                return f"❌ 模型初始化失败: {str(e)}"
        
        def process_video(video_path, prompt, align_method, process_size, upscale):
            if video_path is None:
                return None, "请先上传视频文件", "未处理"
            
            if model is None:
                return None, "请先初始化模型", "模型未初始化"
            
            output_path, message = process_video_super_resolution(
                video_path, prompt, align_method, process_size, upscale
            )
            
            return output_path, message, f"处理完成 - {message}"
        
        # 绑定事件
        init_button.click(
            fn=init_models,
            outputs=init_status
        )
        
        process_button.click(
            fn=process_video,
            inputs=[input_video, prompt, align_method, process_size, upscale],
            outputs=[output_video, status_text, info_text]
        )
    
    return demo

if __name__ == "__main__":
    # 创建Web界面
    demo = create_web_interface()
    
    # 启动应用
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    ) 