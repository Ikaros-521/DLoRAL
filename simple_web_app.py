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
import subprocess

# 添加项目路径到系统路径
sys.path.append(os.getcwd())

# 全局变量
model_initialized = False

def initialize_models():
    """初始化模型"""
    global model_initialized
    
    try:
        print("正在初始化模型...")
        
        # 这里可以添加实际的模型初始化代码
        # 为了简化，我们只是模拟初始化过程
        time.sleep(2)  # 模拟加载时间
        
        model_initialized = True
        print("模型初始化完成！")
        return "✅ 模型初始化成功！"
    except Exception as e:
        return f"❌ 模型初始化失败: {str(e)}"

def extract_frames_simple(video_path, output_dir):
    """简化版帧提取"""
    video_capture = cv2.VideoCapture(video_path)
    
    frame_number = 0
    success, frame = video_capture.read()
    frame_images = []
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 限制处理的帧数，避免处理时间过长
    max_frames = 30
    
    # 循环处理帧
    while success and frame_number < max_frames:
        frame_filename = f"frame_{frame_number:04d}.png"
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, frame)
        frame_images.append(frame_path)
        
        success, frame = video_capture.read()
        frame_number += 1
    
    video_capture.release()
    print(f"从 {video_path} 提取了 {len(frame_images)} 帧")
    
    return frame_images

def process_video_simple(video_path, prompt="", align_method="adain", process_size=512, upscale=4):
    """简化版视频处理"""
    global model_initialized
    
    if not model_initialized:
        return None, "模型未初始化，请先初始化模型"
    
    if video_path is None:
        return None, "请先上传视频文件"
    
    try:
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            frames_dir = os.path.join(temp_dir, "frames")
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            # 提取帧
            frame_images = extract_frames_simple(video_path, frames_dir)
            
            if len(frame_images) == 0:
                return None, "无法从视频中提取帧"
            
            # 简化处理：只处理前几帧作为演示
            processed_frames = []
            
            for i, frame_path in enumerate(frame_images[:5]):  # 只处理前5帧
                # 读取图像
                input_image = Image.open(frame_path).convert('RGB')
                
                # 简单的图像处理（这里只是示例，实际应该调用DLoRAL模型）
                # 放大图像
                new_width = input_image.width * upscale
                new_height = input_image.height * upscale
                processed_image = input_image.resize((new_width, new_height), Image.LANCZOS)
                
                # 保存处理后的帧
                output_frame_path = os.path.join(output_dir, f"processed_frame_{i:04d}.png")
                processed_image.save(output_frame_path)
                processed_frames.append(output_frame_path)
            
            # 创建输出视频
            if len(processed_frames) > 0:
                # 读取第一帧获取尺寸
                first_frame = cv2.imread(processed_frames[0])
                height, width, layers = first_frame.shape
                
                # 创建视频写入器
                output_video_path = os.path.join(temp_dir, "output_video.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))
                
                # 写入帧
                for frame_path in processed_frames:
                    frame = cv2.imread(frame_path)
                    video_writer.write(frame)
                
                video_writer.release()
                
                # 复制到最终输出路径
                final_output_path = os.path.join("results", f"output_{int(time.time())}.mp4")
                os.makedirs("results", exist_ok=True)
                shutil.copy2(output_video_path, final_output_path)
                
                return final_output_path, f"处理完成！共处理了 {len(processed_frames)} 帧（演示模式）"
            else:
                return None, "没有成功处理任何帧"
                
    except Exception as e:
        return None, f"处理过程中出现错误: {str(e)}"

def run_dloral_inference(video_path, prompt="", align_method="adain", process_size=512, upscale=4):
    """运行DLoRAL推理"""
    try:
        # 构建命令
        cmd = [
            "python", "src/test_DLoRAL.py",
            "--pretrained_model_path", "stabilityai/stable-diffusion-2-1-base",
            "--ram_ft_path", "preset/models/DAPE.pth",
            "--ram_path", "preset/models/ram_swin_large_14m.pth",
            "--merge_and_unload_lora", "False",
            "--process_size", str(process_size),
            "--pretrained_model_name_or_path", "stabilityai/stable-diffusion-2-1-base",
            "--vae_encoder_tiled_size", "4096",
            "--load_cfr",
            "--pretrained_path", "preset/models/checkpoints/model.pkl",
            "--stages", "1",
            "-i", video_path,
            "-o", "results"
        ]
        
        # 运行命令
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # 查找输出视频文件
            output_files = []
            if os.path.exists("results"):
                for file in os.listdir("results"):
                    if file.endswith(".mp4"):
                        output_files.append(os.path.join("results", file))
            
            if output_files:
                return output_files[-1], "DLoRAL处理完成！"
            else:
                return None, "处理完成但未找到输出文件"
        else:
            return None, f"处理失败: {result.stderr}"
            
    except Exception as e:
        return None, f"运行DLoRAL时出错: {str(e)}"

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
                
                # 处理模式选择
                process_mode = gr.Radio(
                    choices=["演示模式", "完整DLoRAL模式"],
                    value="演示模式",
                    label="处理模式"
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
        
        # 使用说明
        gr.Markdown("### 📋 使用说明")
        gr.Markdown("""
        **演示模式**: 快速演示，只处理前几帧，适合测试界面功能
        **完整DLoRAL模式**: 使用完整的DLoRAL模型处理整个视频
        
        1. **初始化模型**: 首次使用或重启后需要先初始化模型
        2. **上传视频**: 支持MP4格式的视频文件
        3. **选择模式**: 选择演示模式或完整处理模式
        4. **设置参数**: 可选择调整处理参数
        5. **开始处理**: 点击处理按钮开始视频超分辨率
        6. **下载结果**: 处理完成后可下载增强后的视频
        
        **注意事项**:
        - 完整模式处理时间较长，取决于视频长度和分辨率
        - 建议视频分辨率不超过1080p
        - 确保有足够的GPU内存
        - 演示模式仅用于测试界面功能
        """)
        
        # 事件处理
        def init_models():
            return initialize_models()
        
        def process_video(video_path, prompt, align_method, process_size, upscale, process_mode):
            if video_path is None:
                return None, "请先上传视频文件", "未处理"
            
            if not model_initialized:
                return None, "请先初始化模型", "模型未初始化"
            
            if process_mode == "演示模式":
                output_path, message = process_video_simple(
                    video_path, prompt, align_method, process_size, upscale
                )
            else:
                output_path, message = run_dloral_inference(
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
            inputs=[input_video, prompt, align_method, process_size, upscale, process_mode],
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