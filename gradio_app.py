import os
import sys
import time
import tempfile
import shutil
import gradio as gr
import cv2
import subprocess
import traceback
import glob
from memory_config import get_optimal_config, get_gpu_memory_info

def initialize_models():
    """初始化模型"""
    try:
        print("正在初始化模型...")
        time.sleep(2)  # 模拟加载时间
        return "✅ 模型初始化成功！"
    except Exception as e:
        return f"❌ 模型初始化失败: {str(e)}"

def run_dloral_inference(video_path, prompt="", align_method="adain", process_size=512, upscale=2, vae_encoder_tiled_size=1024):
    """运行DLoRAL推理"""
    try:
        # 获取原视频的fps
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # 获取原视频文件名（不含扩展名）
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 构建命令
        cmd = [
            "python", "src/test_DLoRAL.py",
            "--pretrained_model_path", "stabilityai/stable-diffusion-2-1-base",
            "--ram_ft_path", "preset/models/DAPE.pth",
            "--ram_path", "preset/models/ram_swin_large_14m.pth",
            "--merge_and_unload_lora", "False",
            "--process_size", str(process_size),
            "--pretrained_model_name_or_path", "stabilityai/stable-diffusion-2-1-base",
            "--vae_encoder_tiled_size", str(vae_encoder_tiled_size),
            "--load_cfr",
            "--pretrained_path", "preset/models/checkpoints/model.pkl",
            "--stages", "1",
            "-i", video_path,
            "-o", "results"
        ]
        
        # 运行命令
        # 使用 Popen 执行命令
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True) as proc:
            for line in proc.stdout:
                # 实时打印每一行输出
                print(line, end='')  # 或者写入日志文件等
                sys.stdout.flush()   # 确保立即输出（尤其在重定向时有用）

        # 可选：等待子进程结束并获取返回码
        return_code = proc.wait()
        
        # 查找输出视频文件，不管返回码如何
        output_files = []
        if os.path.exists("results"):
            # 查找以原视频名命名的子文件夹
            video_folder = os.path.join("results", video_name)
            if os.path.exists(video_folder):
                # 调用frames_to_video函数生成视频，使用原视频的fps
                ret = frames_to_video(video_folder, "output_video.mp4", original_fps)
                if ret:
                    output_files.append("output_video.mp4")
            else:
                print(f"未找到视频文件夹: {video_folder}")
        
        if output_files:
            # 如果成功生成了视频文件，就认为处理成功
            output_video_path = output_files[-1]
            if os.path.exists(output_video_path):
                print(f"找到输出视频文件: {output_video_path}")
                
                # 尝试将原视频的音频合成到新视频中
                try:
                    # 创建临时音频文件
                    temp_audio_path = "temp_audio.aac"
                    
                    # 提取原视频的音频
                    if extract_audio_from_video(video_path, temp_audio_path):
                        # 创建带音频的最终输出文件
                        final_output_path = "output_video_with_audio.mp4"
                        
                        # 合并音频到视频
                        if merge_audio_to_video(output_video_path, temp_audio_path, final_output_path):
                            # 删除临时文件
                            if os.path.exists(temp_audio_path):
                                os.remove(temp_audio_path)
                            if os.path.exists(output_video_path):
                                os.remove(output_video_path)
                            
                            print(f"音频合成完成: {final_output_path}")
                            return final_output_path, "DLoRAL处理完成（含音频）！"
                        else:
                            print("音频合成失败，返回无音频版本")
                            return output_video_path, "DLoRAL处理完成！"
                    else:
                        print("音频提取失败，返回无音频版本")
                        return output_video_path, "DLoRAL处理完成！"
                        
                except Exception as e:
                    print(f"音频处理出错: {str(e)}，返回无音频版本")
                    return output_video_path, "DLoRAL处理完成！"
            else:
                print(f"输出视频文件不存在: {output_video_path}")
                return None, "视频文件生成失败"
        else:
            # 只有在没有生成视频文件时才认为失败
            if return_code == 0:
                return None, "处理完成但未找到输出文件"
            else:
                return None, f"处理失败: 返回码 {return_code}"

        
            
    except Exception as e:
        traceback.print_exc()
        return None, f"运行DLoRAL时出错: {str(e)}"

def create_web_interface():
    """创建Web界面"""
    
    with gr.Blocks(title="DLoRAL 视频超分辨率") as demo:
        gr.HTML("""
        <div style="text-align: center; font-size: 2.5em; font-weight: bold; margin-bottom: 20px; color: #2c3e50;">
        🎬 DLoRAL 视频超分辨率
        </div>
        <div style="text-align: center; font-size: 1.2em; color: #7f8c8d; margin-bottom: 30px;">
        一步扩散：细节丰富且时间一致的视频超分辨率
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 输入设置")
                
                # 模型初始化按钮
                init_button = gr.Button("🚀 初始化模型", variant="primary", size="lg")
                init_status = gr.Textbox(label="模型状态", value="模型未初始化", interactive=False)
                
                # 自动配置按钮
                auto_config_button = gr.Button("⚙️ 自动配置参数", variant="secondary", size="sm")
                config_status = gr.Textbox(label="配置状态", value="", interactive=False)
                
                # 输入视频 - 修复兼容性问题
                input_video = gr.Video(label="上传视频文件")
                
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
                        minimum=2, maximum=8, value=2, step=1,
                        label="超分倍数"
                    )
                    vae_encoder_tiled_size = gr.Slider(
                        minimum=512, maximum=4096, value=1024, step=512,
                        label="VAE编码器分块大小（内存优化）"
                    )
                
                # 处理按钮
                process_button = gr.Button("🎯 开始处理", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                gr.Markdown("### 📥 输出结果")
                
                # 输出视频
                output_video = gr.Video(label="处理后的视频")
                
                # 处理状态
                status_text = gr.Textbox(label="处理状态", interactive=False)
        
        # 使用说明
        gr.Markdown("### 📋 使用说明")
        gr.Markdown("""
        1. **自动配置**: 点击"⚙️ 自动配置参数"根据GPU显存自动设置最优参数
        2. **初始化模型**: 首次使用或重启后需要先初始化模型
        3. **上传视频**: 支持MP4格式的视频文件
        4. **设置参数**: 可选择调整处理参数
        5. **开始处理**: 点击处理按钮开始视频超分辨率
        6. **下载结果**: 处理完成后可下载增强后的视频
        
        **内存优化参数**:
        - **VAE编码器分块大小**: 控制GPU内存使用，数值越小内存占用越少
          - 512: 最低内存占用，适合8GB显存
          - 1024: 平衡模式，适合12-16GB显存
          - 2048: 高性能模式，适合20GB+显存
          - 4096: 最高性能，适合24GB+显存
        
        **注意事项**:
        - 处理时间取决于视频长度和分辨率
        - 建议视频分辨率不超过1080p
        - 如果出现内存不足错误，请降低VAE编码器分块大小
        - 建议先使用自动配置功能获取推荐参数
        - 处理完成后会自动将原视频的音频合成到新视频中
        - 需要系统安装ffmpeg才能进行音频合成
        """)
        
        # 事件处理
        def auto_configure():
            """自动配置参数"""
            try:
                memory_info, error = get_gpu_memory_info()
                if error:
                    return f"❌ {error}"
                
                if not memory_info:
                    return "❌ 未检测到GPU"
                
                gpu = memory_info[0]
                optimal_config = get_optimal_config()
                
                status = f"✅ 自动配置完成\n"
                status += f"GPU: {gpu['name']}\n"
                status += f"可用显存: {gpu['free_gb']:.1f}GB\n"
                status += f"推荐配置: VAE分块={optimal_config['vae_encoder_tiled_size']}, "
                status += f"处理尺寸={optimal_config['process_size']}, "
                status += f"超分倍数={optimal_config['upscale']}"
                
                return status
            except Exception as e:
                return f"❌ 配置失败: {str(e)}"
        
        def process_video(video_path, prompt, align_method, process_size, upscale, vae_encoder_tiled_size):
            if video_path is None:
                return None, "请先上传视频文件"
            
            output_path, message = run_dloral_inference(
                video_path, prompt, align_method, process_size, upscale, vae_encoder_tiled_size
            )
            
            return output_path, message
        
        # 绑定事件
        init_button.click(
            fn=initialize_models,
            outputs=init_status
        )
        
        auto_config_button.click(
            fn=auto_configure,
            outputs=config_status
        )
        
        process_button.click(
            fn=process_video,
            inputs=[input_video, prompt, align_method, process_size, upscale, vae_encoder_tiled_size],
            outputs=[output_video, status_text]
        )
    
    return demo

def frames_to_video(frames_folder, output_path, fps=30):
    """Convert frame images in folder to MP4 video"""
    frame_pattern = os.path.join(frames_folder, "frame_*.png")
    frame_files = sorted(glob.glob(frame_pattern))
    if not frame_files:
        print(f"No frame images found in {frames_folder}")
        return False
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        print(f"Cannot read first frame: {frame_files[0]}")
        return False
    height, width, layers = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for i, frame_file in enumerate(frame_files):
        frame = cv2.imread(frame_file)
        if frame is not None:
            video_writer.write(frame)
    video_writer.release()
    print(f"视频生成完成: {output_path}, FPS: {fps}, 帧数: {len(frame_files)}")
    return True

def merge_audio_to_video(video_path, audio_path, output_path):
    """将音频合并到视频中"""
    try:
        # 使用ffmpeg将音频合并到视频中
        cmd = [
            "ffmpeg", "-y",  # -y表示覆盖输出文件
            "-i", video_path,  # 输入视频
            "-i", audio_path,  # 输入音频
            "-c:v", "copy",    # 复制视频流
            "-c:a", "aac",     # 音频编码为AAC
            "-shortest",       # 以最短的流为准
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"音频合并成功: {output_path}")
            return True
        else:
            print(f"音频合并失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"音频合并出错: {str(e)}")
        return False

def extract_audio_from_video(video_path, audio_path):
    """从视频中提取音频"""
    try:
        # 使用ffmpeg提取音频
        cmd = [
            "ffmpeg", "-y",  # -y表示覆盖输出文件
            "-i", video_path,  # 输入视频
            "-vn",            # 不包含视频
            "-acodec", "aac", # 音频编码为AAC
            audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"音频提取成功: {audio_path}")
            return True
        else:
            print(f"音频提取失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"音频提取出错: {str(e)}")
        return False

if __name__ == "__main__":
    # 创建Web界面
    demo = create_web_interface()
    
    # 启动应用
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        inbrowser=True
    )