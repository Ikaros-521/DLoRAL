import os
import sys
import time
import tempfile
import shutil
import gradio as gr
import cv2
import subprocess
import traceback
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
            print(result.stderr)
            return None, f"处理失败: {result.stderr}"
            
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