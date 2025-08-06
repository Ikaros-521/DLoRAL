#!/usr/bin/env python3
"""
DLoRAL Web应用启动脚本
"""

import os
import sys
import subprocess
import argparse

def check_dependencies():
    """检查依赖"""
    try:
        import gradio
        print("✅ Gradio已安装")
    except ImportError:
        print("❌ Gradio未安装，正在安装...")
        subprocess.run([sys.executable, "-m", "pip", "install", "gradio>=4.0.0"])
    
    try:
        import cv2
        print("✅ OpenCV已安装")
    except ImportError:
        print("❌ OpenCV未安装，正在安装...")
        subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python>=4.8.0"])

def check_models():
    """检查模型文件"""
    required_files = [
        "preset/models/ram_swin_large_14m.pth",
        "preset/models/DAPE.pth", 
        "preset/models/checkpoints/model.pkl"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 缺少以下模型文件:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n请按照README.md中的说明下载相应的模型文件")
        return False
    
    print("✅ 所有模型文件已就绪")
    return True

def main():
    parser = argparse.ArgumentParser(description="DLoRAL Web应用启动脚本")
    parser.add_argument("--port", type=int, default=7860, help="服务器端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器地址")
    parser.add_argument("--share", action="store_true", help="是否启用公共链接")
    parser.add_argument("--check-only", action="store_true", help="仅检查依赖和模型文件")
    
    args = parser.parse_args()
    
    print("🚀 DLoRAL Web应用启动中...")
    print("=" * 50)
    
    # 检查依赖
    print("📦 检查依赖...")
    check_dependencies()
    
    # 检查模型文件
    print("\n🔍 检查模型文件...")
    if not check_models():
        if not args.check_only:
            print("\n❌ 模型文件不完整，无法启动Web应用")
            return 1
    
    if args.check_only:
        print("\n✅ 检查完成，所有依赖和模型文件都已就绪")
        return 0
    
    # 启动Web应用
    print("\n🌐 启动Web应用...")
    try:
        from gradio_app import create_web_interface
        
        demo = create_web_interface()
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            debug=True,
            inbrowser=True
        )
    except Exception as e:
        print(f"❌ 启动Web应用失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 