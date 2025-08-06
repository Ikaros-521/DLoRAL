#!/bin/bash

echo "========================================"
echo "DLoRAL Web应用依赖安装脚本"
echo "========================================"
echo

# 检查Python环境
echo "正在检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python 3.8+"
    exit 1
fi

python3 --version

echo
echo "正在安装Web应用依赖..."
echo

# 安装Gradio
echo "安装Gradio..."
pip3 install gradio>=4.0.0
if [ $? -ne 0 ]; then
    echo "警告: Gradio安装失败"
fi

# 安装OpenCV
echo "安装OpenCV..."
pip3 install opencv-python>=4.8.0
if [ $? -ne 0 ]; then
    echo "警告: OpenCV安装失败"
fi

# 安装其他依赖
echo "安装其他依赖..."
pip3 install -r requirements_web.txt
if [ $? -ne 0 ]; then
    echo "警告: 部分依赖安装失败"
fi

echo
echo "========================================"
echo "依赖安装完成！"
echo "========================================"
echo
echo "下一步:"
echo "1. 确保已下载所有模型文件"
echo "2. 运行: python3 start_web_app.py --check-only"
echo "3. 启动Web应用: python3 start_web_app.py"
echo 