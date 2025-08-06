@echo off
echo ========================================
echo DLoRAL Web应用依赖安装脚本
echo ========================================
echo.

echo 正在检查Python环境...
python --version
if %errorlevel% neq 0 (
    echo 错误: 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo.
echo 正在安装Web应用依赖...
echo.

echo 安装Gradio...
pip install gradio>=4.0.0
if %errorlevel% neq 0 (
    echo 警告: Gradio安装失败
)

echo 安装OpenCV...
pip install opencv-python>=4.8.0
if %errorlevel% neq 0 (
    echo 警告: OpenCV安装失败
)

echo 安装其他依赖...
pip install -r requirements_web.txt
if %errorlevel% neq 0 (
    echo 警告: 部分依赖安装失败
)

echo.
echo ========================================
echo 依赖安装完成！
echo ========================================
echo.
echo 下一步:
echo 1. 确保已下载所有模型文件
echo 2. 运行: python start_web_app.py --check-only
echo 3. 启动Web应用: python start_web_app.py
echo.
pause 