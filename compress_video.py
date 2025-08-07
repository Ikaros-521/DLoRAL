import cv2
import sys
import os

def compress_video(input_path, output_path, scale_ratio=0.5):
    """
    按比例压缩视频分辨率
    
    参数:
        input_path (str): 输入视频文件路径
        output_path (str): 输出视频文件路径
        scale_ratio (float): 缩放比例 (0 < ratio ≤ 1)
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"❌ 错误：输入文件 {input_path} 不存在")
        return

    # 创建输出目录（如果需要）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ 错误：无法打开视频文件 {input_path}")
        return

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n原始视频信息:")
    print(f"  文件路径: {input_path}")
    print(f"  分辨率: {original_width}x{original_height}")
    print(f"  帧率: {fps:.2f} FPS")
    print(f"  总帧数: {total_frames}")

    # 检查缩放比例有效性
    if not (0 < scale_ratio <= 1):
        print("❌ 错误：缩放比例必须在 0 到 1 之间")
        cap.release()
        return

    # 计算目标分辨率（保持宽高比）
    new_width = int(original_width * scale_ratio)
    new_height = int(original_height * scale_ratio)
    
    # 防止尺寸过小
    if new_width < 1 or new_height < 1:
        print("❌ 错误：缩放后尺寸过小（至少需要 1x1）")
        cap.release()
        return

    # 设置输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编码器
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    if not out.isOpened():
        print("❌ 错误：无法创建输出视频文件")
        cap.release()
        return

    print(f"\n开始压缩视频到 {new_width}x{new_height} 分辨率 (比例: {scale_ratio})...")
    
    processed_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 调整分辨率
        resized_frame = cv2.resize(frame, (new_width, new_height))
        out.write(resized_frame)
        processed_frames += 1
        
        # 显示进度
        if processed_frames % 30 == 0:
            progress = (processed_frames / total_frames) * 100
            print(f"进度: {progress:.1f}% ({processed_frames}/{total_frames} 帧)")

    # 释放资源
    cap.release()
    out.release()

    # 验证输出文件
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"\n✅ 成功！视频已保存到: {output_path}")
        print(f"  新分辨率: {new_width}x{new_height}")
        print(f"  宽高比: {original_width}:{original_height} -> {new_width}:{new_height}")
    else:
        print("\n❌ 处理失败：输出文件未生成或为空")

if __name__ == "__main__":
    # 检查参数数量
    if len(sys.argv) < 3:
        print("使用方法:")
        print("  python compress_video.py <输入视频路径> <输出视频路径> [缩放比例]")
        print("示例:")
        print("  python compress_video.py input.mp4 out/output.mp4 0.5")
        sys.exit(1)

    # 解析参数
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # 解析缩放比例（默认0.5）
    scale_ratio = 0.5
    if len(sys.argv) > 3:
        try:
            scale_ratio = float(sys.argv[3])
        except ValueError:
            print("❌ 错误：缩放比例必须是数值，例如 0.5")
            sys.exit(1)

    # 执行压缩
    compress_video(input_path, output_path, scale_ratio)