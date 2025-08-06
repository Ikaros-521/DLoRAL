#!/usr/bin/env python3
"""
DLoRAL 内存配置助手
根据GPU显存大小推荐合适的处理参数
"""

import torch

def get_gpu_memory_info():
    """获取GPU内存信息"""
    if not torch.cuda.is_available():
        return None, "CUDA不可用"
    
    gpu_count = torch.cuda.device_count()
    memory_info = []
    
    for i in range(gpu_count):
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
        allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)  # GB
        free_memory = total_memory - allocated_memory
        
        memory_info.append({
            'device_id': i,
            'name': torch.cuda.get_device_name(i),
            'total_gb': total_memory,
            'allocated_gb': allocated_memory,
            'free_gb': free_memory
        })
    
    return memory_info, None

def recommend_parameters(gpu_memory_gb):
    """根据GPU显存推荐参数"""
    if gpu_memory_gb < 8:
        return {
            'vae_encoder_tiled_size': 512,
            'process_size': 256,
            'upscale': 2,
            'description': '低内存模式 - 适合8GB以下显存'
        }
    elif gpu_memory_gb < 12:
        return {
            'vae_encoder_tiled_size': 1024,
            'process_size': 384,
            'upscale': 2,
            'description': '平衡模式 - 适合8-12GB显存'
        }
    elif gpu_memory_gb < 16:
        return {
            'vae_encoder_tiled_size': 1024,
            'process_size': 512,
            'upscale': 4,
            'description': '标准模式 - 适合12-16GB显存'
        }
    elif gpu_memory_gb < 20:
        return {
            'vae_encoder_tiled_size': 2048,
            'process_size': 512,
            'upscale': 4,
            'description': '高性能模式 - 适合16-20GB显存'
        }
    else:
        return {
            'vae_encoder_tiled_size': 4096,
            'process_size': 512,
            'upscale': 4,
            'description': '最高性能模式 - 适合20GB+显存'
        }

def print_memory_analysis():
    """打印内存分析报告"""
    print("🔍 DLoRAL 内存配置分析")
    print("=" * 50)
    
    memory_info, error = get_gpu_memory_info()
    
    if error:
        print(f"❌ {error}")
        return
    
    for gpu in memory_info:
        print(f"\n🎮 GPU {gpu['device_id']}: {gpu['name']}")
        print(f"   总显存: {gpu['total_gb']:.1f} GB")
        print(f"   已使用: {gpu['allocated_gb']:.1f} GB")
        print(f"   可用: {gpu['free_gb']:.1f} GB")
        
        # 推荐参数
        recommended = recommend_parameters(gpu['free_gb'])
        print(f"\n📋 推荐配置 ({recommended['description']}):")
        print(f"   VAE编码器分块大小: {recommended['vae_encoder_tiled_size']}")
        print(f"   处理尺寸: {recommended['process_size']}")
        print(f"   超分倍数: {recommended['upscale']}")
        
        # 内存使用估算
        estimated_memory = estimate_memory_usage(recommended)
        print(f"\n💾 预估内存使用: {estimated_memory:.1f} GB")
        
        if estimated_memory > gpu['free_gb']:
            print("⚠️  警告: 预估内存使用超过可用显存，建议降低参数")
        else:
            print("✅ 内存配置合理")

def estimate_memory_usage(config):
    """估算内存使用量"""
    # 简化的内存估算模型
    base_memory = 2.0  # 基础模型内存
    vae_factor = config['vae_encoder_tiled_size'] / 1024.0
    process_factor = (config['process_size'] / 512.0) ** 2
    upscale_factor = config['upscale'] / 4.0
    
    estimated = base_memory * vae_factor * process_factor * upscale_factor
    return min(estimated, 20.0)  # 最大估算20GB

def get_optimal_config():
    """获取最优配置"""
    memory_info, error = get_gpu_memory_info()
    
    if error or not memory_info:
        # 默认配置
        return {
            'vae_encoder_tiled_size': 1024,
            'process_size': 512,
            'upscale': 2
        }
    
    # 使用第一个GPU的可用内存
    free_memory = memory_info[0]['free_gb']
    return recommend_parameters(free_memory)

if __name__ == "__main__":
    print_memory_analysis() 