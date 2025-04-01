#!/usr/bin/env python
"""
一键安装CausalML所需的全部依赖包。
此脚本会自动检测是否存在CUDA环境，并安装相应版本的PyTorch。
支持选择国内镜像源以解决网络连接问题。
"""

import os
import platform
import subprocess
import sys
import time

def run_command(command, max_retries=3):
    """运行命令并打印输出，支持重试"""
    print(f"执行: {command}")
    
    for attempt in range(max_retries):
        if attempt > 0:
            print(f"重试 ({attempt}/{max_retries})...")
            time.sleep(2)  # 重试前等待一段时间
            
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        error_detected = False
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                # 检测常见的网络错误
                if "CONNECTION FAILED" in output or "Read timed out" in output:
                    error_detected = True
        
        exit_code = process.poll()
        if exit_code == 0 or (not error_detected and attempt == max_retries - 1):
            return exit_code
            
    return 1  # 所有重试都失败

def check_cuda_available():
    """检查是否有CUDA环境"""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        # 如果无法导入torch，尝试使用nvidia-smi命令检测
        try:
            subprocess.check_output('nvidia-smi')
            return True
        except:
            return False

def select_mirror():
    """选择安装源"""
    print("\n请选择安装源:")
    print("1. 默认源 (官方源)")
    print("2. 清华大学镜像源 (推荐国内用户使用)")
    print("3. 阿里云镜像源")
    
    while True:
        try:
            choice = input("请输入选项 [1-3] (默认: 2): ").strip() or "2"
            choice = int(choice)
            if 1 <= choice <= 3:
                return choice
            print("无效的选择，请重新输入")
        except ValueError:
            print("请输入有效的数字")

def get_pip_index_url(mirror_choice):
    """根据选择返回pip镜像URL"""
    if mirror_choice == 1:
        return "", ""  # 默认源
    elif mirror_choice == 2:
        return "-i https://pypi.tuna.tsinghua.edu.cn/simple", "--index-url https://pypi.tuna.tsinghua.edu.cn/simple"
    elif mirror_choice == 3:
        return "-i https://mirrors.aliyun.com/pypi/simple/", "--index-url https://mirrors.aliyun.com/pypi/simple/"
    return "", ""  # 默认情况

def get_pytorch_url(mirror_choice, cuda_available):
    """根据选择返回PyTorch安装URL"""
    cuda_suffix = "cu118" if cuda_available else "cpu"
    
    if mirror_choice == 1:
        # 官方源
        return f"https://download.pytorch.org/whl/{cuda_suffix}"
    elif mirror_choice == 2:
        # 清华源
        return f"https://pypi.tuna.tsinghua.edu.cn/simple"
    elif mirror_choice == 3:
        # 阿里云源
        return f"https://mirrors.aliyun.com/pypi/simple/"
    
    # 默认使用官方源
    return f"https://download.pytorch.org/whl/{cuda_suffix}"

def install_dependencies():
    """安装基础依赖项"""
    print("安装基础依赖项...")
    
    # 选择镜像源
    mirror_choice = select_mirror()
    pip_index, pip_index_url = get_pip_index_url(mirror_choice)
    
    # 先安装Cython以避免后续安装问题
    run_command(f"{sys.executable} -m pip install Cython {pip_index}")
    
    # 安装基础依赖
    run_command(f"{sys.executable} -m pip install -r requirements.txt {pip_index}")
    
    # 根据CUDA环境安装PyTorch
    cuda_available = check_cuda_available()
    pytorch_url = get_pytorch_url(mirror_choice, cuda_available)
    
    if cuda_available:
        print("检测到CUDA环境，安装GPU版PyTorch...")
        if mirror_choice == 1:  # 官方源需要特殊处理
            run_command(f"{sys.executable} -m pip install torch torchvision torchaudio --index-url {pytorch_url}")
        else:
            run_command(f"{sys.executable} -m pip install torch torchvision torchaudio {pip_index}")
    else:
        print("未检测到CUDA环境，安装CPU版PyTorch...")
        if mirror_choice == 1:  # 官方源需要特殊处理
            run_command(f"{sys.executable} -m pip install torch torchvision torchaudio --index-url {pytorch_url}")
        else:
            run_command(f"{sys.executable} -m pip install torch torchvision torchaudio {pip_index}")
    
    # 安装CausalML包
    print("安装CausalML...")
    if os.path.isfile("setup.py"):
        run_command(f"{sys.executable} -m pip install -e .[all] {pip_index}")
    else:
        run_command(f"{sys.executable} -m pip install causalml[all] {pip_index}")

def setup_env():
    """设置环境"""
    print("=" * 60)
    print("CausalML 环境安装工具")
    print("=" * 60)
    
    print(f"Python版本: {platform.python_version()}")
    print(f"运行平台: {platform.platform()}")
    
    try:
        install_dependencies()
        
        print("\n安装完成！")
        print("验证安装:")
        result = run_command(f"{sys.executable} -c \"import causalml; print('CausalML版本:', causalml.__version__)\"")
        
        if result != 0:
            print("\n警告: 验证安装时出现问题，请检查安装日志。")
        
        print("""
提示：
1. 要使用CEVAE模型，请确保已安装了PyTorch。
2. 要运行test_cevae.py测试，请使用: pytest --runtf -vs tests/test_cevae.py
3. 要运行Cython相关代码，请确保已安装了build_ext: python setup.py build_ext --inplace
4. 如果安装过程中遇到网络问题，请尝试选择国内镜像源。
""")
    except KeyboardInterrupt:
        print("\n安装被用户中断")
    except Exception as e:
        print(f"\n安装过程中出现错误: {e}")
        print("请尝试重新运行脚本并选择不同的镜像源。")

if __name__ == "__main__":
    setup_env()
