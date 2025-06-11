import os
import platform
import sys
import subprocess
import psutil
import folder_paths
from datetime import datetime


class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


def get_environment_info():
    """收集并返回环境信息字典"""
    env_info = {}

    # 基础系统信息
    env_info["Date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    env_info["Platform"] = platform.platform()
    env_info["System"] = platform.system()
    env_info["Processor"] = platform.processor()
    env_info["Python Version"] = sys.version.replace("\n", " ")

    # CPU信息
    env_info["CPU Cores (Physical)"] = psutil.cpu_count(logical=False) or "N/A"
    env_info["CPU Cores (Logical)"] = psutil.cpu_count(logical=True) or "N/A"
    env_info["CPU Usage (%)"] = psutil.cpu_percent(interval=1)

    # 内存信息
    mem = psutil.virtual_memory()
    env_info["RAM Total (GB)"] = round(mem.total / (1024**3), 2)
    env_info["RAM Available (GB)"] = round(mem.available / (1024**3), 2)
    env_info["RAM Used (%)"] = mem.percent

    # GPU信息
    try:
        nvidia_smi = (
            subprocess.check_output(
                "nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits",
                shell=True,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
            .split("\n")
        )

        gpus = []
        for i, gpu in enumerate(nvidia_smi):
            name, driver, mem_total, mem_used, gpu_util = gpu.split(", ")
            gpus.append(
                {
                    "GPU Index": i,
                    "Name": name,
                    "Driver Version": driver,
                    "VRAM Total (GB)": round(float(mem_total) / 1024, 2),
                    "VRAM Used (GB)": round(float(mem_used) / 1024, 2),
                    "GPU Utilization (%)": gpu_util,
                }
            )
        env_info["GPUs"] = gpus
    except (subprocess.CalledProcessError, FileNotFoundError, UnicodeDecodeError):
        env_info["GPUs"] = "No NVIDIA GPU detected"

    # 深度学习框架信息
    try:
        import torch

        env_info["PyTorch Version"] = torch.__version__
        env_info["CUDA Available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env_info["CUDA Version"] = torch.version.cuda
            env_info["cuDNN Version"] = torch.backends.cudnn.version()
    except ImportError:
        env_info["PyTorch Version"] = "Not installed"

    # 获取所有已安装的包及其版本
    env_info["all_packages"] = get_all_installed_packages()

    # 获取custom_nodes文件夹名称
    env_info["custom_nodes"] = get_custom_nodes_folders()

    return env_info


def get_all_installed_packages():
    """获取所有已安装的包及其版本信息"""
    packages = {}

    try:
        # Python 3.8+ 使用 importlib.metadata
        from importlib import metadata as importlib_metadata

        distributions = importlib_metadata.distributions()
        for dist in distributions:
            try:
                name = dist.metadata["Name"]
                version = dist.version
                if name:  # 确保包名不为空
                    packages[name] = version
            except KeyError:
                continue
    except ImportError:
        try:
            # 回退到 pkg_resources (Python <3.8)
            import pkg_resources

            for dist in pkg_resources.working_set:
                packages[dist.key] = dist.version
        except ImportError:
            # 最后尝试使用 pip list 命令
            try:
                result = subprocess.run(
                    ["pip", "list", "--format=freeze"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                for line in result.stdout.splitlines():
                    if "==" in line:
                        name, version = line.split("==", 1)
                        packages[name] = version
            except Exception as e:
                return f"Failed to get packages: {str(e)}"

    return packages


def get_custom_nodes_folders():
    """获取custom_nodes的所有文件夹名称"""
    try:
        # 获取自定义节点路径
        node_paths = folder_paths.get_folder_paths("custom_nodes")
        all_node_names = []
        for custom_node_path in node_paths:
            possible_modules = os.listdir(custom_node_path)

            for possible_module in possible_modules:
                module_path = os.path.join(custom_node_path, possible_module)
                if (
                    os.path.isfile(module_path)
                    or module_path.endswith(".disabled")
                    or module_path == "__pycache__"
                ):
                    continue

                # 提取节点名称（目录的最后一级）
                node_name = os.path.basename(module_path)
                all_node_names.append(node_name)
        return all_node_names
    except Exception as e:
        return f"Failed to get folders: {e}"


def format_environment_info(
    info,
    SYSTEM_INFO,
    HARDWARE_INFO,
    GPU_INFO,
    DEEP_LEARNING_FRAMEWORKS_INFO,
    ALL_INSTALLED_PACKAGES_INFO,
    CUSTOM_NODES_FOLDERS_INFO,
):
    """将环境信息格式化为单个字符串"""
    output = []
    output.append("=" * 60)
    output.append("ENVIRONMENT INFO")
    output.append("=" * 60)

    # 系统信息
    if SYSTEM_INFO:
        output.append("\n[SYSTEM INFORMATION]")
        sys_info = [
            f"Date: {info['Date']}",
            f"Platform: {info['Platform']}",
            f"System: {info['System']}",
            f"Processor: {info['Processor']}",
            f"Python Version: {info['Python Version']}",
        ]
        output.extend(sys_info)

    # 硬件信息
    if HARDWARE_INFO:
        output.append("\n[HARDWARE INFORMATION]")
        hw_info = [
            f"Physical CPU Cores: {info['CPU Cores (Physical)']}",
            f"Logical CPU Cores: {info['CPU Cores (Logical)']}",
            f"CPU Usage: {info['CPU Usage (%)']}%",
            f"RAM Total: {info['RAM Total (GB)']} GB",
            f"RAM Available: {info['RAM Available (GB)']} GB",
            f"RAM Used: {info['RAM Used (%)']}%",
        ]
        output.extend(hw_info)

    # GPU信息
    if GPU_INFO:
        output.append("\n[GPU INFORMATION]")
        if isinstance(info["GPUs"], list):
            for gpu in info["GPUs"]:
                output.append(f"GPU {gpu['GPU Index']}:")
                output.append(f"  Name: {gpu['Name']}")
                output.append(f"  Driver Version: {gpu['Driver Version']}")
                output.append(f"  VRAM Total: {gpu['VRAM Total (GB)']} GB")
                output.append(f"  VRAM Used: {gpu['VRAM Used (GB)']} GB")
                output.append(f"  GPU Utilization: {gpu['GPU Utilization (%)']}%")
        else:
            output.append(f"GPU Status: {info['GPUs']}")

    # 深度学习框架
    if DEEP_LEARNING_FRAMEWORKS_INFO:
        output.append("\n[DEEP LEARNING FRAMEWORKS]")
        dl_info = [
            f"PyTorch Version: {info.get('PyTorch Version', 'Not installed')}",
            f"CUDA Available: {'Yes' if info.get('CUDA Available', False) else 'No'}",
        ]
        if info.get("CUDA Version"):
            dl_info.append(f"CUDA Version: {info['CUDA Version']}")
            dl_info.append(f"cuDNN Version: {info['cuDNN Version']}")
        output.extend(dl_info)

    # 所有已安装的包
    if ALL_INSTALLED_PACKAGES_INFO:
        output.append("\n[ALL INSTALLED PACKAGES]")
        if isinstance(info["all_packages"], dict):
            # 按包名排序
            for name, version in sorted(info["all_packages"].items()):
                output.append(f"{name}=={version}")
            output.append(f"\nTotal packages: {len(info['all_packages'])}")
        else:
            output.append(f"Error: {info['all_packages']}")

    # 自定义节点文件夹
    if CUSTOM_NODES_FOLDERS_INFO:
        output.append("\n[CUSTOM NODES FOLDERS]")
        folders = info.get("custom_nodes", [])
        if isinstance(folders, list):
            for folder in folders:
                output.append(f"{folder}")
        else:
            output.append(f"{folders}")
        output.append(f"\nTotal custom nodes: {len(folders)}")

    # 添加结束分隔线
    output.append("\n" + "=" * 60)

    # 将所有内容合并为一个字符串
    return "\n".join(output)
