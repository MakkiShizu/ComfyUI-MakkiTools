# ComfyUI - MakkiTools

[English](README.md)

自用的 ComfyUI 自定义节点。

有时我需要根据需求创建简单的节点，但找不到现有的节点。可能只是重复造轮子的情况。

# 2025/08/29:重大修改

- **To ensure that future and current nodes do not conflict with other nodes, all node indices have been renamed. If a node is not found, you will need to pull the node again. I'm apologize for the inconvenience.**
- **为确保未来和当前节点不会与其他节点冲突，所有节点索引都已重命名。如果找不到节点，则需要重新拉取该节点。对于给您带来的不便，我深表歉意。**

---

## 功能介绍

### 图像操作

1. **GetImageNthCount**：获取图像序列中的第 N 张图像。
2. **ImageChannelSeparate**：分离图像的指定通道。
3. **MergeImageChannels**：合并图像的不同通道。
4. **ImageCountConcatenate**：拼接多个图像批次。
5. **ImageWidthStitch**：（已弃用）横向拼接图像。
6. **ImageHeigthStitch**：（已弃用）纵向拼接图像。
7. **AnyImageStitch**：可按指定维度和参考尺寸拼接任意数量的图像。
8. **Image_Resize**：调整图像的大小。
9. **Prism_Mirage**：对两张图像进行特定处理。

### 视频处理

1. **AutoLoop_create_pseudo_loop_video**：（已弃用）将非循环视频转换为循环视频。

### 信息显示

1. **Environment_INFO**：显示系统的环境信息。
2. **show_type**：显示输入数据的类型。
3. **timer**：显示任意流程运行时间。

### 翻译功能

1. **translators**：集成多语言翻译工具。
2. **translator_m2m100**：使用 m2m100 模型进行多语言翻译。

### 随机数生成

1. **random_any**：生成随机整数和浮点数。

### 统计计算

1. **int_calculate_statistics**：计算整数的各种统计量。

### LoRA 加载

1. **BatchLoraLoader**：批量加载 LoRA 模型。

### 包安装

1. **UniversalInstaller**：安装指定的 Python 包。

## 如何安装

### **推荐**

- 通过 [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) 安装。

### **手动**

- 在终端（cmd）中导航到 `ComfyUI/custom_nodes`。
- 使用以下命令在 `custom_nodes` 目录下克隆存储库：
  ```
  git clone https://github.com/MakkiShizu/ComfyUI-MakkiTools
  cd ComfyUI-MakkiTools
  ```
- 在你的 Python 环境中安装依赖项。
  - 对于 Windows 便携版，在 `ComfyUI\custom_nodes\ComfyUI-MakkiTools`内运行以下命令：
    ```
    ..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
    ```
  - 如果使用 venv 或 conda，请先激活您的 Python 环境，然后运行：
    ```
    pip install -r requirements.txt
    ```

## 注意事项

- 部分功能已弃用，请谨慎使用。
- 在使用翻译功能时，可能需要确保网络连接正常。
- 安装 Python 包后，需要完全重启 ComfyUI 以使更改生效。
