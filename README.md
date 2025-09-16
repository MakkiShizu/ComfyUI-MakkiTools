# ComfyUI - MakkiTools

[中文介绍](README_zh.md)

self custom nodes for ComfyUI.

Sometimes I have to create simple nodes when needed, but can't find existing ones. It might just be cases of reinventing the wheel.

# 2025/08/29:Major modification

- **To ensure that future and current nodes do not conflict with other nodes, all node indices have been renamed. If a node is not found, you will need to pull the node again. I'm apologize for the inconvenience.**
- **为确保未来和当前节点不会与其他节点冲突，所有节点索引都已重命名。如果找不到节点，则需要重新拉取该节点。对于给您带来的不便，我深表歉意。**

---

## Function Introduction

### Image Manipulation

1. **GetImageNthCount**: Retrieve the Nth image from an image sequence.
2. **ImageChannelSeparate**: Separate a specified channel from an image.
3. **MergeImageChannels**: Merge different channels of an image.
4. **ImageCountConcatenate**: Concatenate multiple image batches.
5. **ImageWidthStitch**: (Deprecated) Stitch images horizontally.
6. **ImageHeigthStitch**: (Deprecated) Stitch images vertically.
7. **AnyImageStitch**: Stitch any number of images according to the specified dimension and reference size.
8. **Image_Resize**: Resize an image.
9. **Prism_Mirage**: Perform specific processing on two images.

### Video Processing

1. **AutoLoop_create_pseudo_loop_video**: (Deprecated) Convert a non - looping video into a looping video.

### Information Display

1. **Environment_INFO**: Display system environment information.
2. **show_type**: Display the type of input data.
3. **timer**: Display the running time of any process.

### Translation Functions

1. **translators**: Integrate a multi - language translation tool.
2. **translator_m2m100**: Perform multi - language translation using the m2m100 model.

### Random Number Generation

1. **random_any**: Generate random integers and floating - point numbers.

### Statistical Calculation

1. **int_calculate_statistics**: Calculate various statistics for integers.

### LoRA Loading

1. **BatchLoraLoader**: Load LoRA models in batches.

### Package Installation

1. **UniversalInstaller**: Install specified Python packages.

## How To Install

### **Recommended**

- Install via [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager).

### **Manual**

- Navigate to `ComfyUI/custom_nodes` in your terminal (cmd).
- Clone the repository under the `custom_nodes` directory using the following command:
  ```
  git clone https://github.com/MakkiShizu/ComfyUI-MakkiTools
  cd ComfyUI-MakkiTools
  ```
- Install dependencies in your Python environment.
  - For Windows Portable, run the following command inside `ComfyUI\custom_nodes\ComfyUI-MakkiTools`:
    ```
    ..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
    ```
  - If using venv or conda, activate your Python environment first, then run:
    ```
    pip install -r requirements.txt
    ```

## Notes

- Some functions are deprecated. Use them with caution.
- When using the translation functions, make sure the network connection is normal.
- After installing Python packages, you need to fully restart ComfyUI for the changes to take effect.
