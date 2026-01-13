import cv2
import numpy as np
import torch
from scipy.interpolate import Rbf
from .convert_tensor import tensor2pil, pil2tensor
from PIL import Image


def get_contours(img):
    """提取最大轮廓"""
    # 确保图像是8位单通道格式
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 将图像转换为uint8格式
    if gray.dtype != np.uint8:
        # 如果是浮点型，先将其标准化到[0,1]或[0,255]范围
        if gray.max() <= 1.0:
            gray = (gray * 255).astype(np.uint8)
        else:
            gray = gray.astype(np.uint8)

    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contours = [c for c in contours if cv2.contourArea(c) > 500]
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    return c.reshape(-1, 2)


def sample_points_from_contour(contour, n_points=50):
    """均匀采样轮廓点"""
    diffs = np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
    diffs[diffs < 1e-6] = 1e-6
    cumulative_dist = np.concatenate(([0], np.cumsum(diffs)))
    perimeter = cumulative_dist[-1]
    if perimeter < 1e-6:
        return np.tile(contour[0], (n_points, 1)).astype(np.float32)
    target_dists = np.linspace(0, perimeter, n_points + 1)[:-1]
    x_interp = np.interp(target_dists, cumulative_dist, contour[:, 0])
    y_interp = np.interp(target_dists, cumulative_dist, contour[:, 1])
    return np.column_stack((x_interp, y_interp)).astype(np.float32)


def align_contours_robust(src_pts, dst_pts):
    """鲁棒对齐：暴力搜索最佳 shift"""
    n = len(src_pts)
    if n == 0:
        return src_pts, dst_pts
    best_shift = 0
    min_dist = float("inf")
    src_center = np.mean(src_pts, axis=0)
    dst_center = np.mean(dst_pts, axis=0)
    src_norm = src_pts - src_center
    dst_norm = dst_pts - dst_center
    src_scale = np.mean(np.linalg.norm(src_norm, axis=1)) + 1e-9
    dst_scale = np.mean(np.linalg.norm(dst_norm, axis=1)) + 1e-9
    src_norm /= src_scale
    dst_norm /= dst_scale
    for shift in range(n):
        rolled_dst = np.roll(dst_norm, shift, axis=0)
        dist = np.sum(np.linalg.norm(src_norm - rolled_dst, axis=1))
        if dist < min_dist:
            min_dist = dist
            best_shift = shift
    aligned_dst = np.roll(dst_pts, best_shift, axis=0)
    return src_pts, aligned_dst


def warp_image_tps_comfyui(
    source_img, source_mask, target_mask, n_points=150, smooth=0.1
):
    """
    TPS 变形主函数 (加入了最终强制 Mask 步骤)
    """
    # 将PyTorch tensors转换为PIL Images，然后再转换为numpy arrays
    source_img_pil = tensor2pil(source_img)
    source_mask_pil = tensor2pil(source_mask)
    target_mask_pil = tensor2pil(target_mask)

    # 转换为numpy数组
    source_img_np = np.array(source_img_pil).astype(np.float32) / 255.0
    source_mask_np = np.array(source_mask_pil.convert("L"))  # 转换为灰度图
    target_mask_np = np.array(target_mask_pil.convert("L"))  # 转换为灰度图

    # 确保所有图像尺寸一致（以目标遮罩尺寸为准）
    target_h, target_w = target_mask_np.shape
    if source_img_np.shape[0] != target_h or source_img_np.shape[1] != target_w:
        # 调整源图像尺寸
        source_img_np = cv2.resize(
            source_img_np, (target_w, target_h), interpolation=cv2.INTER_CUBIC
        )

    if source_mask_np.shape[0] != target_h or source_mask_np.shape[1] != target_w:
        # 调整源遮罩尺寸
        source_mask_np = cv2.resize(
            source_mask_np, (target_w, target_h), interpolation=cv2.INTER_CUBIC
        )

    h, w = target_mask_np.shape[:2]

    # 1. 获取并采样轮廓
    src_cnt = get_contours(source_mask_np)
    dst_cnt = get_contours(target_mask_np)
    if src_cnt is None or dst_cnt is None:
        # 如果轮廓提取失败，返回原始图像
        print("警告：轮廓提取失败，返回原始图像")
        return source_img

    # 增加采样点以更好地应对复杂形状
    src_pts = sample_points_from_contour(src_cnt, n_points)
    dst_pts = sample_points_from_contour(dst_cnt, n_points)

    # 2. 对齐
    src_pts, dst_pts = align_contours_robust(src_pts, dst_pts)

    # 3. 锚点
    corners = np.array(
        [
            [0, 0],
            [w // 2, 0],
            [w - 1, 0],
            [w - 1, h // 2],
            [w - 1, h - 1],
            [w // 2, h - 1],
            [0, h - 1],
            [0, h / 2],
        ]
    )
    src_final = np.vstack([src_pts, corners])
    dst_final = np.vstack([dst_pts, corners])

    # 4. TPS 计算
    # 【调整】对于这种极端变形，减少 smooth 值，让 TPS 尽量贴合边界点
    # 哪怕这会让内部纹理拉伸得更厉害，但我们需要纹理尽量到达边界
    current_smooth = smooth
    try:
        rbf_x = Rbf(
            dst_final[:, 0],
            dst_final[:, 1],
            src_final[:, 0],
            function="thin_plate",
            smooth=current_smooth,
        )
        rbf_y = Rbf(
            dst_final[:, 0],
            dst_final[:, 1],
            src_final[:, 1],
            function="thin_plate",
            smooth=current_smooth,
        )
    except np.linalg.LinAlgError:
        print("警告：矩阵奇异，增加平滑度")
        current_smooth += 1.0
        rbf_x = Rbf(
            dst_final[:, 0],
            dst_final[:, 1],
            src_final[:, 0],
            function="thin_plate",
            smooth=current_smooth,
        )
        rbf_y = Rbf(
            dst_final[:, 0],
            dst_final[:, 1],
            src_final[:, 1],
            function="thin_plate",
            smooth=current_smooth,
        )

    # 5. 生成映射
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    flat_x = grid_x.flatten()
    flat_y = grid_y.flatten()
    map_x = rbf_x(flat_x, flat_y).reshape(h, w).astype(np.float32)
    map_y = rbf_y(flat_x, flat_y).reshape(h, w).astype(np.float32)

    # 6. 重采样 (得到初步变形结果，边缘可能是乱的)
    # 需要将源图像转换为0-255的uint8格式才能使用cv2.remap
    source_img_uint8 = (source_img_np * 255).astype(np.uint8)
    warped_raw = cv2.remap(
        source_img_uint8,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # ============================================================
    # 【新增步骤 7】 强制应用目标 Mask (Hard Masking)
    # ============================================================
    # 确保 target_mask 是干净的二值图像 (0 或 255)
    _, binary_mask = cv2.threshold(target_mask_np, 127, 255, cv2.THRESH_BINARY)

    # 将单通道 mask 转为三通道，以便与彩色图像进行运算
    mask_3ch = cv2.merge([binary_mask, binary_mask, binary_mask])

    # 使用按位与操作：Mask 为黑色的地方，结果强制变黑；Mask 为白色的地方，保留变形后的纹理
    final_result = cv2.bitwise_and(warped_raw, mask_3ch)

    # 将结果转换回tensor格式
    # 转换为PIL Image
    if final_result.shape[-1] == 1:
        # 灰度图
        pil_img = pil2tensor(Image.fromarray(final_result.squeeze(-1), mode="L"))
    else:
        # RGB或其他多通道图像
        pil_img = pil2tensor(Image.fromarray(final_result))

    return pil_img
