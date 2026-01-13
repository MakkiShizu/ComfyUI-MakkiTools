import cv2
import numpy as np
import torch
from scipy.spatial import Delaunay
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

    return src_pts, np.roll(dst_pts, best_shift, axis=0)


def apply_affine_transform(src, src_tri, dst_tri, size):
    """
    对单个三角形区域进行仿射变换
    """
    # 给定三角形的三个顶点，计算仿射变换矩阵
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))

    # 对图像片段进行变形
    dst = cv2.warpAffine(
        src,
        warp_mat,
        (size[0], size[1]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return dst


def create_rectangular_mask(img_shape):
    """创建一个矩形遮罩，用于当没有提供遮罩时使用整个图像"""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    # 创建一个稍微小于图像边界的矩形（留一点边距）
    margin_h, margin_w = h // 20, w // 20
    mask[margin_h : h - margin_h, margin_w : w - margin_w] = 255
    return mask


def warp_image_triangulation_comfyui(
    source_img, source_mask, target_mask, n_points=100
):
    """
    基于三角剖分的图像变形 - ComfyUI适配版本
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

    # 1. 轮廓提取与对齐
    src_cnt = get_contours(source_mask_np)
    dst_cnt = get_contours(target_mask_np)
    if src_cnt is None or dst_cnt is None:
        # 如果轮廓提取失败，返回原始图像
        print("警告：轮廓提取失败，返回原始图像")
        return source_img

    src_pts = sample_points_from_contour(src_cnt, n_points)
    dst_pts = sample_points_from_contour(dst_cnt, n_points)
    src_pts, dst_pts = align_contours_robust(src_pts, dst_pts)

    # 2. 增加内部点
    src_center = np.mean(src_pts, axis=0)
    dst_center = np.mean(dst_pts, axis=0)

    src_pts = np.vstack([src_pts, src_center])
    dst_pts = np.vstack([dst_pts, dst_center])

    # 3. 目标形状的 Delaunay 三角剖分
    tri = Delaunay(dst_pts)

    # 创建输出画布
    warped_img = np.zeros_like(source_img_np)

    # 4. 遍历每一个三角形进行变形
    for simplex in tri.simplices:
        # 获取三角形的三个顶点索引
        pt1_idx, pt2_idx, pt3_idx = simplex

        # 获取源三角形和目标三角形的坐标
        tri_src = np.array([src_pts[pt1_idx], src_pts[pt2_idx], src_pts[pt3_idx]])
        tri_dst = np.array([dst_pts[pt1_idx], dst_pts[pt2_idx], dst_pts[pt3_idx]])

        # 【关键过滤】检查三角形重心是否在 Target Mask 内部
        centroid_dst = np.mean(tri_dst, axis=0)
        cx, cy = int(centroid_dst[0]), int(centroid_dst[1])
        if cx < 0 or cx >= w or cy < 0 or cy >= h or target_mask_np[cy, cx] == 0:
            continue

        # --- 以下是局部变形逻辑 ---

        # 4.1 计算三角形的边界框 (Bounding Box)
        r_src = cv2.boundingRect(np.float32([tri_src]))
        r_dst = cv2.boundingRect(np.float32([tri_dst]))

        # 4.2 从对应的边界框中裁剪出局部图像
        tri_src_cropped = []
        tri_dst_cropped = []

        for i in range(3):
            tri_src_cropped.append((tri_src[i][0] - r_src[0], tri_src[i][1] - r_src[1]))
            tri_dst_cropped.append((tri_dst[i][0] - r_dst[0], tri_dst[i][1] - r_dst[1]))

        # 裁剪源图像 patch
        img_src_crop = source_img_np[
            r_src[1] : r_src[1] + r_src[3], r_src[0] : r_src[0] + r_src[2]
        ]

        if img_src_crop.size == 0:
            continue

        # 4.3 创建目标 patch 的 mask
        mask_tri = np.zeros((r_dst[3], r_dst[2]), dtype=np.uint8)
        cv2.fillConvexPoly(
            mask_tri, np.int32(tri_dst_cropped), (1, 1, 1)
        )  # 1 表示 mask

        # 4.4 对裁剪的小块进行仿射变换
        img_warped_crop = apply_affine_transform(
            img_src_crop, tri_src_cropped, tri_dst_cropped, (r_dst[2], r_dst[3])
        )

        # 4.5 将变形后的小块，通过 mask 贴回主画布
        roi_warped = warped_img[
            r_dst[1] : r_dst[1] + r_dst[3], r_dst[0] : r_dst[0] + r_dst[2]
        ]

        # 确保维度匹配
        if (
            roi_warped.shape[:2] == mask_tri.shape
            and img_warped_crop.shape[:2] == mask_tri.shape
        ):
            # 更新 ROI：只更新 mask 覆盖的区域
            if len(roi_warped.shape) == 3 and len(img_warped_crop.shape) == 3:
                # 彩色图像
                for c in range(roi_warped.shape[2]):
                    roi_warped[:, :, c][mask_tri == 1] = img_warped_crop[:, :, c][
                        mask_tri == 1
                    ]
            else:
                # 灰度图像
                roi_warped[mask_tri == 1] = img_warped_crop[mask_tri == 1]
        else:
            print(
                f"尺寸不匹配: roi_warped {roi_warped.shape}, img_warped_crop {img_warped_crop.shape}, mask_tri {mask_tri.shape}"
            )

    # 将numpy数组转换为PIL Image，然后再转换为tensor
    # 确保数值在[0,1]范围内
    warped_img = np.clip(warped_img, 0, 1)

    # 转换为PIL Image
    if warped_img.shape[-1] == 1:
        # 灰度图
        pil_img = pil2tensor(
            Image.fromarray((warped_img.squeeze(-1) * 255).astype(np.uint8), mode="L")
        )
    else:
        # RGB或其他多通道图像
        pil_img = pil2tensor(Image.fromarray((warped_img * 255).astype(np.uint8)))

    return pil_img
