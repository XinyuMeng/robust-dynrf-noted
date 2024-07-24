# Copyright (c) Meta Platforms, Inc. and affiliates.

import cv2
import numpy as np
import scipy.signal
import torch
import torchvision.transforms as T
from PIL import Image

mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]))


def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = np.nan_to_num(depth)  # 将NaN值替换为0，确保数据类型正确
    if minmax is None:
        mi = np.min(x[x > 0])  # 获取最小正深度值（忽略背景）
        ma = np.max(x)  # 获取最大深度值
    else:
        mi, ma = minmax  # 使用传入的最小最大值

    x = (x - mi) / (ma - mi + 1e-8)  # 将深度值归一化到0~1范围，避免除零错误
    x = (255 * x).astype(np.uint8)  # 将归一化的深度值转换为0~255的整数
    x_ = cv2.applyColorMap(x, cmap)  # 应用颜色映射，转换为彩色深度图
    return x_, [mi, ma]  # 返回彩色深度图和深度值的范围


def init_log(log, keys):
    for key in keys:
        log[key] = torch.tensor([0.0], dtype=float)
    return log


def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth)  # change nan to 0
    if minmax is None:
        mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = minmax

    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_, [mi, ma]

# 将体素数转换为分辨率
def N_to_reso(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / 3)  # 计算体素大小
    return ((xyz_max - xyz_min) / voxel_size).long().tolist() # 返回体素分辨率列表


def cal_n_samples(reso, step_ratio=0.5):
    return int(np.linalg.norm(reso) / step_ratio)


__LPIPS__ = {}


def init_lpips(net_name, device):
    assert net_name in ["alex", "vgg"]
    import lpips

    print(f"init_lpips: lpips_{net_name}")
    return lpips.LPIPS(net=net_name, version="0.1").eval().to(device)


def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()


def findItem(items, target):
    for one in items:
        if one[: len(target)] == target:
            return one
    return None


""" Evaluation metrics (ssim, lpips)
"""


def rgb_ssim(
    img0,
    img1,
    max_val,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    return_map=False,
):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode="valid")

    filt_fn = lambda z: np.stack(
        [
            convolve2d(convolve2d(z[..., i], filt[:, None]), filt[None, :])
            for i in range(z.shape[-1])
        ],
        -1,
    )
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0.0, sigma00)
    sigma11 = np.maximum(0.0, sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


import torch.nn as nn


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x, ignore_axis=None):
        # x: [1, 96, xyzt, 1]
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        if ignore_axis is None:
            return (
                self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
            )  # TODO: this line causing NaN due to count_h or count_w == 0
        elif ignore_axis == "h":
            return self.TVLoss_weight * 2 * (w_tv / count_w) / batch_size
        elif ignore_axis == "w":
            return self.TVLoss_weight * 2 * (h_tv / count_h) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


import plyfile
import skimage.measure


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,  # 传入的 PyTorch 张量，表示 3D 的 SDF（Signed Distance Function）样本
    ply_filename_out,  # 输出的 PLY 文件路径
    bbox,  # 包围盒，格式为二维数组，包含了最小和最大边界
    level=0.5,  # SDF 函数的等值面（isosurface）水平值，默认为 0.5
    offset=None,  # 额外的偏移量，默认为 None
    scale=None,  # 额外的缩放比例，默认为 None
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    # 将 PyTorch 张量转换为 NumPy 数组
    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    # 计算体素的尺寸
    voxel_size = list((bbox[1] - bbox[0]) / np.array(pytorch_3d_sdf_tensor.shape))

    # 使用 Marching Cubes 算法提取等值面
    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=voxel_size
    )
    # 翻转面的方向，以符合 PLY 文件格式的要求
    faces = faces[..., ::-1]  # inverse face orientation

    # 将顶点从体素坐标系转换为相机坐标系
    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = bbox[0, 0] + verts[:, 0]
    mesh_points[:, 1] = bbox[0, 1] + verts[:, 1]
    mesh_points[:, 2] = bbox[0, 2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]  # 顶点数量
    num_faces = faces.shape[0]  # 面数量

    # 构建顶点元组：创建了一个结构化的 NumPy 数组，用于存储顶点的坐标信息，每个顶点都有 x、y 和 z 三个坐标值，均初始化为零
    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    # 构建面元组
    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    # 创建 PLY 数据
    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])

    # 将数据写入 PLY 文件
    print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)
