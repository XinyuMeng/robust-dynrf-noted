# Copyright (c) Meta Platforms, Inc. and affiliates.

import datetime
import os
import sys
from typing import List
import io
import imageio
import numpy as np
import torch
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch_efficient_distloss import (
    eff_distloss,
    eff_distloss_native,
    flatten_eff_distloss,
)
from utils import visualize_depth_numpy


from camera import (
    pose_to_mtx,
    cam2world,
    lie,
    pose,
    procrustes_analysis,
    rotation_distance,
    get_novel_view_poses,
)

from dataLoader import dataset_dict
from dataLoader.ray_utils import (
    get_ray_directions_blender,
    get_ray_directions_lean,
    get_rays,
    get_rays_lean,
    get_rays_with_batch,
    ndc_rays_blender,
    ndc_rays_blender2,
)
from models.tensoRF import TensorVMSplit, TensorVMSplit_TimeEmbedding
from opt import config_parser
from renderer import (
    evaluation,
    evaluation_path,
    OctreeRender_trilinear_fast,
    render,
    induce_flow,
    render_3d_point,
    render_single_3d_point,
    NDC2world,
    induce_flow_single,
    raw2outputs,
    sampleXYZ,
    contract2world,
)
from utils import cal_n_samples, convert_sdf_samples_to_ply, N_to_reso, TVLoss
from flow_viz import flow_to_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast


# Dummy tensorboard logger
class DummyWriter:
    def add_scalar(*args, **kwargs):
        pass

    def add_images(*args, **kwargs):
        pass


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr : self.curr + self.batch]


def ids2pixel(W, H, ids):
    """
    Regress pixel coordinates from

    Args:
        W (int): 图像宽度
        H (int): 图像高度
        ids (Tensor): 样本索引张量

    Returns:
        tuple: 包含像素列坐标、行坐标和视图ID的元组
    """
    # 计算像素列坐标
    col = ids % W
    # 计算像素行坐标
    row = (ids // W) % H
    # 计算视图ID
    view_ids = ids // (W * H)
    return col, row, view_ids


@torch.no_grad()
def export_mesh(args):
    # 加载检查点
    ckpt = None
    ckpt = torch.load(args.ckpt, map_location=device)
    # 从检查点中获取参数
    kwargs = ckpt["kwargs"]
    kwargs.update({"device": device})
    # 根据模型名称和参数创建模型对象
    tensorf = eval(args.model_name)(**kwargs)
    # 加载模型权重
    tensorf.load(ckpt)

    # 获取密度体积和法向量
    alpha, _ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(
        alpha.cpu(),  # 将密度体积转移到CPU上
        f"{args.ckpt[:-3]}.ply",  # 输出文件路径，去掉扩展名的ckpt，并使用ply扩展名
        bbox=tensorf.aabb.cpu(),  # 包围盒信息，也转移到CPU上
        level=0.005  # 网格化时的阈值
    )


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


# from DynDyn
def generate_path(c2w, focal, sc, length=None):
    # hwf = c2w[:, 4:5]
    num_novelviews = 60
    max_disp = 48.0
    # H, W, focal = hwf[:, 0]
    # downsample = 2.0
    # focal = (854 / 2 * np.sqrt(3)) / float(downsample)

    max_trans = max_disp / focal[0] * sc
    dolly_poses = []
    dolly_focals = []

    # Dolly zoom
    for i in range(30):
        x_trans = 0.0
        y_trans = 0.0
        z_trans = max_trans * 2.5 * i / float(30 // 2)
        i_pose = np.concatenate(
            [
                np.concatenate(
                    [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]],
                    axis=1,
                ),
                np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
            ],
            axis=0,
        )
        i_pose = np.linalg.inv(i_pose)
        ref_pose = np.concatenate(
            [c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0
        )
        render_pose = np.dot(ref_pose, i_pose)
        dolly_poses.append(render_pose[:3, :])
        new_focal = focal[0] - focal[0] * 0.1 * z_trans / max_trans / 2.5
        dolly_focals.append(new_focal)
    dolly_poses = np.stack(dolly_poses, 0)[:, :3]

    zoom_poses = []
    zoom_focals = []
    # Zoom in
    for i in range(30):
        x_trans = 0.0
        y_trans = 0.0
        # z_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_novelviews)) * args.z_trans_multiplier
        z_trans = max_trans * 2.5 * i / float(30 // 2)
        i_pose = np.concatenate(
            [
                np.concatenate(
                    [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]],
                    axis=1,
                ),
                np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
            ],
            axis=0,
        )

        i_pose = np.linalg.inv(i_pose)  # torch.tensor(np.linalg.inv(i_pose)).float()

        ref_pose = np.concatenate(
            [c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0
        )

        render_pose = np.dot(ref_pose, i_pose)
        zoom_poses.append(render_pose[:3, :])
        zoom_focals.append(focal[0])
    zoom_poses = np.stack(zoom_poses, 0)[:, :3]

    spiral_poses = []
    spiral_focals = []
    # Rendering teaser. Add translation.
    for i in range(30):
        x_trans = max_trans * 1.5 * np.sin(2.0 * np.pi * float(i) / float(30)) * 2.0
        y_trans = (
            max_trans
            * 1.5
            * (np.cos(2.0 * np.pi * float(i) / float(30)) - 1.0)
            * 2.0
            / 3.0
        )
        z_trans = 0.0

        i_pose = np.concatenate(
            [
                np.concatenate(
                    [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]],
                    axis=1,
                ),
                np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
            ],
            axis=0,
        )

        i_pose = np.linalg.inv(i_pose)

        ref_pose = np.concatenate(
            [c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0
        )

        render_pose = np.dot(ref_pose, i_pose)
        # output_poses.append(np.concatenate([render_pose[:3, :], hwf], 1))
        spiral_poses.append(render_pose[:3, :])
        spiral_focals.append(focal[0])
    spiral_poses = np.stack(spiral_poses, 0)[:, :3]

    fix_view_poses = []
    fix_view_focals = []
    # fix view
    for i in range(length):
        render_pose = np.concatenate(
            [c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0
        )
        # output_poses.append(np.concatenate([render_pose[:3, :], hwf], 1))
        fix_view_poses.append(render_pose[:3, :])
        fix_view_focals.append(focal[0])
    fix_view_poses = np.stack(fix_view_poses, 0)[:, :3]

    change_view_time_poses = []
    change_view_time_focals = []
    # Rendering teaser. Add translation.
    for i in range(length):
        x_trans = max_trans * 1.5 * np.sin(2.0 * np.pi * float(i) / float(30)) * 2.0
        y_trans = (
            max_trans
            * 1.5
            * (np.cos(2.0 * np.pi * float(i) / float(30)) - 1.0)
            * 2.0
            / 3.0
        )
        z_trans = 0.0

        i_pose = np.concatenate(
            [
                np.concatenate(
                    [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]],
                    axis=1,
                ),
                np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
            ],
            axis=0,
        )

        i_pose = np.linalg.inv(i_pose)

        ref_pose = np.concatenate(
            [c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0
        )

        render_pose = np.dot(ref_pose, i_pose)
        # output_poses.append(np.concatenate([render_pose[:3, :], hwf], 1))
        change_view_time_poses.append(render_pose[:3, :])
        change_view_time_focals.append(focal[0])
    change_view_time_poses = np.stack(change_view_time_poses, 0)[:, :3]

    return (
        dolly_poses,
        dolly_focals,
        zoom_poses,
        zoom_focals,
        spiral_poses,
        spiral_focals,
        fix_view_poses,
        fix_view_focals,
        change_view_time_poses,
        change_view_time_focals,
    )


# from DynDyn
def generate_follow_spiral(c2ws, focal, sc):
    num_novelviews = int(c2ws.shape[0] * 2)
    max_disp = 48.0 * 2

    max_trans = max_disp / focal[0] * sc
    output_poses = []
    output_focals = []

    # Rendering teaser. Add translation.
    for i in range(c2ws.shape[0]):
        x_trans = (
            max_trans
            * np.sin(2.0 * np.pi * float(i) / float(num_novelviews) * 4.0)
            * 1.0
        )
        y_trans = (
            max_trans
            * (np.cos(2.0 * np.pi * float(i) / float(num_novelviews) * 4.0) - 1.0)
            * 0.33
        )
        z_trans = 0.0

        i_pose = np.concatenate(
            [
                np.concatenate(
                    [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]],
                    axis=1,
                ),
                np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
            ],
            axis=0,
        )

        i_pose = np.linalg.inv(i_pose)

        ref_pose = np.concatenate(
            [c2ws[i, :3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0
        )

        render_pose = np.dot(ref_pose, i_pose)
        # output_poses.append(np.concatenate([render_pose[:3, :], hwf], 1))
        output_poses.append(render_pose[:3, :])
    # backward
    for i in range(c2ws.shape[0]):
        x_trans = (
            max_trans
            * np.sin(2.0 * np.pi * float(i) / float(num_novelviews) * 2.0)
            * 1.0
        )
        y_trans = (
            max_trans
            * (np.cos(2.0 * np.pi * float(i) / float(num_novelviews) * 2.0) - 1.0)
            * 0.33
        )
        z_trans = 0.0

        i_pose = np.concatenate(
            [
                np.concatenate(
                    [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]],
                    axis=1,
                ),
                np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
            ],
            axis=0,
        )

        i_pose = np.linalg.inv(i_pose)

        ref_pose = np.concatenate(
            [
                c2ws[c2ws.shape[0] - 1 - i, :3, :4],
                np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
            ],
            axis=0,
        )

        render_pose = np.dot(ref_pose, i_pose)
        output_poses.append(render_pose[:3, :])
    return output_poses


@torch.no_grad()
def render_test(args, logfolder):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    # 初始化测试集数据对象
    test_dataset = dataset(
        args.datadir,  # 数据集的根目录
        split="test",  # 使用测试集
        downsample=args.downsample_train,  # 是否下采样
        is_stack=True,  # 是否为堆叠图像
        use_disp=args.use_disp,  # 是否使用位移图
        use_foreground_mask=args.use_foreground_mask,  # 是否使用前景遮罩
    )
    # 获取测试数据集的背景颜色信息
    white_bg = test_dataset.white_bg
    # 设置射线类型
    ray_type = args.ray_type

    if not os.path.exists(args.ckpt):
        raise RuntimeError("the ckpt path does not exists!!")

    # dynamic
    # 加载模型的检查点文件
    ckpt = torch.load(args.ckpt, map_location=device)
    # 从检查点中获取参数字典
    kwargs = ckpt["kwargs"]
    # 从参数字典中弹出相机位姿信息，并将其移到指定的设备上
    poses_mtx = kwargs.pop("se3_poses").to(device)
    # 从参数字典中弹出焦距微调信息，并将其移到指定的设备上
    focal_refine = kwargs.pop("focal_ratio_refine").to(device)
    # 更新参数字典，添加设备信息
    kwargs.update({"device": device})
    # 根据模型名称和参数字典创建模型对象
    tensorf = eval(args.model_name)(**kwargs)
    # 加载模型的权重
    tensorf.load(ckpt)
    # static
    # 加载静态模型的检查点文件
    ckpt_static = torch.load(args.ckpt[:-3] + "_static.th", map_location=device)
    # 从静态模型的检查点中获取参数字典
    kwargs_static = ckpt_static["kwargs"]
    # 从参数字典中弹出相机位姿信息，并将其移到指定的设备上
    poses_mtx = kwargs_static.pop("se3_poses").to(device)
    # 从参数字典中弹出焦距微调信息，并将其移到指定的设备上
    focal_refine = kwargs_static.pop("focal_ratio_refine").to(device)
    # 更新参数字典，添加设备信息
    kwargs_static.update({"device": device})
    # 根据模型名称和参数字典创建静态模型对象
    tensorf_static = TensorVMSplit(**kwargs_static)
    # 加载静态模型的权重
    tensorf_static.load(ckpt_static)
    # 创建保存测试集图片的文件夹
    os.makedirs(f"{logfolder}/{args.expname}/imgs_test_all", exist_ok=True)
    # 保存相机位姿信息为.npy文件
    np.save(f"{logfolder}/{args.expname}/poses.npy", poses_mtx.detach().cpu().numpy())
    # 保存焦距微调信息为.npy文件
    np.save(
        f"{logfolder}/{args.expname}/focal.npy", focal_refine.detach().cpu().numpy()
    )

    if args.render_train:
        # 如果需要渲染训练集
        os.makedirs(f"{logfolder}/imgs_train_all", exist_ok=True)
        # 创建一个目录，用于保存训练集的图像数据，如果目录已存在则不做任何操作
        train_dataset = dataset(
            args.datadir, split="train", downsample=args.downsample_train, is_stack=True
        )
        # 初始化一个训练集数据集对象，包括设置数据集路径、拆分为训练集、指定是否进行下采样、设置是否为堆叠数据
        # 调用evaluation函数评估训练集的性能，计算PSNR等指标
        PSNRs_test, _ = evaluation(
            train_dataset,
            poses_mtx,
            tensorf,
            args,
            renderer,
            f"{logfolder}/imgs_train_all",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ray_type=args.ray_type,
            device=device,
        )
        print(
            f"======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================"
        )

    if args.render_test:
        os.makedirs(f"{logfolder}/{args.expname}/imgs_test_all", exist_ok=True)
        _, near_fars, depth_test_all = evaluation(
            test_dataset,
            poses_mtx,
            focal_refine.cpu(),
            tensorf_static,
            tensorf,
            args,
            renderer,
            f"{logfolder}/{args.expname}/imgs_test_all",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ray_type=args.ray_type,
            device=device,
        )

    if args.render_path:
        SE3_poses = poses_mtx  # 获取姿势矩阵

        # 计算均值姿势和选择最佳渲染姿势
        mean_pose = torch.mean(poses_mtx[:, :, 3], 0)
        render_idx = 0
        best_dist = 1000000000
        for iidx in range(SE3_poses.shape[0]):
            cur_dist = torch.mean((SE3_poses[iidx, :, 3] - mean_pose) ** 2)
            if cur_dist < best_dist:
                best_dist = cur_dist
                render_idx = iidx
        print(render_idx)  # 打印最佳渲染索引
        sc = near_fars[render_idx][0] * 0.75  # 计算缩放因子
        c2w = SE3_poses.cpu().detach().numpy()[render_idx]  # 获取相机到世界坐标系的变换矩阵

        # 获取平均姿势
        up_m = normalize(SE3_poses.cpu().detach().numpy()[:, :3, 1].sum(0))
        # 生成渲染路径
        (
            dolly_poses,
            dolly_focals,
            zoom_poses,
            zoom_focals,
            spiral_poses,
            spiral_focals,
            fix_view_poses,
            fix_view_focals,
            change_view_time_poses,
            change_view_time_focals,
        ) = generate_path(
            SE3_poses.cpu().detach().numpy()[render_idx],
            focal=[focal_refine.item(), focal_refine.item()],
            sc=sc,
            length=SE3_poses.shape[0],
        )

        # fix view, change time
        os.makedirs(f"{logfolder}/{args.expname}/fix_view", exist_ok=True)  # 创建固定视角文件夹
        _, depth_fix_view_all = evaluation_path(
            test_dataset,
            focal_refine.cpu(),
            tensorf_static,
            tensorf,
            fix_view_poses,
            renderer,
            f"{logfolder}/{args.expname}/fix_view",  # 保存路径
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ray_type=args.ray_type,
            device=device,
            change_view=False,  # 不改变视角
            change_time="change",  # 改变时间
            render_focal=fix_view_focals,
        )

        # 改变视角，改变时间
        os.makedirs(f"{logfolder}/{args.expname}/change_view_time", exist_ok=True)  # 创建改变视角时间文件夹
        _, depth_change_view_time_all = evaluation_path(
            test_dataset,
            focal_refine.cpu(),
            tensorf_static,
            tensorf,
            change_view_time_poses,
            renderer,
            f"{logfolder}/{args.expname}/change_view_time",  # 保存路径
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ray_type=args.ray_type,
            device=device,
            change_view=False,  # 不改变视角
            change_time="change",  # 改变时间
            render_focal=change_view_time_focals,
        )
        # dolly 摄像机推进
        os.makedirs(f"{logfolder}/{args.expname}/dolly", exist_ok=True)
        _, depth_dolly_all = evaluation_path(
            test_dataset,
            focal_refine.cpu(),
            tensorf_static,
            tensorf,
            dolly_poses,
            renderer,
            f"{logfolder}/{args.expname}/dolly",  # 保存路径
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ray_type=args.ray_type,
            device=device,
            change_view=True,  # 改变视角
            change_time=render_idx / (args.N_voxel_t - 1) * 2.0 - 1.0,  # 改变时间
            render_focal=dolly_focals,
        )
        # zoom  缩放
        os.makedirs(f"{logfolder}/{args.expname}/zoom", exist_ok=True)
        _, depth_zoom_all = evaluation_path(
            test_dataset,
            focal_refine.cpu(),
            tensorf_static,
            tensorf,
            zoom_poses,
            renderer,
            f"{logfolder}/{args.expname}/zoom",  # 保存路径
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ray_type=args.ray_type,
            device=device,
            change_view=True,  # 改变视角
            change_time=render_idx / (args.N_voxel_t - 1) * 2.0 - 1.0,  # 改变时间
            render_focal=zoom_focals,
        )
        # spiral 螺旋运动
        os.makedirs(f"{logfolder}/{args.expname}/spiral", exist_ok=True)
        _, depth_spiral_all = evaluation_path(
            test_dataset,
            focal_refine.cpu(),
            tensorf_static,
            tensorf,
            spiral_poses,
            renderer,
            f"{logfolder}/{args.expname}/spiral",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ray_type=args.ray_type,
            device=device,
            change_view=True,
            change_time=render_idx / (args.N_voxel_t - 1) * 2.0 - 1.0,
            render_focal=spiral_focals,
        )

        # 将所有深度图堆叠成一个张量
        all_depth = torch.stack(
            depth_test_all  # 测试数据集的深度图
            + depth_fix_view_all  # 固定视角的深度图
            + depth_change_view_time_all  # 改变视角和时间的深度图
            + depth_dolly_all  # dolly运动的深度图
            + depth_zoom_all  # 缩放运动的深度图
            + depth_spiral_all  # 螺旋运动的深度图
        )
        # 计算深度图的最小值和最大值
        depth_map_min = torch.quantile(all_depth[:, ::4, ::4], 0.05).item()
        depth_map_max = torch.quantile(all_depth[:, ::4, ::4], 0.95).item()
        # 对测试数据集中的每个深度图进行可视化处理
        for idx in range(len(depth_test_all)):
            depth_test_all[idx] = visualize_depth_numpy(
                torch.clamp(
                    depth_test_all[idx], min=depth_map_min, max=depth_map_max
                ).numpy(),
                (depth_map_min, depth_map_max),
            )[0]

        # 将深度图序列保存为视频文件
        imageio.mimwrite(
            f"{logfolder}/{args.expname}/imgs_test_all/depthvideo.mp4",
            np.stack(depth_test_all),  # 将深度图堆叠成一个numpy数组
            fps=30,  # 每秒帧数
            quality=8,  # 视频质量
            format="ffmpeg",  # 视频格式
            output_params=["-f", "mp4"],  # 输出参数
        )

        # 遍历固定视角的深度图列表
        for idx in range(len(depth_fix_view_all)):
            # 对每个深度图进行可视化处理：
            # 1. 将深度值限制在指定范围内
            # 2. 将torch张量转换为numpy数组
            # 3. 使用可视化函数对深度图进行处理
            # 4. 仅保留处理后的深度图
            depth_fix_view_all[idx] = visualize_depth_numpy(
                torch.clamp(
                    depth_fix_view_all[idx], min=depth_map_min, max=depth_map_max
                ).numpy(),
                (depth_map_min, depth_map_max),
            )[0]
        imageio.mimwrite(
            f"{logfolder}/{args.expname}/fix_view/depthvideo.mp4",
            np.stack(depth_fix_view_all),
            fps=30,
            quality=8,
            format="ffmpeg",
            output_params=["-f", "mp4"],
        )

        for idx in range(len(depth_change_view_time_all)):
            depth_change_view_time_all[idx] = visualize_depth_numpy(
                torch.clamp(
                    depth_change_view_time_all[idx],
                    min=depth_map_min,
                    max=depth_map_max,
                ).numpy(),
                (depth_map_min, depth_map_max),
            )[0]
        imageio.mimwrite(
            f"{logfolder}/{args.expname}/change_view_time/depthvideo.mp4",
            np.stack(depth_change_view_time_all),
            fps=30,
            quality=8,
            format="ffmpeg",
            output_params=["-f", "mp4"],
        )

        for idx in range(len(depth_dolly_all)):
            depth_dolly_all[idx] = visualize_depth_numpy(
                torch.clamp(
                    depth_dolly_all[idx], min=depth_map_min, max=depth_map_max
                ).numpy(),
                (depth_map_min, depth_map_max),
            )[0]
        imageio.mimwrite(
            f"{logfolder}/{args.expname}/dolly/depthvideo.mp4",
            np.stack(depth_dolly_all),
            fps=30,
            quality=8,
            format="ffmpeg",
            output_params=["-f", "mp4"],
        )

        for idx in range(len(depth_zoom_all)):
            depth_zoom_all[idx] = visualize_depth_numpy(
                torch.clamp(
                    depth_zoom_all[idx], min=depth_map_min, max=depth_map_max
                ).numpy(),
                (depth_map_min, depth_map_max),
            )[0]
        imageio.mimwrite(
            f"{logfolder}/{args.expname}/zoom/depthvideo.mp4",
            np.stack(depth_zoom_all),
            fps=30,
            quality=8,
            format="ffmpeg",
            output_params=["-f", "mp4"],
        )

        for idx in range(len(depth_spiral_all)):
            depth_spiral_all[idx] = visualize_depth_numpy(
                torch.clamp(
                    depth_spiral_all[idx], min=depth_map_min, max=depth_map_max
                ).numpy(),
                (depth_map_min, depth_map_max),
            )[0]
        imageio.mimwrite(
            f"{logfolder}/{args.expname}/spiral/depthvideo.mp4",
            np.stack(depth_spiral_all),
            fps=30,
            quality=8,
            format="ffmpeg",
            output_params=["-f", "mp4"],
        )

    return args.ckpt


@torch.no_grad()
def prealign_cameras(pose_in, pose_GT):
    # compute 3D similarity transform via Procrustes analysis
    center = torch.zeros(1, 1, 3, device=pose_in.device)
    center_pred = cam2world(center, pose_in)[:, 0]  # [N,3]
    center_GT = cam2world(center, pose_GT)[:, 0]  # [N,3]
    try:
        sim3 = procrustes_analysis(center_GT, center_pred)
    except:
        print("warning: SVD did not converge...")
        sim3 = edict(t0=0, t1=0, s0=1, s1=1, R=torch.eye(3, device=pose_in.device))
    # align the camera poses
    center_aligned = (center_pred - sim3.t1) / sim3.s1 @ sim3.R.t() * sim3.s0 + sim3.t0
    R_aligned = pose_in[..., :3] @ sim3.R.t()
    t_aligned = (-R_aligned @ center_aligned[..., None])[..., 0]
    pose_aligned = pose(R=R_aligned, t=t_aligned)
    return pose_aligned, sim3


@torch.no_grad()
def evaluate_camera_alignment(pose_aligned, pose_GT):
    # measure errors in rotation and translation
    # pose_aligned: [N, 3, 4]
    # pose_GT:      [N, 3, 4]
    R_aligned, t_aligned = pose_aligned.split([3, 1], dim=-1)  # [N, 3, 3], [N, 3, 1]
    R_GT, t_GT = pose_GT.split([3, 1], dim=-1)  # [N, 3, 3], [N, 3, 1]
    R_error = rotation_distance(R_aligned, R_GT)
    t_error = (t_aligned - t_GT)[..., 0].norm(dim=-1)
    return R_error, t_error


def get_camera_mesh(pose, depth=1):
    vertices = (
        torch.tensor(
            [[-0.5, -0.5, 1], [0.5, -0.5, 1], [0.5, 0.5, 1], [-0.5, 0.5, 1], [0, 0, 0]]
        )
        * depth
    )
    faces = torch.tensor(
        [[0, 1, 2], [0, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
    )
    # vertices = cam2world(vertices[None],pose)
    vertices = vertices @ pose[:, :3, :3].transpose(-1, -2)
    vertices += pose[:, None, :3, 3]
    wireframe = vertices[:, [0, 1, 2, 3, 0, 4, 1, 2, 4, 3]]
    return vertices, faces, wireframe


def merge_wireframes(wireframe):
    wireframe_merged = [[], [], []]
    for w in wireframe:
        wireframe_merged[0] += [float(n) for n in w[:, 0]]
        wireframe_merged[1] += [float(n) for n in w[:, 1]]
        wireframe_merged[2] += [float(n) for n in w[:, 2]]
    return wireframe_merged


def compute_depth_loss(dyn_depth, gt_depth):
    t_d = torch.median(dyn_depth)  # 计算动态深度的中值
    s_d = torch.mean(torch.abs(dyn_depth - t_d))  # 计算动态深度的标准差
    dyn_depth_norm = (dyn_depth - t_d) / (s_d + 1e-10)  # 归一化动态深度

    t_gt = torch.median(gt_depth)  # 计算GT深度的中值
    s_gt = torch.mean(torch.abs(gt_depth - t_gt))  # 计算GT深度的标准差
    gt_depth_norm = (gt_depth - t_gt) / (s_gt + 1e-10)  # 归一化GT深度

    # 返回归一化深度之间的平方差
    return torch.sum((dyn_depth_norm - gt_depth_norm) ** 2)


def get_stats(X, norm=2):
    """
    :param X (N, H, W, C)
    :returns mean (1, 1, 1, C), scale (1)
    """
    mean = X.mean(dim=(0, 1, 2), keepdim=True)  # (1, 1, 1, C)
    if norm == 1:
        mag = torch.abs(X - mean).sum(dim=-1)  # (N, H, W)
    else:
        mag = np.sqrt(2) * torch.sqrt(torch.square(X - mean).sum(dim=-1))  # (N, H, W)
    scale = mag.mean() + 1e-6
    return mean, scale


def reconstruction(args):
    # 初始化数据集
    dataset = dataset_dict[args.dataset_name]  # 从数据集字典中获取指定名称的数据集类
    # 训练数据集
    train_dataset = dataset(
        args.datadir,  # 数据集目录路径
        split="train",  # 使用训练数据集划分
        downsample=args.downsample_train,  # 训练数据降采样因子
        is_stack=False,  # 是否堆叠输入图像
        use_disp=args.use_disp,  # 是否使用视差图
        use_foreground_mask=args.use_foreground_mask,  # 是否使用前景遮罩
        with_GT_poses=args.with_GT_poses,  # 是否提供真实相机位姿信息
        ray_type=args.ray_type,  # 光线类型
    )
    # 测试数据集
    test_dataset = dataset(
        args.datadir,  # 数据集目录路径
        split="test",  # 使用测试数据集划分
        downsample=args.downsample_train,  # 测试数据降采样因子
        is_stack=True,  # 是否堆叠输入图像
        use_disp=args.use_disp,  # 是否使用视差图
        use_foreground_mask=args.use_foreground_mask,  # 是否使用前景遮罩
        with_GT_poses=args.with_GT_poses,  # 是否提供真实相机位姿信息
        ray_type=args.ray_type,  # 光线类型
    )
    white_bg = train_dataset.white_bg  # 获取训练数据集的背景颜色信息
    near_far = train_dataset.near_far  # 获取训练数据集的近远距离信息
    W, H = train_dataset.img_wh  # 获取训练数据集图像的宽度和高度

    # 初始化分辨率参数
    upsamp_list = args.upsamp_list  # 上采样列表
    n_lamb_sigma = args.n_lamb_sigma  # 泊松方程中的 sigma 参数
    n_lamb_sh = args.n_lamb_sh  # 泊松方程中的 sh 参数

    # 添加时间戳标记
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f"{args.basedir}/{args.expname}"

    # 初始化日志文件目录
    os.makedirs(logfolder, exist_ok=True)  # 创建主日志目录
    os.makedirs(f"{logfolder}/imgs_vis", exist_ok=True)  # 创建用于可视化图像的子目录
    os.makedirs(f"{logfolder}/imgs_rgba", exist_ok=True)  # 创建用于 RGBA 图像的子目录
    os.makedirs(f"{logfolder}/rgba", exist_ok=True)  # 创建用于 RGBA 文件的子目录
    summary_writer = SummaryWriter(logfolder)  # 初始化 Tensorboard 日志记录器

    # 初始化参数
    aabb = train_dataset.scene_bbox.to(device)  # 将场景边界框转移到指定的设备上
    reso_cur = N_to_reso(args.N_voxel_init, aabb)  # 根据初始体素数和边界框计算当前体素分辨率
    nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))  # 计算采样点数量，取较小值

    # static TensoRF
    tensorf_static = TensorVMSplit(
        aabb,
        reso_cur,
        args.N_voxel_t,
        device,
        density_n_comp=n_lamb_sigma,
        appearance_n_comp=n_lamb_sh,
        app_dim=args.data_dim_color,
        near_far=near_far,
        shadingMode=args.shadingModeStatic,
        alphaMask_thres=args.alpha_mask_thre,
        density_shift=args.density_shift,
        distance_scale=args.distance_scale,
        pos_pe=args.pos_pe,
        view_pe=args.view_pe,
        fea_pe=2,
        featureC=args.featureC,
        step_ratio=args.step_ratio,
        fea2denseAct=args.fea2denseAct,
    )

    # dynamic tensorf
    if args.ckpt is not None:
        # 加载模型检查点文件
        ckpt = torch.load(args.ckpt, map_location=device)
        # 获取参数字典
        kwargs = ckpt["kwargs"]
        # 更新参数字典中的设备信息
        kwargs.update({"device": device})
        # 根据参数字典中的信息实例化模型
        tensorf = eval(args.model_name)(**kwargs)
        # 加载模型权重
        tensorf.load(ckpt)
    # 如果未提供模型检查点文件，则根据给定的参数初始化新的模型
    else:
        # 根据给定的参数初始化新的模型
        tensorf = eval(args.model_name)(
            aabb,
            reso_cur,
            args.N_voxel_t,
            device,
            density_n_comp=n_lamb_sigma,
            appearance_n_comp=n_lamb_sh,
            app_dim=args.data_dim_color,
            near_far=near_far,
            shadingMode=args.shadingMode,
            alphaMask_thres=args.alpha_mask_thre,
            density_shift=args.density_shift,
            distance_scale=args.distance_scale,
            pos_pe=args.pos_pe,
            view_pe=args.view_pe,
            fea_pe=0,
            featureC=args.featureC,
            step_ratio=args.step_ratio,
            fea2denseAct=args.fea2denseAct,
        )

    # 获取静态模型和动态模型的优化参数组
    grad_vars = tensorf_static.get_optparam_groups(args.lr_init, args.lr_basis)
    grad_vars.extend(tensorf.get_optparam_groups(args.lr_init, args.lr_basis))

    # 如果学习率衰减步数大于0，则计算学习率衰减的系数
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
    else:
        # 如果学习率衰减步数为0，则将学习率衰减步数设置为总迭代步数
        args.lr_decay_iters = args.n_iters
        # 计算学习率衰减的系数
        lr_factor = args.lr_decay_target_ratio ** (1 / args.n_iters)

    # 打印学习率衰减相关参数信息
    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)

    # 使用Adam优化器，并传入优化参数组和动量参数
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    # linear in logrithmic space
    N_voxel_list = (
        torch.round(
            torch.exp(
                torch.linspace(
                    np.log(args.N_voxel_init),
                    np.log(args.N_voxel_final),
                    len(upsamp_list) + 1,
                )
            )
        ).long()
    ).tolist()[1:]

    # 初始化PSNR列表，用于存储训练和测试过程中的PSNR值
    PSNRs, PSNRs_test = [], [0]

    allrgbs = train_dataset.all_rgbs
    allts = train_dataset.all_ts

    # 如果存在真实姿态信息，则获取训练数据集中的所有姿态信息
    if args.with_GT_poses:
        allposes = train_dataset.all_poses  # (12, 3, 4)

    # 获取训练数据集中的前向和后向光流以及光流掩码
    allflows_f = train_dataset.all_flows_f.to(device)
    allflowmasks_f = train_dataset.all_flow_masks_f.to(device)
    allflows_b = train_dataset.all_flows_b.to(device)
    allflowmasks_b = train_dataset.all_flow_masks_b.to(device)

    # 如果使用视差信息，则获取训练数据集中的所有视差信息
    if args.use_disp:
        alldisps = train_dataset.all_disps

    # 获取训练数据集中的所有前景掩码
    allforegroundmasks = train_dataset.all_foreground_masks

    # 初始化姿态矩阵
    init_poses = torch.zeros(args.N_voxel_t, 9)

    # 如果存在真实姿态信息，则将其作为初始姿态
    if args.with_GT_poses:
        init_poses[..., 0:3] = allposes[..., :, 0]
        init_poses[..., 3:6] = allposes[..., :, 1]
        init_poses[..., 6:9] = allposes[..., :, 3]
    else:
        # 否则，将初始姿态设置为单位矩阵
        init_poses[..., 0] = 1
        init_poses[..., 4] = 1

    # 使用Embedding层初始化姿态矩阵，将初始姿态复制到设备上
    poses_refine = torch.nn.Embedding(args.N_voxel_t, 9).to(device)
    poses_refine.weight.data.copy_(init_poses.to(device))

    # optimizing focal length
    fov_refine_embedding = torch.nn.Embedding(1, 1).to(device)
    # 将焦距修正参数设置为初始值（30度）
    fov_refine_embedding.weight.data.copy_(
        torch.ones(1, 1).to(device) * 30 / 180 * np.pi
    )

    # 如果存在真实姿态信息，则将真实焦距设为初始焦距
    if args.with_GT_poses:
        focal_refine = torch.tensor(train_dataset.focal[0]).to(device)

    # 生成网格点坐标
    ii, jj = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )
    grid = torch.from_numpy(np.stack([ii, jj], -1)).to(device)
    # 将网格点坐标扩展为与姿态数量相匹配的形状
    grid = torch.tile(torch.unsqueeze(grid, 0), (args.N_voxel_t, 1, 1, 1))
    # 将所有网格点坐标展平
    allgrids = grid.view(-1, 2)

    # setup optimizer
    if args.optimize_poses:
        lr_pose = 3e-3 # 初始学习率
        lr_pose_end = 1e-5  # 5:X, 10:X  # 终止学习率
        # 创建优化器和学习率调度器
        optimizer_pose = torch.optim.Adam(poses_refine.parameters(), lr=lr_pose)
        gamma = (lr_pose_end / lr_pose) ** (
            1.0 / (args.n_iters // 2 - args.upsamp_list[-1])
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_pose, gamma=gamma)

    if args.optimize_focal_length:
        lr_pose = 3e-3
        lr_pose_end = 1e-5  # 5:X, 10:X
        # 创建优化器和学习率调度器
        optimizer_focal = torch.optim.Adam(fov_refine_embedding.parameters(), lr=0.0)
        gamma = (lr_pose_end / lr_pose) ** (
            1.0 / (args.n_iters // 2 - args.upsamp_list[-1])
        )
        scheduler_focal = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_focal, gamma=gamma
        )

    # 创建训练采样器
    trainingSampler = SimpleSampler(allts.shape[0], args.batch_size)
    trainingSampler_2 = SimpleSampler(allts.shape[0], args.batch_size)

    # 初始化正交正则化权重
    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    # 初始化L1正则化权重
    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    # 初始化TV正则化权重（密度和外观）
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    # 初始化失真权重（静态和动态）
    distortion_weight_static, distortion_weight_dynamic = (
        args.distortion_weight_static,
        args.distortion_weight_dynamic,
    )
    # 创建TVLoss实例
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")

    # 设置衰减迭代次数
    decay_iteration = 100
    # 创建进度条
    pbar = tqdm(
        range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout
    )
    for iteration in pbar:
        # Lambda decay. 用于损失加权的动态参数
        Temp_static = 1.0 / (10 ** (iteration / (100000))) # 随着迭代次数的增加而指数级下降
        # Temp随着迭代次数的增加而指数级下降，衰减速率与Temp_static不一致；
        # decay_iteration用于控制 Temp 的衰减速度，使其可以在更长的迭代周期内保持较高的值。
        Temp = 1.0 / (10 ** (iteration // (decay_iteration * 1000)))
        # 位移总变化（TV）正则化的 Lambda 衰减。
        Temp_disp_TV = 1.0 / (10 ** (iteration // (50000)))

        # 优化焦距长度
        if args.optimize_focal_length:
            focal_refine = (
                np.maximum(H, W) / 2.0 / torch.tan(fov_refine_embedding.weight[0, 0])
            )

        # 获取当前迭代下的样本索引、RGB值、时间采样值和网格
        ray_idx = trainingSampler.nextids()

        # 获取当前迭代下的样本索引对应的RGB值、时间采样值和网格
        rgb_train, ts_train, grid_train = (
            allrgbs[ray_idx].to(device),
            allts[ray_idx].to(device),
            allgrids[ray_idx],
        )

        # 获取当前迭代下的样本索引对应的前向流、前向流遮罩、后向流和后向流遮罩
        flow_f_train, flow_mask_f_train, flow_b_train, flow_mask_b_train = (
            allflows_f[ray_idx],
            allflowmasks_f[ray_idx][..., None],
            allflows_b[ray_idx],
            allflowmasks_b[ray_idx][..., None],
        )

        # 如果使用视差图，则获取当前迭代下的样本索引对应的视差图
        if args.use_disp:
            alldisps_train = alldisps[ray_idx].to(device)

        # 获取当前迭代下的样本索引对应的所有前景遮罩
        allforegroundmasks_train = allforegroundmasks[ray_idx].to(device)

        # 复制姿势参数矩阵，并调整其格式
        poses_refine2 = poses_refine.weight.clone()
        poses_refine2[..., 6:9] = poses_refine2[..., 6:9]
        poses_mtx = pose_to_mtx(poses_refine2)

        # 将样本索引转换为像素坐标和视图ID
        i, j, view_ids = ids2pixel(W, H, ray_idx.to(device))

        # 获取射线方向
        directions = get_ray_directions_lean(
            i, j, [focal_refine, focal_refine], [W / 2, H / 2]
        )

        # 获取射线的原点和方向
        poses_mtx_batched = poses_mtx[view_ids]
        rays_o, rays_d = get_rays_lean(directions, poses_mtx_batched)  # both (b, 3)

        # 如果射线类型是 "ndc"，则调整射线到归一化设备坐标系
        if args.ray_type == "ndc":
            rays_o, rays_d = ndc_rays_blender2(
                H, W, [focal_refine, focal_refine], 1.0, rays_o, rays_d
            )

        # 将射线原点和方向连接为 (N, 6) 的张量
        rays_train = torch.cat([rays_o, rays_d], -1).view(-1, 6)

        # 计算时间和像素索引的参考值
        t_ref = ray_idx // (H * W)
        u_ref = (ray_idx % (H * W)) // W  # height
        v_ref = (ray_idx % (H * W)) % W  # width
        t_interval = 2 / (args.N_voxel_t - 1)

        # index the pose for forward and backward
        allposes_refine_f = torch.cat((poses_mtx[1:], poses_mtx[-1:]), 0)
        allposes_refine_b = torch.cat((poses_mtx[0:1], poses_mtx[:-1]), 0)
        allposes_refine_f_train = allposes_refine_f[t_ref]
        allposes_refine_b_train = allposes_refine_b[t_ref]
        # 计算总损失初始化
        total_loss = 0.0

        # 在射线上采样点
        # xyz_sampled: 采样得到的点的坐标
        # z_vals: 插值得到的深度值
        # ray_valid: 指示哪些射线采样点在边界框内
        xyz_sampled, z_vals, ray_valid = sampleXYZ(
            tensorf,
            rays_train.detach(),
            N_samples=nSamples,
            ray_type=args.ray_type,
            is_train=True,
        )
        # static tensorf 使用静态模型进行渲染
        # rgb_points_static: 静态模型渲染得到的颜色
        # sigmas_static: 静态模型渲染得到的密度
        _, _, _, _, _, _, rgb_points_static, sigmas_static, _, _ = tensorf_static(
            rays_train.detach(),  # 射线方向
            ts_train,  # 插值深度值
            None,  # 空
            xyz_sampled,  # 采样点坐标
            z_vals,  # 插值深度值
            ray_valid,  # 指示哪些射线采样点在边界框内
            is_train=True,  # 是否处于训练模式
            white_bg=white_bg,  # 是否使用白色背景
            ray_type=args.ray_type,  # 射线类型
            N_samples=nSamples,  # 采样数量
        )

        # dynamic tensorf  使用动态模型进行渲染
        # 使用动态模型进行渲染
        # rgb_points_dynamic: 动态模型渲染得到的颜色
        # sigmas_dynamic: 动态模型渲染得到的密度
        # z_vals_dynamic: 动态模型渲染得到的深度值
        # dists_dynamic: 动态模型渲染得到的距离值
        (
            _,
            _,
            blending,
            pts_ref,
            _,
            _,
            rgb_points_dynamic,
            sigmas_dynamic,
            z_vals_dynamic,
            dists_dynamic,
        ) = tensorf(
            rays_train.detach(),  # 射线方向
            ts_train,  # 插值深度值
            None,  # 空
            xyz_sampled,  # 采样点坐标
            z_vals,  # 插值深度值
            ray_valid,  # 指示哪些射线采样点在边界框内
            is_train=True,  # 是否处于训练模式
            white_bg=white_bg,  # 是否使用白色背景
            ray_type=args.ray_type,  # 射线类型
            N_samples=nSamples,  # 采样数量
        )

        (   # 这个部分将静态模型和动态模型的渲染结果转换为可输出的格式。
            # 静态渲染的结果
            rgb_map_full,  # 完整场景的颜色映射
            _,  # 深度图（暂不使用）
            _,  # 精确度图（暂不使用）
            _,  # 权重图（暂不使用）
            # 动态渲染的结果
            rgb_map_s,  # 单个场景的颜色映射
            depth_map_s,  # 单个场景的深度图
            _,  # 精确度图（暂不使用）
            weights_s,  # 单个场景的权重图
            # 深度图
            rgb_map_d,  # 深度场景的颜色映射
            depth_map_d,  # 深度场景的深度图
            _,  # 精确度图（暂不使用）
            weights_d,  # 深度场景的权重图
            dynamicness_map,  # 动态度图
        ) = raw2outputs(
            rgb_points_static.detach(),  # 静态模型渲染得到的颜色
            sigmas_static.detach(),  # 静态模型渲染得到的密度
            rgb_points_dynamic,  # 动态模型渲染得到的颜色
            sigmas_dynamic,  # 动态模型渲染得到的密度
            dists_dynamic,  # 动态模型渲染得到的距离值
            blending,  # 动态模型的混合权重
            z_vals_dynamic,  # 动态模型渲染得到的深度值
            rays_train.detach(),  # 射线方向
            is_train=True,  # 是否处于训练模式
            ray_type=args.ray_type,  # 射线类型
        )

        # novel mask zero loss
        # sample training view and novel time combination;从训练集中随机采样视图和新的时间组合
        # 从训练集中随机采样视图和新的时间组合
        ray_idx_rand = trainingSampler_2.nextids()  # 随机选择一组光线索引
        ts_train_rand = allts[ray_idx_rand].to(device)  # 获取对应的时间步
        # 对随机采样的光线进行采样，获取采样点坐标、深度值和光线有效性
        xyz_sampled_rand, z_vals_rand, ray_valid_rand = sampleXYZ(
            tensorf,
            rays_train.detach(),
            N_samples=nSamples,
            ray_type=args.ray_type,
            is_train=True,
        )
        # 使用静态的Tensor Flow模型对随机采样的光线进行前向推断，获取RGB颜色和密度值
        (
            _,
            _,
            _,
            _,
            _,
            _,
            rgb_points_static_rand,
            sigmas_static_rand,
            _,
            _,
        ) = tensorf_static(
            rays_train.detach(),
            ts_train_rand,
            None,
            xyz_sampled_rand,
            z_vals_rand,
            ray_valid_rand,
            is_train=True,
            white_bg=white_bg,
            ray_type=args.ray_type,
            N_samples=nSamples,
        )
        (
            _,
            _,
            blending_rand,
            _,
            _,
            _,
            rgb_points_dynamic_rand,
            sigmas_dynamic_rand,
            z_vals_dynamic_rand,
            dists_dynamic_rand,
        ) = tensorf(
            rays_train.detach(),
            ts_train_rand,
            None,
            xyz_sampled_rand,
            z_vals_rand,
            ray_valid_rand,
            is_train=True,
            white_bg=white_bg,
            ray_type=args.ray_type,
            N_samples=nSamples,
        )
        (
            _,
            _,
            _,
            _,
            _,
            depth_map_s_rand,
            _,
            _,
            _,
            depth_map_d_rand,
            _,
            weights_d_rand,
            dynamicness_map_rand,
        ) = raw2outputs(
            rgb_points_static_rand.detach(),  # RGB颜色（静态）
            sigmas_static_rand.detach(),      # 密度值（静态）
            rgb_points_dynamic_rand,          # RGB颜色（动态）
            sigmas_dynamic_rand,              # 密度值（动态）
            dists_dynamic_rand,               # 距离值（动态）
            blending_rand,                    # 融合权重
            z_vals_dynamic_rand,              # 动态深度值
            rays_train.detach(),              # 光线
            is_train=True,                    # 是否训练模式
            ray_type=args.ray_type,           # 光线类型
        )

        if iteration >= args.upsamp_list[3]:  # 如果当前迭代次数大于等于指定迭代阈值
            # 计算倾斜掩码损失
            clamped_mask_rand = torch.clamp(dynamicness_map_rand, min=1e-6, max=1.0 - 1e-6)  # 将动态掩码限制在合理范围内
            skewed_mask_loss_rand = torch.mean( #该损失是通过计算动态掩码的负对数似然来衡量掩码的不规则程度，其中较小的值表示掩码更加均匀。
                -(
                    (clamped_mask_rand**2) * torch.log((clamped_mask_rand**2))  # 计算掩码的负对数似然
                    + (1 - (clamped_mask_rand**2)) * torch.log(1 - (clamped_mask_rand**2))  # 计算1减去掩码的负对数似然
                )
            )
            total_loss += 0.01 * skewed_mask_loss_rand  # 将倾斜掩码损失添加到总损失中
            summary_writer.add_scalar(  # 记录倾斜掩码损失到 TensorBoard
                "train/skewed_mask_loss_rand",
                skewed_mask_loss_rand.detach().item(),
                global_step=iteration,
            )

            # 计算新视角时间掩码损失
            # 衡量在每个迭代步骤中动态掩码的变化程度，这可以帮助确保模型生成的视角在时间上保持稳定性。
            novel_view_time_mask_loss = torch.mean(torch.abs(dynamicness_map_rand))  # 计算动态掩码的绝对值均值，表示时间掩码损失
            total_loss += 0.01 * novel_view_time_mask_loss  # 将新视角时间掩码损失添加到总损失中
            summary_writer.add_scalar(  # 记录新视角时间掩码损失到TensorBoard
                "train/novel_view_time_mask_loss",
                novel_view_time_mask_loss.detach().item(),
                global_step=iteration,
            )

        # novel adaptive Order loss
        if args.ray_type == "ndc":  # 如果光线类型为归一化设备坐标（NDC）
            # novel_order_loss 衡量动态和静态深度图之间的不一致性
            novel_order_loss = torch.sum(  # 计算损失值
                ((depth_map_d_rand - depth_map_s_rand.detach()) ** 2)  # 计算深度图之间的平方差
                * (1.0 - dynamicness_map_rand.detach())  # 乘以动态性掩码
            ) / (torch.sum(1.0 - dynamicness_map_rand.detach()) + 1e-8)  # 归一化
        elif args.ray_type == "contract":  # 如果光线类型为收缩坐标（Contract）
            novel_order_loss = torch.sum(  # 计算损失值
                (
                    (
                        1.0 / (depth_map_d_rand + 1e-6)  # 计算收缩深度倒数
                        - 1.0 / (depth_map_s_rand.detach() + 1e-6)  # 计算静态深度倒数
                    )
                    ** 2
                )
                * (1.0 - dynamicness_map_rand.detach())  # 乘以动态性掩码
            ) / (torch.sum((1.0 - dynamicness_map_rand.detach())) + 1e-8)  # 归一化
        total_loss += novel_order_loss * 10.0  # 将新颖自适应顺序损失添加到总损失中，并乘以权重
        summary_writer.add_scalar(  # 记录新颖自适应顺序损失到TensorBoard
            "train/novel_order_loss",
            (novel_order_loss).detach().item(),
            global_step=iteration,
        )

        if distortion_weight_dynamic > 0:  # 如果动态畸变权重大于0
            ray_id = torch.tile(  # 创建沿着指定维度重复的张量
                torch.range(0, args.batch_size - 1, dtype=torch.int64)[:, None],  # 创建射线ID张量
                (1, weights_d_rand.shape[1]),  # 重复张量的形状与权重张量相同
            ).to(device)  # 将张量移到指定设备上
            # 计算扁平化有效畸变损失，即在动态场景下光线在重建过程中产生的畸变
            loss_distortion = flatten_eff_distloss(
                torch.flatten(weights_d_rand),  # 扁平化权重张量
                torch.flatten(z_vals_dynamic_rand.detach()),  # 扁平化并分离梯度的动态深度值张量
                1 / (weights_d_rand.shape[1]),  # 计算权重的逆归一化因子
                torch.flatten(ray_id),  # 扁平化射线ID张量
            )
            # 将畸变损失乘以权重并添加到总损失中，并随迭代次数线性衰减
            total_loss += (
                loss_distortion * distortion_weight_dynamic * (iteration / args.n_iters)
            )
            # 将畸变损失记录到TensorBoard
            summary_writer.add_scalar(
                "train/loss_distortion_rand",
                (loss_distortion).detach().item(),
                global_step=iteration,
            )

        scene_flow_f, scene_flow_b = tensorf.get_forward_backward_scene_flow(
            pts_ref, ts_train.to(device)
        )  # 获取前向和后向场景流

        # 计算完全渲染的RGB图像与训练RGB图像之间的均方误差损失
        loss = torch.mean((rgb_map_full - rgb_train) ** 2)
        # 计算并记录训练过程中的峰值信噪比（PSNR）
        PSNRs.append(-10.0 * np.log(loss.detach().item()) / np.log(10.0))
        summary_writer.add_scalar("train/PSNR", PSNRs[-1], global_step=iteration)  # 记录PSNR
        summary_writer.add_scalar(
            "train/mse", loss.detach().item(), global_step=iteration
        )  # 记录均方误差
        total_loss += 3.0 * loss  # 将完全渲染的RGB图像损失添加到总损失中

        # 计算深度图像渲染的RGB图像与训练RGB图像之间的均方误差损失
        img_d_loss = torch.mean((rgb_map_d - rgb_train) ** 2)
        total_loss += 1.0 * img_d_loss  # 将深度图像渲染的RGB图像损失添加到总损失中
        summary_writer.add_scalar(
            "train/img_d_loss", img_d_loss.detach().item(), global_step=iteration
        ) # 记录深度图像渲染的RGB图像损失

        # Flow grouping loss

        if iteration >= args.upsamp_list[0]:
            # mask loss：衡量动态渲染深度的动态性和前景掩码之间的差异；
            # 在动态场景下，前景物体的位置和形状可能会发生变化，因此将深度图中动态物体的位置与前景掩码进行对比，以评估动态性的程度。
            # 如果动态渲染深度与前景掩码不一致，说明模型需要更好地适应动态场景的变化，从而减小这种差异。
            mask_loss = torch.mean(
                torch.abs(dynamicness_map - allforegroundmasks_train[..., 0])
            )
            total_loss += 0.1 * mask_loss * Temp_disp_TV  # 将掩码损失添加到总损失中
            summary_writer.add_scalar(
                "train/mask_loss", mask_loss.detach().item(), global_step=iteration
            )  # 记录掩码损失


        if iteration >= args.upsamp_list[3]:
            # skewed mask loss
            # 计算偏斜掩码损失，衡量动态渲染深度动态性的偏差
            clamped_mask = torch.clamp(dynamicness_map, min=1e-6, max=1.0 - 1e-6)
            skewed_mask_loss = torch.mean(
                -(
                    (clamped_mask**2) * torch.log((clamped_mask**2)) #部分计算了当动态性接近 1（即深度值接近 1）时的信息熵，它衡量了动态性接近 1 时的不确定性或混乱程度
                    + (1 - (clamped_mask**2)) * torch.log(1 - (clamped_mask**2)) #部分计算了当动态性接近 0（即深度值接近 0）时的信息熵，它同样衡量了动态性接近 0 时的不确定性或混乱程度
                )
            )
            total_loss += 0.01 * skewed_mask_loss  # 将偏斜掩码损失添加到总损失中
            summary_writer.add_scalar(
                "train/skewed_mask_loss",
                skewed_mask_loss.detach().item(),
                global_step=iteration,
            )  # 记录偏斜掩码损失

            # 计算掩码L1正则化损失，通过对动态渲染深度的绝对值进行求和，惩罚动态性的不连续性，从而促进深度图的平滑性和稳定性。
            mask_L1_reg_loss = torch.mean(torch.abs(dynamicness_map))
            total_loss += 0.01 * mask_L1_reg_loss  # 将掩码L1正则化损失添加到总损失中
            summary_writer.add_scalar(
                "train/mask_L1_reg_loss",
                mask_L1_reg_loss.detach().item(),
                global_step=iteration,
            )  # 记录掩码L1正则化损失


        if args.ray_type == "ndc":
            # 根据相机坐标系中的正规化设备坐标(ndc)类型，计算前向和后向场景流场的点位置
            pts_f = pts_ref + scene_flow_f  # 计算前向场景流场点位置
            pts_b = pts_ref + scene_flow_b  # 计算后向场景流场点位置
        elif args.ray_type == "contract":
            # 根据相机坐标系中的收缩(ray contraction)类型，计算前向和后向场景流场的点位置
            # 对点位置进行限制，确保其在合理范围内
            pts_f = torch.clamp(pts_ref + scene_flow_f, min=-2.0 + 1e-6, max=2.0 - 1e-6)
            pts_b = torch.clamp(pts_ref + scene_flow_b, min=-2.0 + 1e-6, max=2.0 - 1e-6)

        # 利用前向和后向场景流场以及其他参数，诱导出前向和后向的光流和视差
        induced_flow_f, induced_disp_f = induce_flow(
            H,
            W,
            focal_refine.detach(),
            allposes_refine_f_train.detach(),
            weights_d,
            pts_f,
            grid_train,
            rays_train.detach(),
            ray_type=args.ray_type,
        )
        # 计算前向光流的损失
        flow_f_loss = (
            torch.sum(torch.abs(induced_flow_f - flow_f_train) * flow_mask_f_train)
            / (torch.sum(flow_mask_f_train) + 1e-8)
            / flow_f_train.shape[-1]
        )
        total_loss += 0.02 * flow_f_loss * Temp  # 将前向光流损失添加到总损失中
        # 记录前向光流损失
        summary_writer.add_scalar(
            "train/flow_f_loss", flow_f_loss.detach().item(), global_step=iteration
        )

        # 利用前向和后向场景流场以及其他参数，诱导出后向的光流和视差
        induced_flow_b, induced_disp_b = induce_flow(
            H,
            W,
            focal_refine.detach(),
            allposes_refine_b_train.detach(),
            weights_d,
            pts_b,
            grid_train,
            rays_train.detach(),
            ray_type=args.ray_type,
        )
        # 计算后向光流的损失
        flow_b_loss = (
            torch.sum(torch.abs(induced_flow_b - flow_b_train) * flow_mask_b_train)
            / (torch.sum(flow_mask_b_train) + 1e-8)
            / flow_b_train.shape[-1]
        )
        total_loss += 0.02 * flow_b_loss * Temp  # 将后向光流损失添加到总损失中
        # 记录后向光流损失
        summary_writer.add_scalar(
            "train/flow_b_loss", flow_b_loss.detach().item(), global_step=iteration
        )

        # 计算小场景流场损失，衡量场景流场的平滑性
        small_scene_flow_loss = torch.mean(torch.abs(scene_flow_f)) + torch.mean(
            torch.abs(scene_flow_b)
        )
        total_loss += args.small_scene_flow_weight * small_scene_flow_loss  # 将小场景流场损失添加到总损失中
        # 记录小场景流场损失
        summary_writer.add_scalar(
            "train/small_scene_flow_loss",
            small_scene_flow_loss.detach().item(),
            global_step=iteration,
        )

        # disparity loss
        # forward
        uv_f = (
            torch.stack((v_ref + 0.5, u_ref + 0.5), -1).to(flow_f_train.device)
            + flow_f_train
        )  # 根据前向光流调整像素坐标
        directions_f = torch.stack(
            [
                (uv_f[..., 0] - W / 2) / (focal_refine.detach()),  # x方向
                -(uv_f[..., 1] - H / 2) / (focal_refine.detach()),  # y方向
                -torch.ones_like(uv_f[..., 0]),  # z方向
            ],
            -1,
        )  # (H, W, 3)
        rays_f_o, rays_f_d = get_rays_lean(directions_f, allposes_refine_f_train) # 获取光线的起点和方向
        if args.ray_type == "ndc":
            rays_f_o, rays_f_d = ndc_rays_blender2(
                H,
                W,
                [focal_refine.detach(), focal_refine.detach()],
                1.0,
                rays_f_o,
                rays_f_d,
            )  # 根据视角和相机参数转换为归一化设备坐标系中的光线
        rays_f_train = torch.cat([rays_f_o, rays_f_d], -1).view(-1, 6)  # 构建训练光线
        xyz_sampled_f, z_vals_f, ray_valid_f = sampleXYZ(
            tensorf,
            rays_f_train.detach(),
            N_samples=nSamples,
            ray_type=args.ray_type,
            is_train=True,
        )  # 采样前向场景点

        # static tensorf
        _, _, _, _, _, _, rgb_points_static_f, sigmas_static_f, _, _ = tensorf_static(
            rays_f_train.detach(),
            ts_train + t_interval,
            None,
            xyz_sampled_f,
            z_vals_f,
            ray_valid_f.detach(),
            is_train=True,
            white_bg=white_bg,
            ray_type=args.ray_type,
            N_samples=nSamples,
        )  # 计算前向静态场景点
        # dynamic tensorf
        (
            _,
            _,
            blending_f,
            pts_ref_ff,
            _,
            _,
            rgb_points_dynamic_f,
            sigmas_dynamic_f,
            z_vals_dynamic_f,
            dists_dynamic_f,
        ) = tensorf(
            rays_f_train.detach(),
            ts_train + t_interval,
            None,
            xyz_sampled_f,
            z_vals_f,
            ray_valid_f.detach(),
            is_train=True,
            white_bg=white_bg,
            ray_type=args.ray_type,
            N_samples=nSamples,
        )  # 计算前向动态场景点
        _, _, _, _, _, _, _, _, _, _, _, weights_d_f, _ = raw2outputs(
            rgb_points_static_f.detach(),
            sigmas_static_f.detach(),
            rgb_points_dynamic_f,
            sigmas_dynamic_f,
            dists_dynamic_f,
            blending_f,
            z_vals_dynamic_f,
            rays_f_train.detach(),
            is_train=True,
            ray_type=args.ray_type,
        )  # 计算前向视差权重
        _, induced_disp_ff = induce_flow(
            H,
            W,
            focal_refine.detach(),
            allposes_refine_f_train.detach(),
            weights_d_f,
            pts_ref_ff,
            grid_train,
            rays_f_train.detach(),
            ray_type=args.ray_type,
        )  # 诱导前向视差

        # 计算前向视差损失
        disp_f_loss = torch.sum(
            torch.abs(induced_disp_f - induced_disp_ff) * flow_mask_f_train
        ) / (torch.sum(flow_mask_f_train) + 1e-8)  # 计算前向视差损失
        total_loss += 0.04 * disp_f_loss * Temp  # 将前向视差损失添加到总损失中
        # 记录前向视差损失
        summary_writer.add_scalar(
            "train/disp_f_loss", disp_f_loss.detach().item(), global_step=iteration
        )
          # backward
        # 根据后向光流调整像素坐标，并计算光线的起点和方向
        uv_b = (
            torch.stack((v_ref + 0.5, u_ref + 0.5), -1).to(flow_b_train.device)
            + flow_b_train
        )  # 根据后向光流调整像素坐标
        directions_b = torch.stack(
            [
                (uv_b[..., 0] - W / 2) / (focal_refine.detach()),  # x方向
                -(uv_b[..., 1] - H / 2) / (focal_refine.detach()),  # y方向
                -torch.ones_like(uv_b[..., 0]),  # z方向
            ],
            -1,
        )  # (H, W, 3)  #根据后向光流计算相机坐标系中的方向
        rays_b_o, rays_b_d = get_rays_lean(directions_b, allposes_refine_b_train)
        if args.ray_type == "ndc":
            rays_b_o, rays_b_d = ndc_rays_blender2(
                H,
                W,
                [focal_refine.detach(), focal_refine.detach()],
                1.0,
                rays_b_o,
                rays_b_d,
            )  # 根据视角和相机参数转换为归一化设备坐标系中的光线
        rays_b_train = torch.cat([rays_b_o, rays_b_d], -1).view(-1, 6)  # 构建训练光线
        xyz_sampled_b, z_vals_b, ray_valid_b = sampleXYZ(
            tensorf,
            rays_b_train.detach(),
            N_samples=nSamples,
            ray_type=args.ray_type,
            is_train=True,
        )  # 采样后向场景点

        # static tensorf
        _, _, _, _, _, _, rgb_points_static_b, sigmas_static_b, _, _ = tensorf_static(
            rays_b_train.detach(),
            ts_train - t_interval,
            None,
            xyz_sampled_b,
            z_vals_b,
            ray_valid_b.detach(),
            is_train=True,
            white_bg=white_bg,
            ray_type=args.ray_type,
            N_samples=nSamples,
        )  # 计算后向静态场景点
        # dynamic tensorf
        (
            _,
            _,
            blending_b,
            pts_ref_bb,
            _,
            _,
            rgb_points_dynamic_b,
            sigmas_dynamic_b,
            z_vals_dynamic_b,
            dists_dynamic_b,
        ) = tensorf(
            rays_b_train.detach(),
            ts_train - t_interval,
            None,
            xyz_sampled_b,
            z_vals_b,
            ray_valid_b.detach(),
            is_train=True,
            white_bg=white_bg,
            ray_type=args.ray_type,
            N_samples=nSamples,
        )  # 计算后向动态场景点
        _, _, _, _, _, _, _, _, _, _, _, weights_d_b, _ = raw2outputs(
            rgb_points_static_b.detach(),
            sigmas_static_b.detach(),
            rgb_points_dynamic_b,
            sigmas_dynamic_b,
            dists_dynamic_b,
            blending_b,
            z_vals_dynamic_b,
            rays_b_train.detach(),
            is_train=True,
            ray_type=args.ray_type,
        )  # 计算后向视差权重
        _, induced_disp_bb = induce_flow(
            H,
            W,
            focal_refine.detach(),
            allposes_refine_b_train.detach(),
            weights_d_b,
            pts_ref_bb,
            grid_train,
            rays_b_train.detach(),
            ray_type=args.ray_type,
        )  # 诱导后向视差
        # 计算后向视差损失
        disp_b_loss = torch.sum(
            torch.abs(induced_disp_b - induced_disp_bb) * flow_mask_b_train
        ) / (torch.sum(flow_mask_b_train) + 1e-8)  # 计算后向视差损失
        total_loss += 0.04 * disp_b_loss * Temp  # 将后向视差损失添加到总损失中
        summary_writer.add_scalar(
            "train/disp_b_loss", disp_b_loss.detach().item(), global_step=iteration
        )  # 记录后向视差损失

        # 平滑场景流损失
        smooth_scene_flow_loss = torch.mean(torch.abs(scene_flow_f + scene_flow_b))  # 计算平滑场景流损失
        total_loss += smooth_scene_flow_loss * args.smooth_scene_flow_weight  # 将平滑场景流损失添加到总损失中
        summary_writer.add_scalar(
            "train/smooth_scene_flow_loss",
            smooth_scene_flow_loss.detach().item(),
            global_step=iteration,
        ) # 记录平滑场景流损失

        # Monocular depth loss
        # 初始化总的单眼深度损失和计数器
        total_mono_depth_loss = 0.0
        counter = 0.0
        total_mono_depth_loss_list = []  # 存储每个有效视角的单眼深度损失
        counter_list = []  # 存储每个有效视角的像素计数器
        # 遍历每个摄像机视角
        for cam_idx in range(args.N_voxel_t):
            # 获取当前视角下的有效掩码
            valid = t_ref == cam_idx
            # 如果有效像素数大于1
            if torch.sum(valid) > 1.0:
                # 根据光线类型计算深度损失
                if args.ray_type == "ndc":
                    total_mono_depth_loss += compute_depth_loss(
                        depth_map_d[valid], -alldisps_train[valid]
                    )  # 计算Monocular深度损失
                elif args.ray_type == "contract":
                    total_mono_depth_loss += compute_depth_loss(
                        1.0 / (depth_map_d[valid] + 1e-6), alldisps_train[valid]
                    )  # 计算Monocular深度损失
                    total_mono_depth_loss_list.append(
                        compute_depth_loss(
                            1.0 / (depth_map_d[valid] + 1e-6), alldisps_train[valid]
                        )
                    )
                # 更新计数器
                counter += torch.sum(valid)
                counter_list.append(valid)
        # 计算平均单眼深度损失
        total_mono_depth_loss = total_mono_depth_loss / counter  # 计算平均单目深度损失
        # 将动态 TensoRF 的总单眼深度损失添加到总损失中
        total_loss += total_mono_depth_loss * args.monodepth_weight_dynamic * Temp  # 将单目深度损失添加到总损失中
        # 记录动态 TensoRF 的总单眼深度损失
        summary_writer.add_scalar(
            "train/total_mono_depth_loss_dynamic",
            total_mono_depth_loss.detach().item(),
            global_step=iteration,
        ) # 记录单目深度损失

        # distortion loss from DVGO
        if distortion_weight_dynamic > 0:  # 如果失真权重大于0
            ray_id = torch.tile(  # 生成光线ID
                torch.range(0, args.batch_size - 1, dtype=torch.int64)[:, None],  # 生成从0到batch_size-1的索引
                (1, weights_d.shape[1]),  # 将索引复制weights_d.shape[1]次，以匹配权重张量的形状
            ).to(device)  # 将光线ID移到相同的设备上
            loss_distortion = flatten_eff_distloss(  # 计算有效的失真损失
                torch.flatten(weights_d),  # 将权重张量展平
                torch.flatten(z_vals_dynamic.detach()),  # 将动态z值展平
                1 / (weights_d.shape[1]),  # 计算每个样本的平均权重
                torch.flatten(ray_id),  # 将光线ID展平
            )
            loss_distortion += flatten_eff_distloss(  # 计算前向光线的有效失真损失
                torch.flatten(weights_d_f),  # 将前向光线的权重张量展平
                torch.flatten(z_vals_dynamic_f.detach()),  # 将前向光线的动态z值展平
                1 / (weights_d_f.shape[1]),  # 计算每个样本的平均权重
                torch.flatten(ray_id),  # 将光线ID展平
            )
            loss_distortion += flatten_eff_distloss(  # 计算后向光线的有效失真损失
                torch.flatten(weights_d_b),  # 将后向光线的权重张量展平
                torch.flatten(z_vals_dynamic_b.detach()),  # 将后向光线的动态z值展平
                1 / (weights_d_b.shape[1]),  # 计算每个样本的平均权重
                torch.flatten(ray_id),  # 将光线ID展平
            )
            total_loss += (  # 将所有光线的失真损失加到总损失中，根据迭代次数归一化
                loss_distortion * distortion_weight_dynamic * (iteration / args.n_iters)
            )
            summary_writer.add_scalar(  # 记录失真损失
                "train/loss_distortion",
                (loss_distortion).detach().item(),
                global_step=iteration,
            )

        # TV losses
        if Ortho_reg_weight > 0:  # 如果正交正则化权重大于0
            loss_reg = tensorf.vector_comp_diffs()  # 计算矢量成分差异
            total_loss += Ortho_reg_weight * loss_reg  # 将正交正则化损失添加到总损失中
            summary_writer.add_scalar(  # 记录正交正则化损失
                "train/reg", loss_reg.detach().item(), global_step=iteration
            )

        if L1_reg_weight > 0:  # 如果L1正则化权重大于0
            loss_reg_L1_density = tensorf.density_L1()  # 计算密度的L1正则化
            total_loss += L1_reg_weight * loss_reg_L1_density  # 将密度的L1正则化损失添加到总损失中
            summary_writer.add_scalar(  # 记录密度的L1正则化损失
                "train/loss_reg_L1_density",
                loss_reg_L1_density.detach().item(),
                global_step=iteration,
            )

        if TV_weight_density > 0:  # 如果TV权重（密度）大于0
            TV_weight_density *= lr_factor  # 调整学习率因子
            loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density  # 计算密度的TV损失
            total_loss = total_loss + loss_tv  # 将密度的TV损失添加到总损失中
            summary_writer.add_scalar(  # 记录密度的TV损失
                "train/reg_tv_density", loss_tv.detach().item(), global_step=iteration
            )
            # TV for blending
            loss_tv = tensorf.TV_loss_blending(tvreg) * TV_weight_density  # 计算混合的TV损失
            total_loss = total_loss + loss_tv  # 将混合的TV损失添加到总损失中
            summary_writer.add_scalar(  # 记录混合的TV损失
                "train/reg_tv_blending", loss_tv.detach().item(), global_step=iteration
            )

        if TV_weight_app > 0:  # 如果TV权重（应用）大于0
            TV_weight_app *= lr_factor  # 调整学习率因子
            loss_tv = tensorf.TV_loss_app(tvreg) * TV_weight_app  # 计算应用的TV损失
            total_loss = total_loss + loss_tv  # 将应用的TV损失添加到总损失中
            summary_writer.add_scalar(  # 记录应用的TV损失
                "train/reg_tv_app", loss_tv.detach().item(), global_step=iteration
            )

        # static part for pose estimation
        # 对姿态估计的静态部分进行处理

        # 使用 sampleXYZ 函数对射线进行采样，生成用于静态场景的采样点的坐标、深度值和有效射线掩码
        xyz_sampled, z_vals, ray_valid = sampleXYZ(
            tensorf,
            rays_train,  # 射线训练数据
            N_samples=nSamples,  # 采样数量
            ray_type=args.ray_type,  # 射线类型参数
            is_train=True,  # 是否用于训练
        )
        # static tensorf
        # 对静态场景的处理
        (
            _, _, _, pts_ref_s, _, _,  # 未使用的变量，暂时不需要注释
            rgb_points_static, sigmas_static, _, _  # 静态点的颜色信息、标准差等
        ) = tensorf_static(
            rays_train,  # 射线训练数据
            ts_train,  # 时间数据
            None,  # 未使用的变量，暂时不需要注释
            xyz_sampled,  # 静态场景的采样点坐标
            z_vals,  # 静态场景的深度值
            ray_valid,  # 有效射线掩码
            is_train=True,  # 是否用于训练
            white_bg=white_bg,  # 背景颜色参数
            ray_type=args.ray_type,  # 射线类型参数
            N_samples=nSamples,  # 采样数量
        )
        # dynamic tensorf
        # 对动态场景的处理
        (
            _, _, blending, pts_ref, _, _,  # 未使用的变量，暂时不需要注释
            rgb_points_dynamic, sigmas_dynamic, z_vals_dynamic, dists_dynamic  # 动态点的颜色信息、深度值等
        ) = tensorf(
            rays_train,  # 射线训练数据
            ts_train,  # 时间数据
            None,  # 未使用的变量，暂时不需要注释
            xyz_sampled,  # 动态场景的采样点坐标
            z_vals,  # 动态场景的深度值
            ray_valid,  # 有效射线掩码
            is_train=True,  # 是否用于训练
            white_bg=white_bg,  # 背景颜色参数
            ray_type=args.ray_type,  # 射线类型参数
            N_samples=nSamples,  # 采样数量
        )

        # 使用 raw2outputs 函数将静态场景和动态场景的信息转换为输出格式
        # 包括静态部分的颜色图像、深度图像、权重图像等
        _, _, _, _, rgb_map_s, depth_map_s, _, weights_s, _, _, _, _, _ = raw2outputs(
            rgb_points_static,  # 静态点的颜色信息
            sigmas_static,  # 静态点的标准差
            rgb_points_dynamic,  # 动态点的颜色信息
            sigmas_dynamic,  # 动态点的标准差
            dists_dynamic,  # 动态点的距离
            blending,  # 混合参数
            z_vals_dynamic,  # 动态场景的深度值
            rays_train,  # 射线训练数据
            is_train=True,  # 是否用于训练
            ray_type=args.ray_type,  # 射线类型参数
        )

        ### static losses
        # RGB loss
        # RGB图像损失，考虑前景掩码
        img_s_loss = (
            torch.sum(
                (rgb_map_s - rgb_train) ** 2  # RGB图像差异的平方
                * (1.0 - allforegroundmasks_train[..., 0:1])  # 考虑前景掩码
            )
            / (torch.sum((1.0 - allforegroundmasks_train[..., 0:1])) + 1e-8)  # 规范化
            / rgb_map_s.shape[-1]  # 归一化
        )
        total_loss += 1.0 * img_s_loss  # 加入总损失
        summary_writer.add_scalar(
            "train/img_s_loss", img_s_loss.detach().item(), global_step=iteration  # 记录损失
        )

        # static distortion loss from DVGO
        # 来自 DVGO 的静态畸变损失
        if distortion_weight_static > 0:  # 如果畸变权重大于0
            ray_id = torch.tile(
                torch.range(0, args.batch_size - 1, dtype=torch.int64)[:, None],
                (1, weights_s.shape[1]),
            ).to(device)  # 生成射线ID
            loss_distortion_static = flatten_eff_distloss(
                torch.flatten(weights_s),  # 权重
                torch.flatten(z_vals),  # 深度值
                1 / (weights_s.shape[1]),  # 权重归一化因子
                torch.flatten(ray_id),  # 射线ID
            )
            total_loss += (
                loss_distortion_static
                * distortion_weight_static
                * (iteration / args.n_iters)  # 加入总损失
            )
            summary_writer.add_scalar(
                "train/loss_distortion_static",
                (loss_distortion_static).detach().item(),
                global_step=iteration,  # 记录损失
            )

        # L1 regularization loss for density
        # 密度的L1正则化损失
        if L1_reg_weight > 0:  # 如果L1正则化权重大于0
            loss_reg_L1_density_s = tensorf_static.density_L1()  # 计算密度的L1损失
            total_loss += L1_reg_weight * loss_reg_L1_density_s  # 加入总损失
            summary_writer.add_scalar(
                "train/loss_reg_L1_density_s",
                loss_reg_L1_density_s.detach().item(),
                global_step=iteration,  # 记录损失
            )

        # TV regularization loss for density
        # 密度的TV正则化损失
        if TV_weight_density > 0:  # 如果TV正则化权重大于0
            loss_tv_static = tensorf_static.TV_loss_density(tvreg) * TV_weight_density  # 计算密度的TV损失
            total_loss = total_loss + loss_tv_static  # 加入总损失
            summary_writer.add_scalar(
                "train/reg_tv_density_static",
                loss_tv_static.detach().item(),
                global_step=iteration,  # 记录损失
            )

        # TV regularization loss for appearance
        # 外观的TV正则化损失
        if TV_weight_app > 0:  # 如果TV正则化权重大于0
            loss_tv_static = tensorf_static.TV_loss_app(tvreg) * TV_weight_app  # 计算外观的TV损失
            total_loss = total_loss + loss_tv_static  # 加入总损失
            summary_writer.add_scalar(
                "train/reg_tv_app_static",
                loss_tv_static.detach().item(),
                global_step=iteration,  # 记录损失
            )

        # 记录参数调整后的焦距
        summary_writer.add_scalar(
            "train/focal_ratio_refine",
            focal_refine.detach().item(),
            global_step=iteration,  # 记录参数
        )

        if args.optimize_poses:
            # static motion loss

            # 计算前向运动的诱导流和视差
            induced_flow_f_s, induced_disp_f_s = induce_flow(
                H,
                W,
                focal_refine,
                allposes_refine_f_train,
                weights_s,
                pts_ref_s,
                grid_train,
                rays_train,
                ray_type=args.ray_type,
            )

            # 计算前向运动的流损失
            flow_f_s_loss = (
                torch.sum(
                    torch.abs(induced_flow_f_s - flow_f_train)  # 流的绝对差异
                    * flow_mask_f_train  # 流的掩码
                    * (1.0 - allforegroundmasks_train[..., 0:1])  # 考虑前景掩码
                )
                / (
                    torch.sum(
                        flow_mask_f_train * (1.0 - allforegroundmasks_train[..., 0:1])
                    )  # 流掩码乘以前景掩码的和
                    + 1e-8  # 避免除零错误
                )
                / flow_f_train.shape[-1]  # 平均到每个像素
            )

            # 将前向运动的流损失添加到总损失中，并乘以动态系数
            total_loss += 0.02 * flow_f_s_loss * Temp_static

            # 计算后向运动的诱导流和视差
            induced_flow_b_s, induced_disp_b_s = induce_flow(
                H,
                W,
                focal_refine,
                allposes_refine_b_train,
                weights_s,
                pts_ref_s,
                grid_train,
                rays_train,
                ray_type=args.ray_type,
            )

            # 计算后向运动的流损失
            flow_b_s_loss = (
                torch.sum(
                    torch.abs(induced_flow_b_s - flow_b_train)  # 流的绝对差异
                    * flow_mask_b_train  # 流的掩码
                    * (1.0 - allforegroundmasks_train[..., 0:1])  # 考虑前景掩码
                )
                / (
                    torch.sum(
                        flow_mask_b_train * (1.0 - allforegroundmasks_train[..., 0:1])
                    )  # 流掩码乘以前景掩码的和
                    + 1e-8  # 避免除零错误
                )
                / flow_b_train.shape[-1]  # 平均到每个像素
            )

            # 将后向运动的流损失添加到总损失中，并乘以温度参数
            total_loss += 0.02 * flow_b_s_loss * Temp_static

            # 记录前向和后向运动的流损失
            summary_writer.add_scalar(
                "train/flow_f_s_loss",
                flow_f_s_loss.detach().item(),
                global_step=iteration,
            )
            summary_writer.add_scalar(
                "train/flow_b_s_loss",
                flow_b_s_loss.detach().item(),
                global_step=iteration,
            )

            # static disparity loss
            # forward
            uv_f = (
                torch.stack((v_ref + 0.5, u_ref + 0.5), -1).to(flow_f_train.device)  # 像素坐标
                + flow_f_train  # 添加光流
            )
            directions_f = torch.stack(
                [
                    (uv_f[..., 0] - W / 2) / (focal_refine),  # x方向
                    -(uv_f[..., 1] - H / 2) / (focal_refine),  # y方向（反转）
                    -torch.ones_like(uv_f[..., 0]),  # z方向（朝向摄像机）
                ],
                -1,
            )  # 光线方向 (H, W, 3)

            # 获取前向光线的原点和方向
            rays_f_o, rays_f_d = get_rays_lean(
                directions_f, allposes_refine_f_train
            )  # both (b, 3)

            # 如果光线类型是"ndc"，则将光线转换为归一化设备坐标系
            if args.ray_type == "ndc":
                rays_f_o, rays_f_d = ndc_rays_blender2(
                    H, W, [focal_refine, focal_refine], 1.0, rays_f_o, rays_f_d
                )

            # 将前向光线组合成一个张量
            rays_f_train = torch.cat([rays_f_o, rays_f_d], -1).view(-1, 6)

            # 对前向光线进行采样，获取采样点、深度值和光线有效性
            xyz_sampled_f, z_vals_f, ray_valid_f = sampleXYZ(
                tensorf_static,
                rays_f_train,
                N_samples=nSamples,
                ray_type=args.ray_type,
                is_train=True,
            )

            # 使用静态 TensoRF 计算前向光线的参考点、权重等
            _, _, _, pts_ref_s_ff, weights_s_ff, _, _, _, _, _ = tensorf_static(
                rays_f_train,
                ts_train,
                None,
                xyz_sampled_f,
                z_vals_f,
                ray_valid_f,
                is_train=True,
                white_bg=white_bg,
                ray_type=args.ray_type,
                N_samples=nSamples,
            )

            # 计算前向光线的诱导视差
            _, induced_disp_s_ff = induce_flow(
                H,
                W,
                focal_refine,
                allposes_refine_f_train,
                weights_s_ff,
                pts_ref_s_ff,
                grid_train,
                rays_f_train,
                ray_type=args.ray_type,
            )

            # 计算前向光线的视差损失
            disp_f_s_loss = torch.sum(
                torch.abs(induced_disp_f_s - induced_disp_s_ff)  # 视差的绝对差异
                * flow_mask_f_train  # 视差的掩码
                * (1.0 - allforegroundmasks_train[..., 0:1])  # 考虑前景掩码
            ) / (
                torch.sum(
                    flow_mask_f_train * (1.0 - allforegroundmasks_train[..., 0:1])
                )  # 视差掩码乘以前景掩码的和
                + 1e-8  # 避免除零错误
            )

            # 将前向光线的视差损失添加到总损失中，并乘以温度参数
            total_loss += 0.04 * disp_f_s_loss * Temp_static

            # 记录前向光线的视差损失
            summary_writer.add_scalar(
                "train/disp_f_s_loss",
                disp_f_s_loss.detach().item(),
                global_step=iteration,
            )
            # backward

            # 计算后向运动的像素坐标并转换为光线方向
            uv_b = (
                torch.stack((v_ref + 0.5, u_ref + 0.5), -1).to(flow_b_train.device)  # 像素坐标
                + flow_b_train  # 添加光流
            )
            directions_b = torch.stack(
                [
                    (uv_b[..., 0] - W / 2) / (focal_refine),  # x方向
                    -(uv_b[..., 1] - H / 2) / (focal_refine),  # y方向（反转）
                    -torch.ones_like(uv_b[..., 0]),  # z方向（朝向摄像机）
                ],
                -1,
            )  # 光线方向 (H, W, 3)

            # 获取后向光线的原点和方向
            rays_b_o, rays_b_d = get_rays_lean(
                directions_b, allposes_refine_b_train
            )  # both (b, 3)

            # 如果光线类型是"ndc"，则将光线转换为归一化设备坐标系
            if args.ray_type == "ndc":
                rays_b_o, rays_b_d = ndc_rays_blender2(
                    H, W, [focal_refine, focal_refine], 1.0, rays_b_o, rays_b_d
                )

            # 将后向光线组合成一个张量
            rays_b_train = torch.cat([rays_b_o, rays_b_d], -1).view(-1, 6)

            # 对后向光线进行采样，获取采样点、深度值和光线有效性
            xyz_sampled_b, z_vals_b, ray_valid_b = sampleXYZ(
                tensorf_static,
                rays_b_train,
                N_samples=nSamples,
                ray_type=args.ray_type,
                is_train=True,
            )

            # 使用静态 TensoRF 计算后向光线的参考点、权重等
            _, _, _, pts_ref_s_bb, weights_s_bb, _, _, _, _, _ = tensorf_static(
                rays_b_train,
                ts_train,
                None,
                xyz_sampled_b,
                z_vals_b,
                ray_valid_b,
                is_train=True,
                white_bg=white_bg,
                ray_type=args.ray_type,
                N_samples=nSamples,
            )

            # 计算后向光线的诱导视差
            _, induced_disp_s_bb = induce_flow(
                H,
                W,
                focal_refine,
                allposes_refine_b_train,
                weights_s_bb,
                pts_ref_s_bb,
                grid_train,
                rays_b_train,
                ray_type=args.ray_type,
            )

            # 计算后向光线的视差损失
            disp_b_s_loss = torch.sum(
                torch.abs(induced_disp_b_s - induced_disp_s_bb)  # 视差的绝对差异
                * flow_mask_b_train  # 视差的掩码
                * (1.0 - allforegroundmasks_train[..., 0:1])  # 考虑前景掩码
            ) / (
                torch.sum(
                    flow_mask_b_train * (1.0 - allforegroundmasks_train[..., 0:1])
                )  # 视差掩码乘以前景掩码的和
                + 1e-8  # 避免除零错误
            )

            # 将后向光线的视差损失添加到总损失中，并乘以温度参数
            total_loss += 0.04 * disp_b_s_loss * Temp_static

            # 记录后向光线的视差损失
            summary_writer.add_scalar(
                "train/disp_b_s_loss",
                disp_b_s_loss.detach().item(),
                global_step=iteration,
            )

            # Monocular depth loss with mask for static TensoRF

            # 初始化总的单眼深度损失和计数器
            total_mono_depth_loss = 0.0
            counter = 0.0

            # 遍历所有摄像机视角
            for cam_idx in range(args.N_voxel_t):
                # 获取当前视角下的有效掩码
                valid = torch.bitwise_and(
                    t_ref == cam_idx, allforegroundmasks_train[..., 0].cpu() < 0.5
                )

                # 如果有效像素数大于1
                if torch.sum(valid) > 1.0:
                    # 根据光线类型计算深度损失
                    if args.ray_type == "ndc":
                        total_mono_depth_loss += compute_depth_loss(
                            depth_map_s[valid], -alldisps_train[valid]
                        )
                    elif args.ray_type == "contract":
                        total_mono_depth_loss += compute_depth_loss(
                            1.0 / (depth_map_s[valid] + 1e-6), alldisps_train[valid]
                        )
                    # 更新计数器
                    counter += torch.sum(valid)

            # 计算平均单眼深度损失
            total_mono_depth_loss = total_mono_depth_loss / counter
            # 将单眼深度损失添加到总损失中，并乘以静态温度参数
            total_loss += (
                total_mono_depth_loss * args.monodepth_weight_static * Temp_static
            )

            # 记录静态 TensoRF 的总单眼深度损失
            summary_writer.add_scalar(
                "train/total_mono_depth_loss_static",
                total_mono_depth_loss.detach().item(),
                global_step=iteration,
            )

            # sample for patch TV loss
            # 计算相邻像素的索引
            i, j, view_ids = ids2pixel(W, H, ray_idx.to(device))
            # 对相邻像素进行边界裁剪，确保不超出图像边界
            i_neighbor = torch.clamp(i + 1, max=W - 1)
            j_neighbor = torch.clamp(j + 1, max=H - 1)

            # 获取相邻像素的光线方向
            directions_i_neighbor = get_ray_directions_lean(
                i_neighbor, j, [focal_refine, focal_refine], [W / 2, H / 2]
            )

            # 获取相邻像素的光线起点和方向
            rays_o_i_neighbor, rays_d_i_neighbor = get_rays_lean(
                directions_i_neighbor, poses_mtx_batched
            )  # both (b, 3)

            # 如果光线类型是 ndc，则将相邻像素的光线转换为 ndc 坐标系下的光线
            if args.ray_type == "ndc":
                rays_o_i_neighbor, rays_d_i_neighbor = ndc_rays_blender2(
                    H,
                    W,
                    [focal_refine, focal_refine],
                    1.0,
                    rays_o_i_neighbor,
                    rays_d_i_neighbor,
                )

            # 将相邻像素的光线起点和方向拼接在一起，并将形状转换为 (b*num_pixels, 6)
            rays_train_i_neighbor = torch.cat(
                [rays_o_i_neighbor, rays_d_i_neighbor], -1
            ).view(-1, 6)

            # 获取另一个方向相邻像素的光线方向
            directions_j_neighbor = get_ray_directions_lean(
                i, j_neighbor, [focal_refine, focal_refine], [W / 2, H / 2]
            )

            # 获取另一个方向相邻像素的光线起点和方向
            rays_o_j_neighbor, rays_d_j_neighbor = get_rays_lean(
                directions_j_neighbor, poses_mtx_batched
            )  # both (b, 3)

            # 如果光线类型是 ndc，则将另一个方向相邻像素的光线转换为 ndc 坐标系下的光线
            if args.ray_type == "ndc":
                rays_o_j_neighbor, rays_d_j_neighbor = ndc_rays_blender2(
                    H,
                    W,
                    [focal_refine, focal_refine],
                    1.0,
                    rays_o_j_neighbor,
                    rays_d_j_neighbor,
                )

            # 将另一个方向相邻像素的光线起点和方向拼接在一起，并将形状转换为 (b*num_pixels, 6)
            rays_train_j_neighbor = torch.cat(
                [rays_o_j_neighbor, rays_d_j_neighbor], -1
            ).view(-1, 6)
            # 获取另一个方向相邻像素的样本点、深度值和有效光线标志
            xyz_sampled_i_neighbor, z_vals_i_neighbor, ray_valid_i_neighbor = sampleXYZ(
                tensorf,
                rays_train_i_neighbor,
                N_samples=nSamples,
                ray_type=args.ray_type,
                is_train=True,
            )
            # 通过静态 TensorF 对相邻像素 i 的光线进行前向传播
            (
                _,
                _,
                _,
                _,
                _,
                _,
                rgb_points_static_i_neighbor,  # 相邻像素 i 的静态 TensorF 中的 RGB 点
                sigmas_static_i_neighbor,      # 相邻像素 i 的静态 TensorF 中的 Sigma 值
                _,
                _,                             # 不需要的值省略
            ) = tensorf_static(
                rays_train_i_neighbor,         # 相邻像素 i 的光线
                ts_train,
                None,
                xyz_sampled_i_neighbor,        # 采样的点
                z_vals_i_neighbor,             # 深度值
                ray_valid_i_neighbor,          # 有效光线标志
                is_train=True,
                white_bg=white_bg,
                ray_type=args.ray_type,
                N_samples=nSamples,
            )

            # 通过动态 TensorF 对相邻像素 i 的光线进行前向传播
            (
                _,
                _,
                blending_i_neighbor,           # 相邻像素 i 的动态 TensorF 中的混合权重
                _,
                _,
                _,
                rgb_points_dynamic_i_neighbor, # 相邻像素 i 的动态 TensorF 中的 RGB 点
                sigmas_dynamic_i_neighbor,    # 相邻像素 i 的动态 TensorF 中的 Sigma 值
                z_vals_dynamic_i_neighbor,    # 相邻像素 i 的动态 TensorF 中的深度值
                dists_dynamic_i_neighbor,     # 相邻像素 i 的动态 TensorF 中的距离值
            ) = tensorf(
                rays_train_i_neighbor,         # 相邻像素 i 的光线
                ts_train,
                None,
                xyz_sampled_i_neighbor,        # 采样的点
                z_vals_i_neighbor,             # 深度值
                ray_valid_i_neighbor,          # 有效光线标志
                is_train=True,
                white_bg=white_bg,
                ray_type=args.ray_type,
                N_samples=nSamples,
            )

            # 将相邻像素 i 的 RGB 点通过后处理转换为深度图
            _, _, _, _, _, depth_map_s_i_neighbor, _, _, _, _, _, _, _ = raw2outputs(
                rgb_points_static_i_neighbor,      # 静态 TensorF 的 RGB 点
                sigmas_static_i_neighbor,          # 静态 TensorF 的 Sigma 值
                rgb_points_dynamic_i_neighbor,     # 动态 TensorF 的 RGB 点
                sigmas_dynamic_i_neighbor,         # 动态 TensorF 的 Sigma 值
                dists_dynamic_i_neighbor,          # 动态 TensorF 的距离值
                blending_i_neighbor,               # 动态 TensorF 的混合权重
                z_vals_dynamic_i_neighbor,         # 动态 TensorF 的深度值
                rays_train_i_neighbor,             # 相邻像素 i 的光线
                is_train=True,
                ray_type=args.ray_type,
            )

            # 对相邻像素 j 进行相同的操作
            xyz_sampled_j_neighbor, z_vals_j_neighbor, ray_valid_j_neighbor = sampleXYZ(
                tensorf,
                rays_train_j_neighbor,
                N_samples=nSamples,
                ray_type=args.ray_type,
                is_train=True,
            )
            (
                _,
                _,
                _,
                _,
                _,
                _,
                rgb_points_static_j_neighbor,
                sigmas_static_j_neighbor,
                _,
                _,
            ) = tensorf_static(
                rays_train_j_neighbor,
                ts_train,
                None,
                xyz_sampled_j_neighbor,
                z_vals_j_neighbor,
                ray_valid_j_neighbor,
                is_train=True,
                white_bg=white_bg,
                ray_type=args.ray_type,
                N_samples=nSamples,
            )
            (
                _,
                _,
                blending_j_neighbor,
                _,
                _,
                _,
                rgb_points_dynamic_j_neighbor,
                sigmas_dynamic_j_neighbor,
                z_vals_dynamic_j_neighbor,
                dists_dynamic_j_neighbor,
            ) = tensorf(
                rays_train_j_neighbor,
                ts_train,
                None,
                xyz_sampled_j_neighbor,
                z_vals_j_neighbor,
                ray_valid_j_neighbor,
                is_train=True,
                white_bg=white_bg,
                ray_type=args.ray_type,
                N_samples=nSamples,
            )
            _, _, _, _, _, depth_map_s_j_neighbor, _, _, _, _, _, _, _ = raw2outputs(
                rgb_points_static_j_neighbor,
                sigmas_static_j_neighbor,
                rgb_points_dynamic_j_neighbor,
                sigmas_dynamic_j_neighbor,
                dists_dynamic_j_neighbor,
                blending_j_neighbor,
                z_vals_dynamic_j_neighbor,
                rays_train_j_neighbor,
                is_train=True,
                ray_type=args.ray_type,
            )
            # 计算深度图的平滑度损失，包括相邻像素 i 和 j 的深度图之间的差异
            disp_smooth_loss = torch.mean(
                (
                    (1.0 / torch.clamp(depth_map_s, min=1e-6))  # 当前像素的深度的倒数
                    - (1.0 / torch.clamp(depth_map_s_i_neighbor, min=1e-6))  # 相邻像素 i 的深度的倒数
                )
                ** 2
            ) + torch.mean(
                (
                    (1.0 / torch.clamp(depth_map_s, min=1e-6))  # 当前像素的深度的倒数
                    - (1.0 / torch.clamp(depth_map_s_j_neighbor, min=1e-6))  # 相邻像素 j 的深度的倒数
                )
                ** 2
            )

            # 将深度平滑度损失添加到总损失中，并乘以权重和温度
            total_loss += disp_smooth_loss * 50.0 * Temp_disp_TV

            # 记录深度平滑度损失到 TensorBoard
            summary_writer.add_scalar(
                "train/disp_smooth_loss",  # TensorBoard 中的标签
                disp_smooth_loss.detach().item(),  # 损失值
                global_step=iteration,  # 全局步数
            )

        # 如果需要优化姿态，重置姿态优化器的梯度
        if args.optimize_poses:
            optimizer_pose.zero_grad()

        # 如果需要优化焦距，重置焦距优化器的梯度
        if args.optimize_focal_length:
            optimizer_focal.zero_grad()

        # 重置主优化器的梯度
        optimizer.zero_grad()

        # 反向传播总损失
        total_loss.backward()

        # 更新主优化器的参数
        optimizer.step()

        # 如果需要优化姿态，更新姿态优化器的参数，并调整学习率
        if args.optimize_poses:
            optimizer_pose.step()
            scheduler.step()

        # 如果需要优化焦距，更新焦距优化器的参数，并调整学习率
        if args.optimize_focal_length:
            optimizer_focal.step()
            scheduler_focal.step()

        # 克隆当前姿态
        pose_aligned = poses_mtx.clone().detach()
        # 记录学习率到 TensorBoard
        summary_writer.add_scalar(
            "train/density_app_plane_lr",  # TensorBoard 中的标签
            optimizer.param_groups[0]["lr"],  # 密度预测平面的学习率
            global_step=iteration,  # 全局步数
        )
        summary_writer.add_scalar(
            "train/basis_mat_lr",  # TensorBoard 中的标签
            optimizer.param_groups[4]["lr"],  # 基础矩阵的学习率
            global_step=iteration,  # 全局步数
        )
        # 如果需要优化姿态，记录姿态优化器的学习率
        if args.optimize_poses:
            summary_writer.add_scalar(
                "train/lr_pose",  # TensorBoard 中的标签
                optimizer_pose.param_groups[0]["lr"],  # 姿态优化器的学习率
                global_step=iteration,  # 全局步数
        )
        # 如果需要优化焦距，记录焦距优化器的学习率
        if args.optimize_focal_length:
            summary_writer.add_scalar(
                "train/lr_focal",  # TensorBoard 中的标签
                optimizer_focal.param_groups[0]["lr"],  # 焦距优化器的学习率
                global_step=iteration,  # 全局步数
        )

        # 更新所有参数组的学习率
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            if args.with_GT_poses:
                pbar.set_description(
                    f"Iteration {iteration:05d}:"  # 当前迭代次数
                    + f" train_psnr = {float(np.mean(PSNRs)):.2f}"  # 平均训练集 PSNR
                    + f" test_psnr = {float(np.mean(PSNRs_test)):.2f}"  # 平均测试集 PSNR
                )
            else:
                pbar.set_description(f"Iteration {iteration:05d}:")  # 当前迭代次数
            PSNRs = []

            # matplotlib poses visualization
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")

            # 如果存在真实姿态，添加真实相机位置
            if args.with_GT_poses:
                vertices, faces, wireframe = get_camera_mesh(allposes, 0.005)
                center_gt = vertices[:, -1]
                ax.scatter(
                    center_gt[:, 0],
                    center_gt[:, 1],
                    center_gt[:, 2],
                    marker="o",
                    color="C0",
                )
                wireframe_merged = merge_wireframes(wireframe)
                for c in range(center_gt.shape[0]):
                    ax.plot(
                        wireframe_merged[0][c * 10 : (c + 1) * 10],
                        wireframe_merged[1][c * 10 : (c + 1) * 10],
                        wireframe_merged[2][c * 10 : (c + 1) * 10],
                        color="C0",
                    )

            # 添加优化后的相机位置
            vertices, faces, wireframe = get_camera_mesh(pose_aligned.cpu(), 0.005)
            center = vertices[:, -1]
            ax.scatter(center[:, 0], center[:, 1], center[:, 2], marker="o", color="C1")
            wireframe_merged = merge_wireframes(wireframe)
            for c in range(center.shape[0]):
                ax.plot(
                    wireframe_merged[0][c * 10 : (c + 1) * 10],
                    wireframe_merged[1][c * 10 : (c + 1) * 10],
                    wireframe_merged[2][c * 10 : (c + 1) * 10],
                    color="C1",
                )

            # 如果存在真实姿态，绘制姿态之间的连线
            if args.with_GT_poses:
                for i in range(center.shape[0]):
                    ax.plot(
                        [center_gt[i, 0], center[i, 0]],
                        [center_gt[i, 1], center[i, 1]],
                        [center_gt[i, 2], center[i, 2]],
                        color="red",
                    )

            set_axes_equal(ax)  # 设置坐标轴相等
            plt.tight_layout()  # 调整布局
            fig.canvas.draw()  # 绘制画布
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = (np.transpose(img, (2, 0, 1)) / 255.0).astype(np.float32)  # 转换图像格式
            summary_writer.add_image("camera_poses", img, iteration)  # 记录相机姿态图像到 TensorBoard
            plt.close(fig)  # 关闭图形窗口

            # 保存模型参数
            tensorf.save(
                poses_mtx.detach().cpu(),  # 姿态参数
                focal_refine.detach().cpu(),  # 焦距参数
                f"{logfolder}/{args.expname}.th",  # 模型保存路径
            )
            tensorf_static.save(
                poses_mtx.detach().cpu(),
                focal_refine.detach().cpu(),
                f"{logfolder}/{args.expname}_static.th",
            )

        if (
            iteration % args.vis_train_every == args.vis_train_every - 1
            and args.N_vis != 0
        ):
            # 渲染图像以进行可视化
            (
                rgb_maps_tb,  # RGB 图像
                depth_maps_tb,  # 深度图像
                blending_maps_tb,  # 混合图像
                gt_rgbs_tb,  # 真实 RGB 图像
                induced_flow_f_tb,  # 前向诱导光流
                induced_flow_b_tb,  # 后向诱导光流
                induced_flow_s_f_tb,  # 前向静态诱导光流
                induced_flow_s_b_tb,  # 后向静态诱导光流
                delta_xyz_tb,  # XYZ 偏移
                rgb_maps_s_tb,  # 静态场景 RGB 图像
                depth_maps_s_tb,  # 静态场景深度图像
                rgb_maps_d_tb,  # 动态场景 RGB 图像
                depth_maps_d_tb,  # 动态场景深度图像
                monodepth_tb,  # 单目深度图像
            ) = render(
                test_dataset,  # 测试数据集
                poses_mtx,  # 姿态矩阵
                focal_refine.cpu(),  # 优化后的焦距
                tensorf_static,  # 静态场景张量函数
                tensorf,  # 张量函数
                args,  # 参数
                renderer,  # 渲染器
                None,  # 无视觉模型参数
                N_vis=args.N_vis,  # 可视化数量
                prtx="",  # PRTX 参数
                N_samples=nSamples,  # 采样数量
                white_bg=white_bg,  # 背景色是否为白色
                ray_type=args.ray_type,  # 光线类型
                compute_extra_metrics=False,  # 是否计算额外的度量指标
            )
            # 记录渲染结果到 TensorBoard
            summary_writer.add_images(
                "test/rgb_maps_s",
                torch.stack(rgb_maps_s_tb, 0),
                global_step=iteration,
                dataformats="NHWC",
            )
            summary_writer.add_images(
                "test/rgb_maps_d",
                torch.stack(rgb_maps_d_tb, 0),
                global_step=iteration,
                dataformats="NHWC",
            )
            summary_writer.add_images(
                "test/depth_map",
                torch.stack(depth_maps_tb, 0),
                global_step=iteration,
                dataformats="NCHW",
            )
            summary_writer.add_images(
                "test/depth_map_s",
                torch.stack(depth_maps_s_tb, 0),
                global_step=iteration,
                dataformats="NCHW",
            )
            summary_writer.add_images(
                "test/depth_map_d",
                torch.stack(depth_maps_d_tb, 0),
                global_step=iteration,
                dataformats="NCHW",
            )
            summary_writer.add_images(
                "test/monodepth_tb",
                torch.stack(monodepth_tb, 0),
                global_step=iteration,
                dataformats="NCHW",
            )
            summary_writer.add_images(
                "test/blending_maps",
                torch.stack(blending_maps_tb, 0),
                global_step=iteration,
                dataformats="NCHW",
            )
            summary_writer.add_images(
                "test/gt_maps",
                torch.stack(gt_rgbs_tb, 0),
                global_step=iteration,
                dataformats="NHWC",
            )
            summary_writer.add_images(
                "test/induced_flow_f",
                torch.stack(induced_flow_f_tb, 0),
                global_step=iteration,
                dataformats="NHWC",
            )
            summary_writer.add_images(
                "test/induced_flow_b",
                torch.stack(induced_flow_b_tb, 0),
                global_step=iteration,
                dataformats="NHWC",
            )
            summary_writer.add_images(
                "test/induced_flow_s_f",
                torch.stack(induced_flow_s_f_tb, 0),
                global_step=iteration,
                dataformats="NHWC",
            )
            summary_writer.add_images(
                "test/induced_flow_s_b",
                torch.stack(induced_flow_s_b_tb, 0),
                global_step=iteration,
                dataformats="NHWC",
            )
            # visualize the index 1 (has both forward and backward)
            gt_flow_f_tb_list = []  # 存储前向光流图像列表
            gt_flow_b_tb_list = []  # 存储后向光流图像列表
            allflows_f_ = allflows_f.view(args.N_voxel_t, H, W, 2)  # 重塑前向光流张量形状
            allflows_b_ = allflows_b.view(args.N_voxel_t, H, W, 2)  # 重塑后向光流张量形状
            for gt_flo_f, gt_flo_b in zip(allflows_f_, allflows_b_):
                # 将前向光流和后向光流转换为图像并添加到对应列表中
                gt_flow_f_tb_list.append(
                    torch.from_numpy(flow_to_image(gt_flo_f.detach().cpu().numpy()))
                )
                gt_flow_b_tb_list.append(
                    torch.from_numpy(flow_to_image(gt_flo_b.detach().cpu().numpy()))
                )
            # 将前向光流和后向光流图像列表添加到 TensorBoard 中
            summary_writer.add_images(
                "test/gt_flow_f",  # TensorBoard 中的标签
                torch.stack(gt_flow_f_tb_list, 0),  # 前向光流图像堆叠
                global_step=iteration,  # 全局步数
                dataformats="NHWC",  # 数据格式
            )
            summary_writer.add_images(
                "test/gt_flow_b",  # TensorBoard 中的标签
                torch.stack(gt_flow_b_tb_list, 0),  # 后向光流图像堆叠
                global_step=iteration,  # 全局步数
                dataformats="NHWC",  # 数据格式
            )
            # 将 delta_xyz_tb 和 gt_mask_tb_list 添加到 TensorBoard 中
            summary_writer.add_images(
                "test/delta_xyz_tb",  # TensorBoard 中的标签
                torch.stack(delta_xyz_tb, 0),  # Delta XYZ 图像堆叠
                global_step=iteration,  # 全局步数
                dataformats="NHWC",  # 数据格式
            )
            gt_mask_tb_list = []
            allforegroundmasks_ = allforegroundmasks.view(args.N_voxel_t, H, W, 3)
            for foregroundmask in allforegroundmasks_:
                gt_mask_tb_list.append(foregroundmask)
            summary_writer.add_images(
                "test/gt_blending_maps",  # TensorBoard 中的标签
                torch.stack(gt_mask_tb_list, 0),  # 前景遮罩图像堆叠
                global_step=iteration,  # 全局步数
                dataformats="NHWC",  # 数据格式
            )

        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)  # 从列表中弹出下一个体素数量
            reso_cur = N_to_reso(n_voxels, tensorf.aabb)  # 计算当前体素数量对应的分辨率
            nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))  # 更新采样数量
            tensorf.upsample_volume_grid(reso_cur)  # 对动态场景体素网格进行上采样
            tensorf_static.upsample_volume_grid(reso_cur)  # 对静态场景体素网格进行上采样

            if args.lr_upsample_reset:  # 如果设置了上采样时重置学习率
                print("reset lr to initial")
                lr_scale = 1  # 学习率缩放比例 0.1 ** (iteration / args.n_iters)
                if args.optimize_poses:
                    optimizer_pose.param_groups[0]["lr"] = lr_pose  # 将优化器的学习率设置为初始学习率
                if iteration >= args.upsamp_list[3] and args.optimize_focal_length:  # 如果迭代次数超过阈值并且优化焦距长度
                    optimizer_focal.param_groups[0]["lr"] = lr_pose  # 将优化器的学习率设置为初始学习率
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)  # 计算学习率的衰减比例
            # 获取优化器的参数组
            grad_vars = tensorf_static.get_optparam_groups(
                args.lr_init * lr_scale,  # 静态场景张量流的初始学习率乘以缩放比例
                args.lr_basis * lr_scale  # 静态场景基础矩阵的初始学习率乘以缩放比例
            )
            grad_vars.extend(
                tensorf.get_optparam_groups(
                    args.lr_init * lr_scale,  # 动态场景张量流的初始学习率乘以缩放比例
                    args.lr_basis * lr_scale  # 动态场景基础矩阵的初始学习率乘以缩放比例
                )
            )
            # 使用 Adam 优化器进行优化
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

        if iteration > args.n_iters // 2:
            optimizer_pose.param_groups[0]["lr"] = 0.0
            optimizer_focal.param_groups[0]["lr"] = 0.0

    # 将动态场景和静态场景的参数保存到文件中
    tensorf.save(
        poses_mtx.detach().cpu(),  # 分离相机位姿并转移到 CPU
        focal_refine.detach().cpu(),  # 分离焦距并转移到 CPU
        f"{logfolder}/{args.expname}.th",  # 保存路径
    )
    tensorf_static.save(
        poses_mtx.detach().cpu(),  # 分离相机位姿并转移到 CPU
        focal_refine.detach().cpu(),  # 分离焦距并转移到 CPU
        f"{logfolder}/{args.expname}_static.th",  # 保存路径
    )

    # 创建用于保存训练图片的文件夹
    os.makedirs(f"{logfolder}/imgs_train_all", exist_ok=True)

    # 进行训练集评估并保存结果
    PSNRs_train, near_fars, _ = evaluation(
        test_dataset,  # 测试数据集
        poses_mtx,  # 相机位姿
        focal_refine.cpu(),  # 焦距（转移到 CPU）
        tensorf_static,  # 静态场景张量流
        tensorf,  # 动态场景张量流
        args,  # 命令行参数
        renderer,  # 渲染器
        f"{logfolder}/imgs_test_all",  # 保存测试图片的文件夹路径
        N_vis=-1,  # 可视化数量（-1 表示全部）
        N_samples=-1,  # 采样数量（-1 表示全部）
        white_bg=white_bg,  # 是否白色背景
        ray_type=args.ray_type,  # 光线类型
        device=device,  # 设备
    )
    # 打印训练集的平均 PSNR
    print(
        f"======> {args.expname} train all psnr: {np.mean(PSNRs_train)} <========================"
    )
    # 保存 poses_bounds.npy 文件
    poses_saving = poses_mtx.clone()  # 复制相机位姿张量
    poses_saving = torch.cat(  # 拼接变换后的相机位姿
        [-poses_saving[..., 1:2], poses_saving[..., :1], poses_saving[..., 2:4]], -1
    )
    hwf = (  # 构造相机参数张量 [H, W, focal_length]
        torch.from_numpy(np.array([H, W, focal_refine.detach().cpu()]))  # 从 Numpy 数组构造张量并转移到 CPU
        * args.downsample_train  # 调整分辨率
    )
    hwf = torch.stack([hwf] * args.N_voxel_t, 0)[..., None]  # 扩展到与相机位姿相同的形状
    poses_saving = torch.cat([poses_saving.cpu(), hwf], -1).view(args.N_voxel_t, -1)  # 拼接相机位姿和相机参数，并重新形状为每个体素一行
    poses_bounds_saving = (  # 拼接相机位姿和视场范围，并转为 Numpy 数组
        torch.cat([poses_saving, torch.from_numpy(np.array(near_fars))], -1)  # 拼接相机位姿和视场范围
        .detach()  # 分离张量
        .numpy()  # 转为 Numpy 数组
    )
    np.save(  # 保存为 npy 文件
        os.path.join(args.datadir, "poses_bounds_RoDynRF.npy"),  # 保存路径
        poses_bounds_saving  # 要保存的数据
    )



if __name__ == "__main__":
    # 设置PyTorch张量的默认数据类型
    torch.set_default_dtype(torch.float32)
    # 设置随机种子以保证可重复性
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()  # 假设这个函数解析并返回命令行参数
    print(args)  # 打印解析后的参数，以供参考
    # 如果指定了导出网格，则执行导出网格的操作
    if args.export_mesh:
        export_mesh(args)#假设这个函数根据参数导出网格
    if args.render_only and (args.render_test or args.render_path):
        # 如果 render_only 为 True，且同时 render_test 或 render_path 为 True，则执行以下操作
        render_test(args, os.path.join(args.basedir, args.expname))
    else:
        reconstruction(args)
