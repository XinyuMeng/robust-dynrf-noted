# Copyright (c) Meta Platforms, Inc. and affiliates.

import sys
import os
import imageio
import numpy as np
import torch
from tqdm.auto import tqdm

from dataLoader.ray_utils import get_rays
from dataLoader.ray_utils import (
    get_ray_directions_blender,
    get_rays,
    ndc_rays_blender,
    get_rays_with_batch,
)
from dataLoader.ray_utils import ndc_rays_blender
from models.tensoRF import *
from utils import visualize_depth_numpy, visualize_depth, rgb_lpips, rgb_ssim
from camera import lie, pose
from flow_viz import flow_to_image


def OctreeRender_trilinear_fast(
    rays,
    ts,
    timeembeddings,
    tensorf,
    xyz_sampled,
    z_vals_input,
    ray_valid,
    chunk=4096,
    N_samples=-1,
    ray_type="ndc",
    white_bg=True,
    is_train=False,
    device="cuda",
):
    (
        rgbs,
        depth_maps,
        blending_maps,
        pts_refs,
        weights_ds,
        delta_xyzs,
        rgb_points,
        sigmas,
        z_vals,
        dists,
    ) = ([], [], [], [], [], [], [], [], [], [])
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
        ts_chunk = ts[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
        xyz_sampled_chunk = xyz_sampled[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(
            device
        )
        z_vals_chunk = z_vals_input[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(
            device
        )
        ray_valid_chunk = ray_valid[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(
            device
        )

        timeembeddings_chunk = None
        if timeembeddings is not None:
            timeembeddings_chunk = timeembeddings[
                chunk_idx * chunk : (chunk_idx + 1) * chunk
            ].to(device)

        (
            rgb_map,
            depth_map,
            blending_map,
            pts_ref,
            weights_d,
            xyz_prime,
            rgb_point,
            sigma,
            z_val,
            dist,
        ) = tensorf(
            rays_chunk,
            ts_chunk,
            timeembeddings_chunk,
            xyz_sampled_chunk,
            z_vals_chunk,
            ray_valid_chunk,
            is_train=is_train,
            white_bg=white_bg,
            ray_type=ray_type,
            N_samples=N_samples,
        )
        delta_xyz = xyz_prime - xyz_sampled_chunk

        if blending_map is None:
            rgbs.append(rgb_map)
            depth_maps.append(depth_map)
            pts_refs.append(pts_ref)
            weights_ds.append(weights_d)
            rgb_points.append(rgb_point)
            sigmas.append(sigma)
            z_vals.append(z_val)
            dists.append(dist)
            continue

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
        blending_maps.append(blending_map)
        pts_refs.append(pts_ref)
        weights_ds.append(weights_d)
        delta_xyzs.append(delta_xyz)
        rgb_points.append(rgb_point)
        sigmas.append(sigma)
        z_vals.append(z_val)
        dists.append(dist)
    if len(blending_maps) == 0:
        return (
            torch.cat(rgbs),
            torch.cat(depth_maps),
            None,
            torch.cat(pts_refs),
            torch.cat(weights_ds),
            None,
            None,
            torch.cat(rgb_points),
            torch.cat(sigmas),
            torch.cat(z_vals),
            torch.cat(dists),
        )
    else:
        return (
            torch.cat(rgbs),
            torch.cat(depth_maps),
            torch.cat(blending_maps),
            torch.cat(pts_refs),
            torch.cat(weights_ds),
            torch.cat(delta_xyzs),
            None,
            torch.cat(rgb_points),
            torch.cat(sigmas),
            torch.cat(z_vals),
            torch.cat(dists),
        )


# 用于从输入的射线中采样空间中的点
def sampleXYZ(tensorf, rays_train, N_samples, ray_type="ndc", is_train=False):
    # 根据射线类型选择相应的采样函数
    if ray_type == "ndc":  # 如果射线类型为归一化设备坐标（NDC）
        # 调用神经体积渲染模型的 sample_ray_ndc 方法进行采样
        # 返回采样的点坐标、深度值和射线有效性
        xyz_sampled, z_vals, ray_valid = tensorf.sample_ray_ndc(
            rays_train[:, :3],  # 提取射线起点坐标
            rays_train[:, 3:6],  # 提取射线方向
            is_train=is_train,  # 是否处于训练模式
            N_samples=N_samples,  # 每条射线要采样的点数量
        )
    elif ray_type == "contract":  # 如果射线类型为缩放后的射线
        # 调用神经体积渲染模型的 sample_ray_contracted 方法进行采样
        # 返回采样的点坐标、深度值和射线有效性
        xyz_sampled, z_vals, ray_valid = tensorf.sample_ray_contracted(
            rays_train[:, :3],  # 提取射线起点坐标
            rays_train[:, 3:6],  # 提取射线方向
            is_train=is_train,  # 是否处于训练模式
            N_samples=N_samples,  # 每条射线要采样的点数量
        )
    else:  # 其他射线类型
        # 调用神经体积渲染模型的 sample_ray 方法进行采样
        # 返回采样的点坐标、深度值和射线有效性
        xyz_sampled, z_vals, ray_valid = tensorf.sample_ray(
            rays_train[:, :3],  # 提取射线起点坐标
            rays_train[:, 3:6],  # 提取射线方向
            is_train=is_train,  # 是否处于训练模式
            N_samples=N_samples,  # 每条射线要采样的点数量
        )

    # 将深度值扩展为与采样点坐标相同的形状，以便与采样点对齐
    z_vals = torch.tile(z_vals, (xyz_sampled.shape[0], 1))

    # 返回采样的点坐标、深度值和射线有效性
    return xyz_sampled, z_vals, ray_valid


def raw2outputs(
    rgb_s,
    sigma_s,
    rgb_d,
    sigma_d,
    dists,
    blending,
    z_vals,
    rays_chunk,
    is_train=False,
    ray_type="ndc",
):
    """Transforms model's predictions to semantically meaningful values.
    Args:
      raw_d: [num_rays, num_samples along ray, 4]. Prediction from Dynamic model.
      raw_s: [num_rays, num_samples along ray, 4]. Prediction from Static model.
      z_vals: [num_rays, num_samples along ray]. Integration time.
      rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. 光线的估计RGB颜色
      disp_map: [num_rays]. Disparity map. Inverse of depth map. 视差图，即深度图的倒数
      acc_map: [num_rays]. Sum of weights along each ray. 每条光线沿着方向的权重总和
      weights: [num_rays, num_samples]. Weights assigned to each sampled color. 分配给每个采样颜色的权重
      depth_map: [num_rays]. Estimated distance to object.
    """

    # Function for computing density from model prediction. This value is
    # strictly between [0, 1].
    def raw2alpha(sigma, dist):
        return 1.0 - torch.exp(-sigma * dist)

    # # Add noise to model's predictions for density. Can be used to
    # # regularize network during training (prevents floater artifacts).
    # noise = 0.
    # if raw_noise_std > 0.:
    #     noise = torch.randn(raw_d[..., 3].shape) * raw_noise_std

    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point.
    # 预测每个射线沿着每个采样点的密度。较高的值意味着在此点更有可能被吸收。
    alpha_d = raw2alpha(sigma_d, dists)  # [N_rays, N_samples]
    alpha_s = raw2alpha(sigma_s, dists)  # [N_rays, N_samples]
    # 合并动态和静态预测的密度。
    alphas = 1.0 - (1.0 - alpha_s) * (1.0 - alpha_d)  # [N_rays, N_samples]

    # 计算 alpha_d 的累积乘积矩阵 T_d。
    # T_d 矩阵的每一行代表了光线沿着每个射线路径的透明度随采样点变化的累积乘积。
    T_d = torch.cumprod(
        torch.cat(
            [
                torch.ones((alpha_d.shape[0], 1)).to(alpha_d.device), # 首先，创建形状为 (alpha_d.shape[0], 1)的全1张量，这表示了在渲染开始时，光线的初始透明度为1，即没有被任何物体遮挡
                1.0 - alpha_d + 1e-10,#然后，计算当前采样点处的不透明度（alpha_d 表示透明度），并加上一个很小的常数 1e-10，以避免除以零错误
            ],
            -1, # 将上述结果沿着最后一个维度进行拼接
        ),
        -1, #最后，对该张量沿着最后一个维度进行累积乘积操作，得到 T_d，表示了每个采样点处光线的累积透明度
    )[:, :-1] # 由于最后一列对应着当前采样点的透明度，因此在计算中将其移除，得到最终的 T_d 矩阵
    # 计算 alpha_s 的累积乘积矩阵 T_s。
    T_s = torch.cumprod(
        torch.cat(
            [
                torch.ones((alpha_s.shape[0], 1)).to(alpha_s.device),
                1.0 - alpha_s + 1e-10,
            ],
            -1,
        ),
        -1,
    )[:, :-1]

    # 计算 alpha_d 和 alpha_s 组合后的累积乘积矩阵 T_full。
    T_full = torch.cumprod(
        torch.cat(
            [
                torch.ones((alpha_d.shape[0], 1)).to(alpha_d.device),
                (1.0 - alpha_d * blending) * (1.0 - alpha_s * (1.0 - blending)) + 1e-10,
            ],
            -1,
        ),
        -1,
    )[:, :-1]

    # Compute weight for RGB of each sample along each ray.  A cumprod() is
    # used to express the idea of the ray not having reflected up to this
    # sample yet.
    # 计算每个采样点沿每个射线的 RGB 权重。
    # 使用 cumprod() 表达射线到当前采样点的反射情况。
    weights_d = alpha_d * T_d
    weights_s = alpha_s * T_s
    weights_d = weights_d / (torch.sum(weights_d, -1, keepdim=True) + 1e-10)
    weights_full = (alpha_d * blending + alpha_s * (1.0 - blending)) * T_full

    # Computed weighted color of each sample along each ray.
    # 计算每个采样点沿每个射线的加权颜色。
    rgb_map_d = torch.sum(weights_d[..., None] * rgb_d, -2)
    rgb_map_s = torch.sum(weights_s[..., None] * rgb_s, -2)
    rgb_map_full = torch.sum(
        (T_full * alpha_d * blending)[..., None] * rgb_d
        + (T_full * alpha_s * (1.0 - blending))[..., None] * rgb_s,
        -2,
    )

    # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
    # 沿每条射线的权重之和。该值在 [0, 1] 之间，存在数值误差。
    acc_map_d = torch.sum(weights_d, -1)  # 计算动态模型预测的每条光线上各个采样点处的权重累加和，表示动态场景中光线各部分的累积光线强度或积累光量
    acc_map_s = torch.sum(weights_s, -1)  # 计算静态模型预测的每条光线上各个采样点处的权重累加和，表示静态场景中光线各部分的累积光线强度或积累光量
    acc_map_full = torch.sum(weights_full, -1)  # 计算综合考虑了动态模型和静态模型预测的每条光线上各个采样点处的权重累加和，表示整个场景中光线各部分的累积光线强度或积累光量

    if is_train and torch.rand((1,)) < 0.5:  # 如果处于训练模式，并且以50%的概率执行以下操作
        # 对动态模型预测的 RGB 图进行调整，增加未被累积的光线强度，使得光线更加明亮
        rgb_map_d = rgb_map_d + (1.0 - acc_map_d[..., None])
        # 对静态模型预测的 RGB 图进行调整，增加未被累积的光线强度，使得光线更加明亮
        rgb_map_s = rgb_map_s + (1.0 - acc_map_s[..., None])
        # 对综合考虑了动态模型和静态模型预测的 RGB 图进行调整，增加未被累积的光线强度，使得光线更加明亮，并确保结果非负
        rgb_map_full = rgb_map_full + torch.relu(1.0 - acc_map_full[..., None])

    # 估计的深度图表示期望的距离。
    # Estimated depth map is expected distance.
    depth_map_d = torch.sum(weights_d * z_vals, -1)
    depth_map_s = torch.sum(weights_s * z_vals, -1)
    depth_map_full = torch.sum(weights_full * z_vals, -1)

    if ray_type == "ndc": # 如果光线类型为 ndc（归一化设备坐标）
        # 对动态模型估计的深度图进行调整，以考虑光线未被累积的部分，其中使用了相机到近面的距离
        depth_map_d = depth_map_d + (1.0 - acc_map_d) * (
            rays_chunk[..., 2] + rays_chunk[..., -1]
        )
        # 对静态模型估计的深度图进行调整，以考虑光线未被累积的部分，其中使用了相机到近面的距离
        depth_map_s = depth_map_s + (1.0 - acc_map_s) * (
            rays_chunk[..., 2] + rays_chunk[..., -1]
        )
        # 对综合考虑了动态模型和静态模型估计的深度图进行调整，以考虑光线未被累积的部分，其中使用了相机到近面的距离
        depth_map_full = depth_map_full + torch.relu(1.0 - acc_map_full) * (
            rays_chunk[..., 2] + rays_chunk[..., -1]
        )
    elif ray_type == "contract":# 如果光线类型为 contract（合同坐标）
        depth_map_d = depth_map_d + (1.0 - acc_map_d) * 256.0
        # 对静态模型估计的深度图进行调整，以考虑光线未被累积的部分，其中使用了一个固定的距离值（256.0）
        depth_map_s = depth_map_s + (1.0 - acc_map_s) * 256.0
        # 对综合考虑了动态模型和静态模型估计的深度图进行调整，以考虑光线未被累积的部分，其中使用了一个固定的距离值（256.0）
        depth_map_full = depth_map_full + torch.relu(1.0 - acc_map_full) * 256.0


    # 将 RGB 图像限制在 [0, 1] 范围内，确保图像像素值在合理范围内
    rgb_map_d = rgb_map_d.clamp(0, 1)
    rgb_map_s = rgb_map_s.clamp(0, 1)
    rgb_map_full = rgb_map_full.clamp(0, 1)

    # Computed dynamicness
    # 计算动态性映射
    dynamicness_map = torch.sum(weights_full * blending, -1)  # 根据权重和混合度计算动态性
    dynamicness_map = dynamicness_map + torch.relu(1.0 - acc_map_full) * 0.0  # 根据累积权重计算动态性（如果光线没有累积，则动态性为0）

    return (
        rgb_map_full,  # 完整场景的 RGB 图像
        depth_map_full,  # 完整场景的深度图
        acc_map_full,  # 完整场景的累积权重
        weights_full,  # 完整场景的权重
        rgb_map_s,  # 静态模型的 RGB 图像
        depth_map_s,  # 静态模型的深度图
        acc_map_s,  # 静态模型的累积权重
        weights_s,  # 静态模型的权重
        rgb_map_d,  # 动态模型的 RGB 图像
        depth_map_d,  # 动态模型的深度图
        acc_map_d,  # 动态模型的累积权重
        weights_d,  # 动态模型的权重
        dynamicness_map,  # 动态性映射
    )


@torch.no_grad()
def render(
    test_dataset,
    poses_mtx,
    focal_ratio_refine,
    tensorf_static,
    tensorf,
    args,
    renderer,
    savePath=None,
    N_vis=5,
    prtx="",
    N_samples=-1,
    white_bg=False,
    ray_type="ndc",
    compute_extra_metrics=True,
    device="cuda",
):
    (
        rgb_maps_tb,
        depth_maps_tb,
        blending_maps_tb,
        gt_rgbs_tb,
        induced_flow_f_tb,
        induced_flow_b_tb,
        induced_flow_s_f_tb,
        induced_flow_s_b_tb,
        delta_xyz_tb,
        rgb_maps_s_tb,
        depth_maps_s_tb,
        rgb_maps_d_tb,
        depth_maps_d_tb,
    ) = ([], [], [], [], [], [], [], [], [], [], [], [], [])

    W, H = test_dataset.img_wh
    directions = get_ray_directions_blender(
        H, W, [focal_ratio_refine, focal_ratio_refine]
    ).to(
        poses_mtx.device
    )  # (H, W, 3)
    all_rays = []
    for i in range(poses_mtx.shape[0]):
        c2w = poses_mtx[i]
        rays_o, rays_d = get_rays(directions, c2w)  # both (h*w, 3)
        if ray_type == "ndc":
            rays_o, rays_d = ndc_rays_blender(
                H, W, focal_ratio_refine, 1.0, rays_o, rays_d
            )
        all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
    all_rays = torch.stack(all_rays, 0)
    if args.multiview_dataset:
        # duplicate poses for multiple time instances
        all_rays = torch.tile(all_rays, (args.N_voxel_t, 1, 1))

    ii, jj = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )
    grid = torch.from_numpy(np.stack([ii, jj], -1)).view(-1, 2).to(device)

    img_eval_interval = 1 if N_vis < 0 else max(all_rays.shape[0] // N_vis, 1)
    idxs = list(range(0, all_rays.shape[0], img_eval_interval))
    for idx, samples in enumerate(all_rays[0::img_eval_interval]):
        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])
        ts = test_dataset.all_ts[idx].view(-1)

        N_rays_all = rays.shape[0]
        chunk = 1024
        pose_f = poses_mtx[min(idx + 1, poses_mtx.shape[0] - 1), :3, :4]
        pose_b = poses_mtx[max(idx - 1, 0), :3, :4]
        rgb_map_list = []
        rgb_map_s_list = []
        rgb_map_d_list = []
        dynamicness_map_list = []
        depth_map_list = []
        depth_map_s_list = []
        depth_map_d_list = []
        induced_flow_f_list = []
        induced_flow_b_list = []
        induced_flow_s_f_list = []
        induced_flow_s_b_list = []
        weights_d_list = []
        delta_xyz_list = []
        for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
            rays_chunk = rays[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
            ts_chunk = ts[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
            grid_chunk = grid[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
            xyz_sampled, z_vals, ray_valid = sampleXYZ(
                tensorf,
                rays_chunk,
                N_samples=N_samples,
                ray_type=ray_type,
                is_train=False,
            )
            # static
            (
                _,
                _,
                _,
                pts_ref_s,
                _,
                _,
                rgb_point_static,
                sigma_static,
                _,
                _,
            ) = tensorf_static(
                rays_chunk,
                ts_chunk,
                None,
                xyz_sampled,
                z_vals,
                ray_valid,
                is_train=False,
                white_bg=white_bg,
                ray_type=ray_type,
                N_samples=N_samples,
            )
            # dynamic
            (
                _,
                _,
                blending,
                pts_ref,
                _,
                xyz_prime,
                rgb_point_dynamic,
                sigma_dynamic,
                z_val_dynamic,
                dist_dynamic,
            ) = tensorf(
                rays_chunk,
                ts_chunk,
                None,
                xyz_sampled,
                z_vals,
                ray_valid,
                is_train=False,
                white_bg=white_bg,
                ray_type=ray_type,
                N_samples=N_samples,
            )
            delta_xyz = xyz_prime - xyz_sampled
            # blending
            (
                rgb_map,
                depth_map_full,
                acc_map_full,
                weights_full,
                rgb_map_s,
                depth_map_s,
                acc_map_s,
                weights_s,
                rgb_map_d,
                depth_map_d,
                acc_map_d,
                weights_d,
                dynamicness_map,
            ) = raw2outputs(
                rgb_point_static,
                sigma_static,
                rgb_point_dynamic.to(device),
                sigma_dynamic.to(device),
                dist_dynamic.to(device),
                blending,
                z_val_dynamic.to(device),
                rays_chunk,
                ray_type=ray_type,
            )
            # scene flow
            scene_flow_f, scene_flow_b = tensorf.module.get_forward_backward_scene_flow(
                pts_ref, ts_chunk
            )
            pts_f = pts_ref + scene_flow_f
            pts_b = pts_ref + scene_flow_b
            induced_flow_f, _ = induce_flow(
                H,
                W,
                focal_ratio_refine,
                torch.tile(pose_f[None], (weights_d.shape[0], 1, 1)),
                weights_d,
                pts_f,
                grid_chunk,
                rays_chunk,
                ray_type=ray_type,
            )
            induced_flow_b, _ = induce_flow(
                H,
                W,
                focal_ratio_refine,
                torch.tile(pose_b[None], (weights_d.shape[0], 1, 1)),
                weights_d,
                pts_b,
                grid_chunk,
                rays_chunk,
                ray_type=ray_type,
            )
            # induced flow for static
            induced_flow_s_f, _ = induce_flow(
                H,
                W,
                focal_ratio_refine,
                torch.tile(pose_f[None], (weights_s.shape[0], 1, 1)),
                weights_s,
                pts_ref_s,
                grid_chunk,
                rays_chunk,
                ray_type=ray_type,
            )
            induced_flow_s_b, _ = induce_flow(
                H,
                W,
                focal_ratio_refine,
                torch.tile(pose_b[None], (weights_s.shape[0], 1, 1)),
                weights_s,
                pts_ref_s,
                grid_chunk,
                rays_chunk,
                ray_type=ray_type,
            )
            # gather chunks
            rgb_map_list.append(rgb_map)
            rgb_map_s_list.append(rgb_map_s)
            rgb_map_d_list.append(rgb_map_d)
            dynamicness_map_list.append(dynamicness_map)
            depth_map_list.append(depth_map_full)
            depth_map_s_list.append(depth_map_s)
            depth_map_d_list.append(depth_map_d)
            induced_flow_f_list.append(induced_flow_f)
            induced_flow_b_list.append(induced_flow_b)
            induced_flow_s_f_list.append(induced_flow_s_f)
            induced_flow_s_b_list.append(induced_flow_s_b)
            weights_d_list.append(weights_d)
            delta_xyz_list.append(delta_xyz)
        rgb_map = torch.cat(rgb_map_list)
        rgb_map_s = torch.cat(rgb_map_s_list)
        rgb_map_d = torch.cat(rgb_map_d_list)
        dynamicness_map = torch.cat(dynamicness_map_list)
        depth_map = torch.cat(depth_map_list)
        depth_map_s = torch.cat(depth_map_s_list)
        depth_map_d = torch.cat(depth_map_d_list)
        induced_flow_f = torch.cat(induced_flow_f_list)
        induced_flow_b = torch.cat(induced_flow_b_list)
        induced_flow_s_f = torch.cat(induced_flow_s_f_list)
        induced_flow_s_b = torch.cat(induced_flow_s_b_list)
        weights_d = torch.cat(weights_d_list)
        delta_xyzs = torch.cat(delta_xyz_list)

        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map_s = rgb_map_s.clamp(0.0, 1.0)
        rgb_map_d = rgb_map_d.clamp(0.0, 1.0)
        blending_map = dynamicness_map.clamp(0.0, 1.0)

        rgb_map_s, depth_map_s = rgb_map_s.reshape(H, W, 3), depth_map_s.reshape(H, W)
        rgb_map_d, depth_map_d = rgb_map_d.reshape(H, W, 3), depth_map_d.reshape(H, W)
        rgb_map, depth_map, blending_map = (
            rgb_map.reshape(H, W, 3),
            depth_map.reshape(H, W),
            blending_map.reshape(H, W),
        )
        if ray_type == "contract":
            depth_map_s = -1.0 / (depth_map_s + 1e-6)
            depth_map_d = -1.0 / (depth_map_d + 1e-6)
            depth_map = -1.0 / (depth_map + 1e-6)

        gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)

        viz_induced_flow_f = torch.from_numpy(
            flow_to_image(induced_flow_f.view(H, W, 2).detach().cpu().numpy())
        )
        viz_induced_flow_b = torch.from_numpy(
            flow_to_image(induced_flow_b.view(H, W, 2).detach().cpu().numpy())
        )
        viz_induced_flow_s_f = torch.from_numpy(
            flow_to_image(induced_flow_s_f.view(H, W, 2).detach().cpu().numpy())
        )
        viz_induced_flow_s_b = torch.from_numpy(
            flow_to_image(induced_flow_s_b.view(H, W, 2).detach().cpu().numpy())
        )

        rgb_maps_tb.append(rgb_map)  # HWC
        rgb_maps_s_tb.append(rgb_map_s)  # HWC
        rgb_maps_d_tb.append(rgb_map_d)  # HWC
        depth_maps_tb.append(depth_map)  # CHW
        depth_maps_s_tb.append(depth_map_s)  # CHW
        depth_maps_d_tb.append(depth_map_d)  # CHW
        blending_maps_tb.append(blending_map[None].repeat(3, 1, 1))  # CHW
        gt_rgbs_tb.append(gt_rgb)  # HWC
        induced_flow_f_tb.append(viz_induced_flow_f)
        induced_flow_b_tb.append(viz_induced_flow_b)
        induced_flow_s_f_tb.append(viz_induced_flow_s_f)
        induced_flow_s_b_tb.append(viz_induced_flow_s_b)
        delta_xyz_sum = torch.sum(weights_d[..., None] * delta_xyzs, 1).view(H, W, 3)
        delta_xyz_tb.append(
            ((delta_xyz_sum / torch.max(torch.abs(delta_xyz_sum))) + 1.0) / 2.0
        )

    tmp_list = []
    tmp_list.extend(depth_maps_tb)
    tmp_list.extend(depth_maps_s_tb)
    tmp_list.extend(depth_maps_d_tb)
    all_depth = torch.stack(tmp_list)
    depth_map_min = torch.min(all_depth).item()
    depth_map_max = torch.max(all_depth).item()
    for idx, (depth_map, depth_map_s, depth_map_d) in enumerate(
        zip(depth_maps_tb, depth_maps_s_tb, depth_maps_d_tb)
    ):
        depth_maps_tb[idx] = visualize_depth(
            torch.clamp(depth_map, min=depth_map_min, max=depth_map_max),
            (depth_map_min, depth_map_max),
        )[0]
        depth_maps_s_tb[idx] = visualize_depth(
            torch.clamp(depth_map_s, min=depth_map_min, max=depth_map_max),
            (depth_map_min, depth_map_max),
        )[0]
        depth_maps_d_tb[idx] = visualize_depth(
            torch.clamp(depth_map_d, min=depth_map_min, max=depth_map_max),
            (depth_map_min, depth_map_max),
        )[0]

    monodepth_tb = []
    for i in range(test_dataset.all_disps.shape[0]):
        monodepth_tb.append(visualize_depth(test_dataset.all_disps[i])[0])

    return (
        rgb_maps_tb,
        depth_maps_tb,
        blending_maps_tb,
        gt_rgbs_tb,
        induced_flow_f_tb,
        induced_flow_b_tb,
        induced_flow_s_f_tb,
        induced_flow_s_b_tb,
        delta_xyz_tb,
        rgb_maps_s_tb,
        depth_maps_s_tb,
        rgb_maps_d_tb,
        depth_maps_d_tb,
        monodepth_tb,
    )


@torch.no_grad()
def evaluation(
    test_dataset,
    poses_mtx,
    focal_ratio_refine,
    tensorf_static,
    tensorf,
    args,
    renderer,
    savePath=None,
    N_vis=5,
    prtx="",
    N_samples=-1,
    white_bg=False,
    ray_type="ndc",
    compute_extra_metrics=True,
    device="cuda",
):
    # 定义一系列变量用于存储评估结果和中间数据
    (
        PSNRs,
        rgb_maps,
        depth_maps,
        rgb_maps_s,
        depth_maps_s,
        rgb_maps_d,
        depth_maps_d,
        blending_maps,
    ) = ([], [], [], [], [], [], [], [])
    near_fars = []
    ssims, l_alex, l_vgg = [], [], []
    # 创建保存路径
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/rgbd", exist_ok=True)
    os.makedirs(savePath + "_static/rgbd", exist_ok=True)
    os.makedirs(savePath + "_dynamic/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    # 获取测试集中的近距离和远距离信息
    near_far = test_dataset.near_far

    W, H = test_dataset.img_wh
    # 计算射线方向
    directions = get_ray_directions_blender(
        H, W, [focal_ratio_refine, focal_ratio_refine]
    ).to(
        poses_mtx.device
    )  # (H, W, 3)
    all_rays = []
    # 对每个相机位姿计算射线
    for i in range(poses_mtx.shape[0]):
        c2w = poses_mtx[i]
        rays_o, rays_d = get_rays(directions, c2w)  # both (h*w, 3)
        if ray_type == "ndc":
            rays_o, rays_d = ndc_rays_blender(
                H, W, focal_ratio_refine, 1.0, rays_o, rays_d
            )
        all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
    all_rays = torch.stack(all_rays, 0)
    if args.multiview_dataset:
        # duplicate poses for multiple time instances
        # 对多视角数据集进行扩展，以便每个时间步都有相同的数据
        all_rays = torch.tile(all_rays, (args.N_voxel_t, 1, 1))

    # 计算评估图像的间隔
    img_eval_interval = 1 if N_vis < 0 else max(all_rays.shape[0] // N_vis, 1)
    idxs = list(range(0, all_rays.shape[0], img_eval_interval))
    # 对每个间隔进行评估
    for idx, samples in tqdm(
        enumerate(all_rays[0::img_eval_interval]), file=sys.stdout
    ):
        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])
        ts = test_dataset.all_ts[idx].view(-1)

        N_rays_all = rays.shape[0]
        chunk = 512
        # 分块处理射线
        rgb_map_list = []
        depth_map_list = []
        rgb_map_s_list = []
        depth_map_s_list = []
        rgb_map_d_list = []
        depth_map_d_list = []
        dynamicness_map_list = []
        for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
            rays_chunk = rays[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
            ts_chunk = ts[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
            xyz_sampled, z_vals, ray_valid = sampleXYZ(
                tensorf,
                rays_chunk,
                N_samples=N_samples,
                ray_type=ray_type,
                is_train=False,
            )
            # static
            # 静态部分的渲染
            _, _, _, _, _, _, rgb_point_static, sigma_static, _, _ = tensorf_static(
                rays_chunk,
                ts_chunk,
                None,
                xyz_sampled,
                z_vals,
                ray_valid,
                is_train=False,
                white_bg=white_bg,
                ray_type=ray_type,
                N_samples=N_samples,
            )
            # dynamic
            # 动态部分的渲染
            (
                _,
                _,
                blending,
                pts_ref,
                _,
                _,
                rgb_point_dynamic,
                sigma_dynamic,
                z_val_dynamic,
                dist_dynamic,
            ) = tensorf(
                rays_chunk,
                ts_chunk,
                None,
                xyz_sampled,
                z_vals,
                ray_valid,
                is_train=False,
                white_bg=white_bg,
                ray_type=ray_type,
                N_samples=N_samples,
            )
            # blending
            (
                rgb_map,  # RGB颜色图
                depth_map_full,  # 完整深度图
                acc_map_full,  # 完整累积图
                weights_full,  # 完整权重图
                rgb_map_s,  # 静态模型渲染的RGB颜色图
                depth_map_s,  # 静态模型渲染的深度图
                acc_map_s,  # 静态模型渲染的累积图
                weights_s,  # 静态模型渲染的权重图
                rgb_map_d,  # 动态模型渲染的RGB颜色图
                depth_map_d,  # 动态模型渲染的深度图
                acc_map_d,  # 动态模型渲染的累积图
                weights_d,  # 动态模型渲染的权重图
                dynamicness_map,  # 动态性地图
            ) = raw2outputs(
                rgb_point_static,  # 静态模型渲染的RGB点
                sigma_static,  # 静态模型渲染的Sigma值
                rgb_point_dynamic.to(device),  # 动态模型渲染的RGB点
                sigma_dynamic.to(device),  # 动态模型渲染的Sigma值
                dist_dynamic.to(device),  # 动态模型渲染的距离值
                blending,  # 混合值
                z_val_dynamic.to(device),  # 动态模型渲染的深度值
                rays_chunk,  # 射线块
                ray_type=ray_type,  # 射线类型
            )
            # 收集每个chunk渲染得到的RGB颜色图，完整深度图，静态模型渲染的RGB颜色图，静态模型渲染的深度图，动态模型渲染的RGB颜色图，动态模型渲染的深度图，以及动态性地图
            rgb_map_list.append(rgb_map)
            depth_map_list.append(depth_map_full)
            rgb_map_s_list.append(rgb_map_s)
            depth_map_s_list.append(depth_map_s)
            rgb_map_d_list.append(rgb_map_d)
            depth_map_d_list.append(depth_map_d)
            dynamicness_map_list.append(dynamicness_map)
        # 将列表中的各个张量连接成一个张量
        rgb_map = torch.cat(rgb_map_list)
        depth_map = torch.cat(depth_map_list)
        rgb_map_s = torch.cat(rgb_map_s_list)
        depth_map_s = torch.cat(depth_map_s_list)
        rgb_map_d = torch.cat(rgb_map_d_list)
        depth_map_d = torch.cat(depth_map_d_list)
        dynamicness_map = torch.cat(dynamicness_map_list)

        # 将RGB颜色图，静态模型渲染的RGB颜色图，动态模型渲染的RGB颜色图，以及动态性地图的值限制在0到1之间
        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map_s = rgb_map_s.clamp(0.0, 1.0)
        rgb_map_d = rgb_map_d.clamp(0.0, 1.0)
        blending_map = dynamicness_map.clamp(0.0, 1.0)

        # 将RGB颜色图和深度图重塑为指定的形状，并转移到CPU上
        rgb_map, depth_map = (
            rgb_map.reshape(H, W, 3).cpu(),# 重塑RGB颜色图为指定形状，并转移到CPU上
            depth_map.reshape(H, W).cpu(),
        )
        rgb_map_s, depth_map_s = (
            rgb_map_s.reshape(H, W, 3).cpu(),
            depth_map_s.reshape(H, W).cpu(),
        )
        rgb_map_d, depth_map_d = (
            rgb_map_d.reshape(H, W, 3).cpu(),
            depth_map_d.reshape(H, W).cpu(),
        )
        blending_map = blending_map.reshape(H, W).cpu()

        # 根据光线类型确定近远值，并添加到近远值列表中
        if ray_type == "contract":# 如果光线类型为contract
            near_fars.append(     # 将近远值添加到近远值列表中
                (
                    torch.quantile(depth_map_s, 0.01).item(),# 添加深度图的1%分位数作为近值
                    torch.quantile(depth_map_s, 0.99).item(),# 添加深度图的99%分位数作为远值
                )
            )
        else:# 如果光线类型不为contract
            near_fars.append(
                (
                    torch.quantile(1.0 / (depth_map_s + 1e-6), 0.01).item(),# 添加深度图的倒数的1%分位数作为近值
                    torch.quantile(1.0 / (depth_map_s + 1e-6), 0.99).item(),# 添加深度图的倒数的99%分位数作为远值
                )
            )
        if ray_type == "contract": # 如果光线类型为contract
            depth_map_s = -1.0 / (depth_map_s + 1e-6) # 计算静态模型渲染的深度图的倒数
            depth_map_d = -1.0 / (depth_map_d + 1e-6)
            depth_map = -1.0 / (depth_map + 1e-6)

        # 保存静态模型渲染的深度图
        np.save(f"{savePath}_static/rgbd/{prtx}{idx:03d}.npy", depth_map_s)
        # 保存深度图
        np.save(f"{savePath}/rgbd/{prtx}{idx:03d}.npy", depth_map)

        # 如果测试数据集中包含RGB图像
        if len(test_dataset.all_rgbs):
            # 获取真实的RGB图像
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            # 计算渲染图像与真实图像之间的像素均方误差
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            # 计算峰值信噪比并添加到列表中
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            # 如果需要计算额外的评估指标
            if compute_extra_metrics:
                # 计算结构相似性指数(SSIM)
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                # 计算基于AlexNet的感知距离
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), "alex", tensorf.device)
                # 计算基于VGG网络的感知距离
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), "vgg", tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype("uint8")
        rgb_map_s = (rgb_map_s.numpy() * 255).astype("uint8")
        rgb_map_d = (rgb_map_d.numpy() * 255).astype("uint8")
        blending_map = (blending_map.numpy() * 255).astype("uint8")
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        rgb_maps_s.append(rgb_map_s)
        depth_maps_s.append(depth_map_s)
        rgb_maps_d.append(rgb_map_d)
        depth_maps_d.append(depth_map_d)
        blending_maps.append(blending_map)
        if savePath is not None:
            imageio.imwrite(f"{savePath}/{prtx}{idx:03d}.png", rgb_map, format="png")
            imageio.imwrite(
                f"{savePath}_static/{prtx}{idx:03d}.png", rgb_map_s, format="png"
            )
            imageio.imwrite(
                f"{savePath}_dynamic/{prtx}{idx:03d}.png", rgb_map_d, format="png"
            )
            imageio.imwrite(
                f"{savePath}_dynamic/{prtx}{idx:03d}_blending.png",
                blending_map,
                format="png",
            )

    depth_list = []
    depth_list.extend(depth_maps)
    # tmp_list.extend(depth_maps_s)
    # tmp_list.extend(depth_maps_d)
    # all_depth = torch.stack(tmp_list)
    # depth_map_min = torch.min(all_depth).item()
    # depth_map_max = torch.max(all_depth).item()
    # all_depth = torch.stack(tmp_list)
    # depth_map_min = torch.quantile(all_depth[:, ::4, ::4], 0.05).item()
    # depth_map_max = torch.quantile(all_depth[:, ::4, ::4], 0.95).item()
    # for idx in range(len(rgb_maps)):
    #     depth_maps[idx] = visualize_depth_numpy(torch.clamp(depth_maps[idx], min=depth_map_min, max=depth_map_max).numpy(), (depth_map_min, depth_map_max))[0]
    #     depth_maps_s[idx] = visualize_depth_numpy(torch.clamp(depth_maps_s[idx], min=depth_map_min, max=depth_map_max).numpy(), (depth_map_min, depth_map_max))[0]
    #     depth_maps_d[idx] = visualize_depth_numpy(torch.clamp(depth_maps_d[idx], min=depth_map_min, max=depth_map_max).numpy(), (depth_map_min, depth_map_max))[0]
    #     if savePath is not None:
    #         imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', depth_maps[idx], format='png')
    #         imageio.imwrite(f'{savePath}_static/rgbd/{prtx}{idx:03d}.png', depth_maps_s[idx], format='png')
    #         imageio.imwrite(f'{savePath}_dynamic/rgbd/{prtx}{idx:03d}.png', depth_maps_d[idx], format='png')

    imageio.mimwrite(
        f"{savePath}/{prtx}video.mp4",
        np.stack(rgb_maps),
        fps=30,
        quality=8,
        format="ffmpeg",
        output_params=["-f", "mp4"],
    )
    # imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8, format='ffmpeg', output_params=["-f", "mp4"])
    imageio.mimwrite(
        f"{savePath}_static/{prtx}video.mp4",
        np.stack(rgb_maps_s),
        fps=30,
        quality=8,
        format="ffmpeg",
        output_params=["-f", "mp4"],
    )
    # imageio.mimwrite(f'{savePath}_static/{prtx}depthvideo.mp4', np.stack(depth_maps_s), fps=30, quality=8, format='ffmpeg', output_params=["-f", "mp4"])
    imageio.mimwrite(
        f"{savePath}_dynamic/{prtx}video.mp4",
        np.stack(rgb_maps_d),
        fps=30,
        quality=8,
        format="ffmpeg",
        output_params=["-f", "mp4"],
    )
    # imageio.mimwrite(f'{savePath}_dynamic/{prtx}depthvideo.mp4', np.stack(depth_maps_d), fps=30, quality=8, format='ffmpeg', output_params=["-f", "mp4"])
    # imageio.mimwrite(f'{savePath}_dynamic/{prtx}blending.mp4', np.stack(blending_maps), fps=30, quality=8, format='ffmpeg', output_params=["-f", "mp4"])

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f"{savePath}/{prtx}mean.txt", np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f"{savePath}/{prtx}mean.txt", np.asarray([psnr]))

    return PSNRs, near_fars, depth_list


@torch.no_grad()
def evaluation_path(
    test_dataset,
    focal_ratio_refine,
    tensorf_static,
    tensorf,
    c2ws,
    renderer,
    savePath=None,
    N_vis=5,
    prtx="",
    N_samples=-1,
    white_bg=False,
    ray_type="ndc",
    compute_extra_metrics=True,
    device="cuda",
    change_view=True,
    change_time=None,
    evaluation=False,
    render_focal=None,
):
    (
        PSNRs,
        rgb_maps,
        depth_maps,
        rgb_maps_s,
        depth_maps_s,
        rgb_maps_d,
        depth_maps_d,
        blending_maps,
    ) = ([], [], [], [], [], [], [], [])
    ssims, l_alex, l_vgg = [], [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/rgbd", exist_ok=True)
    os.makedirs(savePath + "_static/rgbd", exist_ok=True)
    os.makedirs(savePath + "_dynamic/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):
        if render_focal is not None:
            focal_ratio_refine = render_focal[idx]

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)

        W, H = test_dataset.img_wh
        directions = get_ray_directions_blender(
            H, W, [focal_ratio_refine, focal_ratio_refine]
        ).to(
            c2w.device
        )  # (H, W, 3)

        rays_o, rays_d = get_rays(directions, c2w)  # both (h*w, 3)
        if ray_type == "ndc":
            rays_o, rays_d = ndc_rays_blender(
                H, W, focal_ratio_refine, 1.0, rays_o, rays_d
            )
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        if change_time is "change":
            ts = (
                round(idx / (len(c2ws) - 1) * (len(c2ws) - 1)) / (len(c2ws) - 1) * 2.0
                - 1.0
            ) * torch.ones(
                W * H
            )  # discrete time rendering
        else:
            ts = change_time * torch.ones(W * H)  # first time instance

        N_rays_all = rays.shape[0]
        chunk = 8192
        allposes_refine_train = torch.tile(c2w.cpu()[None], (rays.shape[0], 1, 1))
        rgb_map_list = []
        depth_map_list = []
        rgb_map_s_list = []
        depth_map_s_list = []
        rgb_map_d_list = []
        depth_map_d_list = []
        dynamicness_map_list = []
        weights_s_list = []
        for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
            rays_chunk = rays[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
            ts_chunk = ts[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
            allposes_refine_train_chunk = allposes_refine_train[
                chunk_idx * chunk : (chunk_idx + 1) * chunk
            ].to(device)
            xyz_sampled, z_vals, ray_valid = sampleXYZ(
                tensorf,
                rays_chunk,
                N_samples=N_samples,
                ray_type=ray_type,
                is_train=False,
            )
            # static
            _, _, _, _, _, _, rgb_point_static, sigma_static, _, _ = tensorf_static(
                rays_chunk,
                ts_chunk,
                None,
                xyz_sampled,
                z_vals,
                ray_valid,
                is_train=False,
                white_bg=white_bg,
                ray_type=ray_type,
                N_samples=N_samples,
            )
            # dynamic
            (
                _,
                _,
                blending,
                pts_ref,
                _,
                _,
                rgb_point_dynamic,
                sigma_dynamic,
                z_val_dynamic,
                dist_dynamic,
            ) = tensorf(
                rays_chunk,
                ts_chunk,
                None,
                xyz_sampled,
                z_vals,
                ray_valid,
                is_train=False,
                white_bg=white_bg,
                ray_type=ray_type,
                N_samples=N_samples,
            )
            # blending
            (
                rgb_map,
                depth_map_full,
                acc_map_full,
                weights_full,
                rgb_map_s,
                depth_map_s,
                acc_map_s,
                weights_s,
                rgb_map_d,
                depth_map_d,
                acc_map_d,
                weights_d,
                dynamicness_map,
            ) = raw2outputs(
                rgb_point_static,
                sigma_static,
                rgb_point_dynamic.to(device),
                sigma_dynamic.to(device),
                dist_dynamic.to(device),
                blending,
                z_val_dynamic.to(device),
                rays_chunk,
                ray_type=ray_type,
            )
            # gather chunks
            rgb_map_list.append(rgb_map)
            depth_map_list.append(depth_map_full)
            rgb_map_s_list.append(rgb_map_s)
            depth_map_s_list.append(depth_map_s)
            rgb_map_d_list.append(rgb_map_d)
            depth_map_d_list.append(depth_map_d)
            dynamicness_map_list.append(dynamicness_map)
            weights_s_list.append(weights_s)
        rgb_map = torch.cat(rgb_map_list)
        depth_map = torch.cat(depth_map_list)
        rgb_map_s = torch.cat(rgb_map_s_list)
        depth_map_s = torch.cat(depth_map_s_list)
        rgb_map_d = torch.cat(rgb_map_d_list)
        depth_map_d = torch.cat(depth_map_d_list)
        dynamicness_map = torch.cat(dynamicness_map_list)
        weights_s__ = torch.cat(weights_s_list)
        weights_s__ = weights_s__.reshape(H, W, -1)

        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map_s = rgb_map_s.clamp(0.0, 1.0)
        rgb_map_d = rgb_map_d.clamp(0.0, 1.0)
        blending_map = dynamicness_map.clamp(0.0, 1.0)

        rgb_map, depth_map = (
            rgb_map.reshape(H, W, 3).cpu(),
            depth_map.reshape(H, W).cpu(),
        )
        rgb_map_s, depth_map_s = (
            rgb_map_s.reshape(H, W, 3).cpu(),
            depth_map_s.reshape(H, W).cpu(),
        )
        rgb_map_d, depth_map_d = (
            rgb_map_d.reshape(H, W, 3).cpu(),
            depth_map_d.reshape(H, W).cpu(),
        )
        blending_map = blending_map.reshape(H, W).cpu()
        if ray_type == "contract":
            depth_map_s = -1.0 / (depth_map_s + 1e-6)
            depth_map_d = -1.0 / (depth_map_d + 1e-6)
            depth_map = -1.0 / (depth_map + 1e-6)

        rgb_map = (rgb_map.numpy() * 255).astype("uint8")
        rgb_map_s = (rgb_map_s.numpy() * 255).astype("uint8")
        rgb_map_d = (rgb_map_d.numpy() * 255).astype("uint8")
        blending_map = (blending_map.numpy() * 255).astype("uint8")
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        rgb_maps_s.append(rgb_map_s)
        depth_maps_s.append(depth_map_s)
        rgb_maps_d.append(rgb_map_d)
        depth_maps_d.append(depth_map_d)
        blending_maps.append(blending_map)
        if savePath is not None:
            if evaluation:
                imageio.imwrite(
                    f"{savePath}/{prtx}v000_t{idx:03d}.png", rgb_map, format="png"
                )
            else:
                imageio.imwrite(
                    f"{savePath}/{prtx}{idx:03d}.png", rgb_map, format="png"
                )
            imageio.imwrite(
                f"{savePath}_static/{prtx}{idx:03d}.png", rgb_map_s, format="png"
            )
            imageio.imwrite(
                f"{savePath}_dynamic/{prtx}{idx:03d}.png", rgb_map_d, format="png"
            )
            imageio.imwrite(
                f"{savePath}_dynamic/{prtx}{idx:03d}_blending.png",
                blending_map,
                format="png",
            )

    if evaluation:
        return

    depth_list = []
    depth_list.extend(depth_maps)
    # tmp_list.extend(depth_maps_s)
    # tmp_list.extend(depth_maps_d)
    # all_depth = torch.stack(tmp_list)

    # for idx in range(len(rgb_maps)):
    #     depth_maps[idx] = visualize_depth_numpy(torch.clamp(depth_maps[idx], min=depth_map_min, max=depth_map_max).numpy(), (depth_map_min, depth_map_max))[0]
    #     depth_maps_s[idx] = visualize_depth_numpy(torch.clamp(depth_maps_s[idx], min=depth_map_min, max=depth_map_max).numpy(), (depth_map_min, depth_map_max))[0]
    #     depth_maps_d[idx] = visualize_depth_numpy(torch.clamp(depth_maps_d[idx], min=depth_map_min, max=depth_map_max).numpy(), (depth_map_min, depth_map_max))[0]
    #     if savePath is not None:
    #         imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', depth_maps[idx], format='png')
    #         imageio.imwrite(f'{savePath}_static/rgbd/{prtx}{idx:03d}.png', depth_maps_s[idx], format='png')
    #         imageio.imwrite(f'{savePath}_dynamic/rgbd/{prtx}{idx:03d}.png', depth_maps_d[idx], format='png')

    # if not evaluation:
    imageio.mimwrite(
        f"{savePath}/{prtx}video.mp4",
        np.stack(rgb_maps),
        fps=30,
        quality=8,
        format="ffmpeg",
        output_params=["-f", "mp4"],
    )
    # imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8, format='ffmpeg', output_params=["-f", "mp4"])
    imageio.mimwrite(
        f"{savePath}_static/{prtx}video.mp4",
        np.stack(rgb_maps_s),
        fps=30,
        quality=8,
        format="ffmpeg",
        output_params=["-f", "mp4"],
    )
    # imageio.mimwrite(f'{savePath}_static/{prtx}depthvideo.mp4', np.stack(depth_maps_s), fps=30, quality=8, format='ffmpeg', output_params=["-f", "mp4"])
    imageio.mimwrite(
        f"{savePath}_dynamic/{prtx}video.mp4",
        np.stack(rgb_maps_d),
        fps=30,
        quality=8,
        format="ffmpeg",
        output_params=["-f", "mp4"],
    )
    # imageio.mimwrite(f'{savePath}_dynamic/{prtx}depthvideo.mp4', np.stack(depth_maps_d), fps=30, quality=8, format='ffmpeg', output_params=["-f", "mp4"])
    # imageio.mimwrite(f'{savePath}_dynamic/{prtx}blending.mp4', np.stack(blending_maps), fps=30, quality=8, format='ffmpeg', output_params=["-f", "mp4"])

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f"{savePath}/{prtx}mean.txt", np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f"{savePath}/{prtx}mean.txt", np.asarray([psnr]))

    return PSNRs, depth_list


def NDC2world(pts, H, W, f):
    # NDC coordinate to world coordinate
    pts_z = 2 / (torch.clamp(pts[..., 2:], min=-1.0, max=1 - 1e-6) - 1)
    pts_x = -pts[..., 0:1] * pts_z * W / 2 / f
    pts_y = -pts[..., 1:2] * pts_z * H / 2 / f
    pts_world = torch.cat([pts_x, pts_y, pts_z], -1)

    return pts_world


def world2NDC(pts_world, H, W, f):
    o0 = -1.0 / (W / (2.0 * f)) * pts_world[..., 0:1] / pts_world[..., 2:]
    o1 = -1.0 / (H / (2.0 * f)) * pts_world[..., 1:2] / pts_world[..., 2:]
    o2 = 1.0 + 2.0 * 1 / pts_world[..., 2:]
    pts = torch.cat([o0, o1, o2], -1)

    return pts


def contract2world(pts_contract):
    pts_norm, _ = torch.max(torch.abs(pts_contract.clone()), dim=-1)
    contract_mask = pts_norm > 1.0
    scale = -1 / (pts_norm - 2)
    pts_world = pts_contract
    pts_world[~contract_mask] = pts_contract[~contract_mask]
    pts_world[contract_mask] = (
        pts_contract[contract_mask]
        / (pts_norm[contract_mask][:, None])
        * scale[contract_mask][:, None]
    )  # TODO: NaN?
    return pts_world


def render_single_3d_point(H, W, f, c2w, pt_NDC):
    """Render 3D position along each ray and project it to the image plane."""
    w2c = c2w[:, :3, :3].transpose(1, 2)

    # NDC coordinate to world coordinate
    pts_map_world = NDC2world(pt_NDC, H, W, f)

    # World coordinate to camera coordinate
    # Translate
    pts_map_world = pts_map_world - c2w[..., 3]
    # Rotate
    pts_map_cam = torch.sum(torch.mul(pts_map_world[..., None, :], w2c[:, :3, :3]), -1)
    # pts_map_cam = torch.sum(pts_map_world[..., None, :] * w2c[:3, :3], -1)

    # Camera coordinate to 2D image coordinate
    pts_plane = torch.cat(
        [
            pts_map_cam[..., 0:1] / (-pts_map_cam[..., 2:]) * f + W * 0.5,
            -pts_map_cam[..., 1:2] / (-pts_map_cam[..., 2:]) * f + H * 0.5,
        ],
        -1,
    )
    # pts_disparity = 1.0 / pts_map_cam[..., 2:]

    pts_map_cam_NDC = world2NDC(pts_map_cam, H, W, f)

    return pts_plane, ((pts_map_cam_NDC[:, 2:] + 1.0) / 2.0)


def render_3d_point(H, W, f, c2w, weights, pts, rays, ray_type="ndc"):
    """Render 3D position along each ray and project it to the image plane."""
    w2c = c2w[:, :3, :3].transpose(1, 2)

    # Rendered 3D position in NDC coordinate
    acc_map = torch.sum(weights, -1)[:, None]
    pts_map_NDC = torch.sum(weights[..., None] * pts, -2)
    if ray_type == "ndc":
        pts_map_NDC = pts_map_NDC + (1.0 - acc_map) * (rays[:, :3] + rays[:, 3:])
    elif ray_type == "contract":
        farest_pts = rays[:, :3] + rays[:, 3:] * 256.0
        # convert to contract domain
        farest_pts_norm, _ = torch.max(torch.abs(farest_pts.clone()), dim=-1)
        contract_mask = farest_pts_norm > 1.0
        farest_pts[contract_mask] = (2 - 1 / farest_pts_norm[contract_mask])[
            ..., None
        ] * (farest_pts[contract_mask] / farest_pts_norm[contract_mask][..., None])
        pts_map_NDC = pts_map_NDC + (1.0 - acc_map) * farest_pts

    # NDC coordinate to world coordinate
    if ray_type == "ndc":
        pts_map_world = NDC2world(pts_map_NDC, H, W, f)
    elif ray_type == "contract":
        pts_map_world = contract2world(pts_map_NDC)

    # World coordinate to camera coordinate
    # Translate
    pts_map_world = pts_map_world - c2w[..., 3]
    # Rotate
    pts_map_cam = torch.sum(torch.mul(pts_map_world[..., None, :], w2c[:, :3, :3]), -1)

    # Camera coordinate to 2D image coordinate
    pts_plane = torch.cat(
        [
            pts_map_cam[..., 0:1] / (-pts_map_cam[..., 2:]) * f + W * 0.5,
            -pts_map_cam[..., 1:2] / (-pts_map_cam[..., 2:]) * f + H * 0.5,
        ],
        -1,
    )

    pts_map_cam_NDC = world2NDC(pts_map_cam, H, W, f)

    return pts_plane, pts_map_cam_NDC[:, 2:]


def induce_flow_single(H, W, focal, pose_neighbor, pts_3d_neighbor, pts_2d):
    # Render 3D position along each ray and project it to the neighbor frame's image plane.
    pts_2d_neighbor, _ = render_single_3d_point(
        H, W, focal, pose_neighbor, pts_3d_neighbor
    )
    induced_flow = pts_2d_neighbor - pts_2d

    return induced_flow


def induce_flow(
    H, W, focal, pose_neighbor, weights, pts_3d_neighbor, pts_2d, rays, ray_type="ndc"
):
    # Render 3D position along each ray and project it to the neighbor frame's image plane.
    pts_2d_neighbor, induced_disp = render_3d_point(
        H, W, focal, pose_neighbor, weights, pts_3d_neighbor, rays, ray_type
    )
    induced_flow = pts_2d_neighbor - pts_2d

    return induced_flow, induced_disp
