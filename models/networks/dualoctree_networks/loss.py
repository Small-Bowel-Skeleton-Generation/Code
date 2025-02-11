# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn
import torch
import torch.nn.functional as F
import os
import scipy
import models.networks.skelpoint_networks.DistFunc as DF
from models.networks.skelpoint_networks.GraphAE import LinkPredNet


def compute_gradient(y, x):
    if x.dtype is not torch.float32:
        x = x.to(torch.float32)
    if y.dtype is not torch.float32:
        y = y.to(torch.float32)
    grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs, create_graph=True)[0]
    return grad


def sdf_reg_loss(sdf, grad, sdf_gt, grad_gt, name_suffix=''):
    wg, ws = 1.0, 200.0
    grad_loss = (grad - grad_gt).pow(2).mean() * wg
    sdf_loss = (sdf - sdf_gt).pow(2).mean() * ws
    loss_dict = {'grad_loss' + name_suffix: grad_loss,
                'sdf_loss' + name_suffix: sdf_loss}
    return loss_dict


def sdf_grad_loss(sdf, grad, sdf_gt, grad_gt, name_suffix=''):
    on_surf = sdf_gt != -1
    off_surf = on_surf.logical_not()

    sdf_loss = sdf[on_surf].pow(2).mean() * 200.0
    norm_loss = (grad[on_surf] - grad_gt[on_surf]).pow(2).mean() * 1.0
    intr_loss = torch.exp(-40 * torch.abs(sdf[off_surf])).mean() * 0.1
    grad_loss = (grad[off_surf].norm(2, dim=-1) - 1).abs().mean() * 0.1

    losses = [sdf_loss, intr_loss, norm_loss, grad_loss]
    names = ['sdf_loss', 'inter_loss', 'norm_loss', 'grad_loss']
    names = [name + name_suffix for name in names]
    loss_dict = dict(zip(names, losses))
    return loss_dict


def sdf_grad_regularized_loss(sdf, grad, sdf_gt, grad_gt, name_suffix=''):
    on_surf = sdf_gt != -1
    off_surf = on_surf.logical_not()

    sdf_loss = sdf[on_surf].pow(2).mean() * 200.0
    norm_loss = (grad[on_surf] - grad_gt[on_surf]).pow(2).mean() * 1.0
    intr_loss = torch.exp(-40 * torch.abs(sdf[off_surf])).mean() * 0.1
    grad_loss = (grad[off_surf].norm(2, dim=-1) - 1).abs().mean() * 0.1
    grad_reg_loss = (grad[off_surf] - grad_gt[off_surf]).pow(2).mean() * 0.1

    losses = [sdf_loss, intr_loss, norm_loss, grad_loss, grad_reg_loss]
    names = ['sdf_loss', 'inter_loss', 'norm_loss', 'grad_loss', 'grad_reg_loss']
    names = [name + name_suffix for name in names]
    loss_dict = dict(zip(names, losses))
    return loss_dict


def possion_grad_loss(sdf, grad, sdf_gt, grad_gt, name_suffix=''):
    on_surf = sdf_gt == 0
    out_of_bbox = sdf_gt == 1.0
    off_surf = on_surf.logical_not()

    sdf_loss = sdf[on_surf].pow(2).mean() * 200.0
    norm_loss = (grad[on_surf] - grad_gt[on_surf]).pow(2).mean() * 1.0
    intr_loss = torch.exp(-40 * torch.abs(sdf[off_surf])).mean() * 0.1
    grad_loss = grad[off_surf].pow(2).mean() * 0.1    # poisson loss
    bbox_loss = torch.mean(torch.relu(-sdf[out_of_bbox])) * 100.0

    losses = [sdf_loss, intr_loss, norm_loss, grad_loss, bbox_loss]
    names = ['sdf_loss', 'inter_loss', 'norm_loss', 'grad_loss', 'bbox_loss']
    names = [name + name_suffix for name in names]
    loss_dict = dict(zip(names, losses))
    return loss_dict

def color_loss(color, gt_color, name_suffix=''):
    loss_dict = dict()
    # TODO: color loss type
    color_loss = (color - gt_color).pow(2).mean() * 200.0
    loss_dict['color_loss' + name_suffix] = color_loss

    return loss_dict

def compute_mpu_gradients(mpus, pos, fval_transform=None):
    grads = dict()
    for d in mpus.keys():
        fval, flags = mpus[d]
        if fval_transform is not None:
            fval = fval_transform(fval)
        grads[d] = compute_gradient(fval, pos)[:, :3]
    return grads


def compute_octree_loss(logits, octree_out):
    weights = [1.0] * 16
    # weights = [1.0] * 4 + [0.8, 0.6, 0.4] + [0.2] * 16

    output = dict()
    for d in logits.keys():
        logitd = logits[d]
        label_gt = octree_out.nempty_mask(d).long()
        # label_gt = ocnn.octree_property(octree_out, 'split', d).long()
        output['loss_%d' % d] = F.cross_entropy(logitd, label_gt) * weights[d]
        output['accu_%d' % d] = logitd.argmax(1).eq(label_gt).float().mean()
    return output


def compute_sdf_loss(mpus, grads, sdf_gt, grad_gt, reg_loss_func):
    output = dict()
    for d in mpus.keys():
        sdf, flgs = mpus[d]    # TODO: tune the loss weights and `flgs`
        reg_loss = reg_loss_func(sdf, grads[d], sdf_gt, grad_gt, '_%d' % d)
        # if d < 3:    # ignore depth 2
        #     for key in reg_loss.keys():
        #         reg_loss[key] = reg_loss[key] * 0.0
        output.update(reg_loss)
    return output

def compute_color_loss(colors, color_gt):
    output = dict()
    for d in colors.keys():
        color, flgs = colors[d]    # TODO: tune the loss weights and `flgs`
        reg_loss = color_loss(color, color_gt,'_%d' % d)
        output.update(reg_loss)
    return output

def compute_occu_loss_v0(mpus, grads, occu_gt, grad_gt, weight):
    output = dict()
    for d in mpus.keys():
        occu, flgs, grad = mpus[d]

        # pos_weight = torch.ones_like(occu_gt) * 10.0
        loss_o = F.binary_cross_entropy_with_logits(occu, occu_gt, weight=weight)
        # loss_g = torch.mean((grad - grad_gt) ** 2)

        occu = torch.sigmoid(occu)
        non_surface_points = occu_gt != 0.5
        accu = (occu > 0.5).eq(occu_gt).float()[non_surface_points].mean()

        output['occu_loss_%d' % d] = loss_o
        # output['grad_loss_%d' % d] = loss_g
        output['occu_accu_%d' % d] = accu
    return output

def get_sdf_loss_function(loss_type=''):
    if loss_type == 'sdf_reg_loss':
        return sdf_reg_loss
    elif loss_type == 'sdf_grad_loss':
        return sdf_grad_loss
    elif loss_type == 'possion_grad_loss':
        return possion_grad_loss
    elif loss_type == 'sdf_grad_reg_loss':
        return sdf_grad_regularized_loss
    else:
        return None


def geometry_loss(batch, model_out, reg_loss_type='', codebook_weight = 1.0, kl_weight = 1.0):
    # octree loss
    output = compute_octree_loss(model_out['logits'], model_out['octree_out'])

    # regression loss
    grads = compute_mpu_gradients(model_out['mpus'], batch['pos'])
    reg_loss_func = get_sdf_loss_function(reg_loss_type)
    sdf_loss = compute_sdf_loss(
            model_out['mpus'], grads, batch['sdf'], batch['grad'], reg_loss_func)
    output.update(sdf_loss)

    # signal loss
    signal = model_out['signal']
    scale = 2 ** (model_out['octree_out'].depth - 1)
    depth = model_out['octree_out'].depth
    signal_gt_points = (batch['octree_in'].points[depth] / scale) - 1.0
    signal_gt_features = batch['octree_in'].features[depth]

    # save_path = "/home/data/liangzhichao/Code/octfusion-main/logs/skeleton6_union/test_snet_lr1e-3/test_temp/"
    # os.makedirs(save_path, exist_ok=True)
    # scipy.io.savemat(os.path.join(save_path, "ounet_gt.mat"),
    #                  {'ounet_gt': ounet_gt.cpu().numpy()})
    # scipy.io.savemat(os.path.join(save_path, "signal_gt_points.mat"),
    #                  {'signal_gt_points': signal_gt_points.cpu().numpy()})
    # scipy.io.savemat(os.path.join(save_path, "signal_gt_features.mat"), {'signal_gt_features': signal_gt_features.cpu().numpy()})

    # Calculate position loss and feature loss
    position_loss = F.mse_loss(signal[:, :3], signal_gt_points)
    feature_loss = F.mse_loss(signal[:, 3:], signal_gt_features)

    direction_signal = F.normalize(signal[:, :3], dim=-1)
    direction_gt = F.normalize(signal_gt_points, dim=-1)
    cosine_similarity_loss = 1 - torch.mean(torch.sum(direction_signal * direction_gt, dim=-1))

    # Order-aware loss
    tau = 20.0
    gamma = 100
    sorted_indices = torch.argsort(signal_gt_features.squeeze(), dim=0)  # Sort by ground truth order
    ranks = torch.zeros_like(sorted_indices)
    ranks[sorted_indices] = torch.arange(len(sorted_indices), device=sorted_indices.device)  # Assign ranks based on sorted order
    rank_diff = ranks.unsqueeze(1) - ranks.unsqueeze(0)  # Rank differences
    rank_diff_abs = rank_diff.abs()  # Absolute rank difference
    weights = torch.exp(-rank_diff_abs / tau)  # Weight decay based on rank difference
    predicted_order = signal[:, 3]  # Predicted order values
    order_diff = predicted_order.unsqueeze(1) - predicted_order.unsqueeze(0)  # Order differences
    order_aware_loss = torch.mean(weights * torch.abs(order_diff))  # Weighted order-aware loss

    # Combine both losses
    signal_loss_1 = position_loss + cosine_similarity_loss
    output['signal_loss'] = signal_loss_1
    signal_loss_2 = feature_loss + gamma * order_aware_loss
    output['feature_loss'] = signal_loss_2

    # ounet_out = model_out['ounet_out']
    # ounet_logits = ounet_out['logits']
    # weights = [1.0] * 16
    # for d in ounet_logits.keys():
    #     ounet_logitd = ounet_logits[d]
    #     label_gt = ounet_out['octree_out'].nempty_mask(d).long()
    #     output['ounet_logit_loss_%d' % d] = F.cross_entropy(ounet_logitd, label_gt) * weights[d]
    #     output['ounet_accu_%d' % d] = ounet_logitd.argmax(1).eq(label_gt).float().mean()
    #
    # # octree regression loss
    # ounet_output = ounet_out['signal']
    # ounet_gt = model_out['octree_out']  # 假设你有一个ground truth 模型输出的标签
    # octree_feature = ocnn.modules.InputFeature('PF', nempty=True)
    # ounet_gt = octree_feature(ounet_gt)
    # # output['ounet_position_loss'] = torch.mean(torch.sum((ounet_gt - ounet_output) ** 2, dim=1))
    # output['ounet_position_loss'] = F.mse_loss(ounet_gt[:, :3], ounet_output[:, :3])
    #
    # ounet_signal = F.normalize(ounet_output[:, :3], dim=-1)
    # ounet_signal_gt = F.normalize(ounet_gt[:, :3], dim=-1)
    # output['ounet_cs_loss'] = 1 - torch.mean(torch.sum(ounet_signal * ounet_signal_gt, dim=-1))
    #
    # output['ounet_feature_loss'] = F.mse_loss(ounet_gt[:, 3:], ounet_output[:, 3:])

    # # 创建保存路径
    # save_dir = '/home/data/liangzhichao/Code/octfusion-main/logs/skeleton4_union/test_snet_lr1e-3/latent'
    # os.makedirs(save_dir, exist_ok=True)
    # # 将数据转换为 NumPy 数组
    # ounet_gt_np = ounet_gt.detach().cpu().numpy()
    # ounet_output_np = ounet_output.detach().cpu().numpy()
    # save_data = {
    #     'ounet_gt': ounet_gt_np,
    #     'ounet_output': ounet_output_np
    # }
    # file_path = os.path.join(save_dir, 'ounet_data.mat')
    # scipy.io.savemat(file_path, save_data)

    # skel_out = model_out['skel_out']
    # cd1 = DF.closest_distance_with_batch(batch['skeleton_gt'], skel_out['skel_xyz'])
    # cd2 = DF.closest_distance_with_batch(skel_out['skel_xyz'], batch['skeleton_gt'])
    # loss_cd = cd1 + cd2
    # output['skel_loss'] = loss_cd * 0.0001
    # known_nodes = skel_out['known_mask'].detach().sum()
    # pos_weight = float(known_nodes - skel_out['A_init'].sum()) / skel_out['A_init'].sum()
    # output['gae_loss'] = F.binary_cross_entropy_with_logits(skel_out['A_pred'], skel_out['A_init'], pos_weight=pos_weight, weight=skel_out['known_mask'].detach())


    if 'emb_loss' in model_out.keys():
        output['emb_loss'] = codebook_weight * model_out['emb_loss']    # 只用于graph_vqvae
    if 'kl_loss' in model_out.keys():
        output['kl_loss'] = kl_weight * model_out['kl_loss']    # 只用于graph_vae
    return output

def geometry_color_loss(batch, model_out, reg_loss_type='', codebook_weight = 1.0, kl_weight = 1.0):
    # octree loss
    output = compute_octree_loss(model_out['logits'], model_out['octree_out'])

    # regression loss
    grads = compute_mpu_gradients(model_out['mpus'], batch['pos'])
    reg_loss_func = get_sdf_loss_function(reg_loss_type)
    sdf_loss = compute_sdf_loss(
            model_out['mpus'], grads, batch['sdf'], batch['grad'], reg_loss_func)
    output.update(sdf_loss)
    color_loss = compute_color_loss(model_out['colors'], batch['color'])
    output.update(color_loss)
    if 'emb_loss' in model_out.keys():
        output['emb_loss'] = codebook_weight * model_out['emb_loss']    # 只用于graph_vqvae
    if 'kl_loss' in model_out.keys():
        output['kl_loss'] = kl_weight * model_out['kl_loss']    # 只用于graph_vae
    return output


def dfaust_loss(batch, model_out, reg_loss_type=''):
    # there is no octree loss
    grads = compute_mpu_gradients(model_out['mpus'], batch['pos'])
    reg_loss_func = get_sdf_loss_function(reg_loss_type)
    output = compute_sdf_loss(
            model_out['mpus'], grads, batch['sdf'], batch['grad'], reg_loss_func)
    return output
