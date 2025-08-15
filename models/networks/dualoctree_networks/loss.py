import ocnn
import torch
import torch.nn.functional as F
import os
import scipy
# import models.networks.skelpoint_networks.DistFunc as DF
# from models.networks.skelpoint_networks.GraphAE import LinkPredNet


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


def octree2pts(octree, signal):
    depth = octree.depth
    batch_size = octree.batch_size
    displacement = signal[:, :3]

    x, y, z, _ = octree.xyzb(depth, nempty=True)
    xyz = torch.stack([x, y, z], dim=1) + 0.5 + displacement
    xyz = xyz / 2 ** (depth - 1) - 1.0

    point_cloud = torch.cat([xyz, signal[:, 3:]], dim=1)
    batch_id = octree.batch_id(depth, nempty=True)
    points_num = [torch.sum(batch_id == i) for i in range(batch_size)]
    points = torch.split(point_cloud, points_num)
    return points


def geometry_loss(batch, model_out, reg_loss_type='', codebook_weight = 1.0, kl_weight = 1.0):
    # octree loss
    output = compute_octree_loss(model_out['logits'], model_out['octree_out'])

    # regression loss
    grads = compute_mpu_gradients(model_out['mpus'], batch['pos'])
    reg_loss_func = get_sdf_loss_function(reg_loss_type)
    sdf_loss = compute_sdf_loss(
            model_out['mpus'], grads, batch['sdf'], batch['grad'], reg_loss_func)
    output.update(sdf_loss)

    signal = model_out['signal']
    octree_feature = ocnn.modules.InputFeature('LF', nempty=True)
    signal_gt = octree_feature(model_out['octree_out'])
    signal_gt_points = signal_gt[:, :3]
    signal_gt_features = signal_gt[:, 3:]

    # Calculate position loss and feature loss
    position_loss = F.mse_loss(signal[:, :3], signal_gt_points)
    feature_loss = F.mse_loss(signal[:, 3:], signal_gt_features)

    # Combine both losses
    signal_loss_1 = position_loss #+ cosine_similarity_loss
    output['signal_loss'] = signal_loss_1
    signal_loss_2 = feature_loss #+ gamma * order_aware_loss
    output['feature_loss'] = signal_loss_2

    if 'emb_loss' in model_out.keys():
        output['emb_loss'] = codebook_weight * model_out['emb_loss']    # 只用于graph_vqvae
    if 'kl_loss' in model_out.keys():
        output['kl_loss'] = kl_weight * model_out['kl_loss']    # 只用于graph_vae
    return output
