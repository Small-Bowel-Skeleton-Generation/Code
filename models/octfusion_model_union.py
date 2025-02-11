# Reference: diffusion is borrowed from the LDM repo: https://github.com/CompVis/latent-diffusion
# Specifically, functions from: https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py

import os
import sys
from collections import OrderedDict
from functools import partial
import copy
import scipy.io as sio
import time
import numpy as np
from omegaconf import OmegaConf
from termcolor import colored, cprint
from einops import rearrange, repeat
from tqdm import tqdm
import random
import ocnn
from ocnn.nn import octree2voxel, octree_pad
from ocnn.octree import Octree, Points
from models.networks.dualoctree_networks import dual_octree

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.special import expm1

from models.base_model import BaseModel
from models.networks.diffusion_networks.graph_unet_union import UNet3DModel
from models.model_utils import load_dualoctree
from models.networks.diffusion_networks.ldm_diffusion_util import *

from models.networks.skelpoint_networks.SkelPointNet import SkelPointNet

# distributed
from utils.distributed import reduce_loss_dict, get_rank, get_world_size

# rendering
from utils.util_dualoctree import calc_sdf, octree2split_small, octree2split_large, split2octree_small, split2octree_large
from utils.util import TorchRecoder, seed_everything, category_5_to_label

TRUNCATED_TIME = 0.7


class OctFusionModel(BaseModel):
    def name(self):
        return 'SDFusion-Model-Union-Two-Times'

    def initialize(self, opt):
        self.network_initialize(opt)
        self.optimizer_initialize(opt)
        
    def network_initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.mode == "train"
        self.model_name = self.name()
        self.device = opt.device
        self.gradient_clip_val = 1.
        self.start_iter = opt.start_iter

        if self.isTrain:
            self.log_dir = os.path.join(opt.logs_dir, opt.name)
            self.train_dir = os.path.join(self.log_dir, 'train_temp')
            self.test_dir = os.path.join(self.log_dir, 'test_temp')


        ######## START: Define Networks ########
        assert opt.df_cfg is not None
        assert opt.vq_cfg is not None

        # init df
        df_conf = OmegaConf.load(opt.df_cfg)
        vq_conf = OmegaConf.load(opt.vq_cfg)
        self.batch_size = vq_conf.data.train.batch_size = opt.batch_size
        
        self.vq_conf = vq_conf
        self.solver = self.vq_conf.solver

        self.input_depth = self.vq_conf.model.depth
        self.octree_depth = self.vq_conf.model.depth_stop
        self.small_depth = 6
        self.large_depth = 8
        self.full_depth = self.vq_conf.model.full_depth

        self.load_octree = self.vq_conf.data.train.load_octree
        self.load_pointcloud = self.vq_conf.data.train.load_pointcloud
        self.load_split_small = self.vq_conf.data.train.load_split_small
        self.load_mask = self.vq_conf.data.train.load_mask

        # init diffusion networks
        df_model_params = df_conf.model.params
        unet_params = df_conf.unet.params
        self.conditioning_key = df_model_params.conditioning_key
        self.num_timesteps = df_model_params.timesteps
        self.enable_label = "num_classes" in df_conf.unet.params
        self.df_type = unet_params.df_type

        self.df = UNet3DModel(opt.stage_flag, **unet_params)
        self.df.to(self.device)
        self.stage_flag = opt.stage_flag

        # self.curvesize = 2048
        # self.skel_pointnet = SkelPointNet(num_skel_points=512, input_channels=1)
        # self.skel_pointnet.to(self.device)

        # record z_shape
        self.split_channel = 8
        self.code_channel = self.vq_conf.model.embed_dim
        z_sp_dim = 2 ** self.full_depth
        self.z_shape = (self.split_channel, z_sp_dim, z_sp_dim, z_sp_dim)

        self.ema_df = copy.deepcopy(self.df)
        self.ema_df.to(self.device)
        if opt.isTrain:
            self.ema_rate = opt.ema_rate
            self.ema_updater = EMA(self.ema_rate)
            self.reset_parameters()
            set_requires_grad(self.ema_df, False)

        self.noise_schedule = "linear"
        if self.noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif self.noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {self.noise_schedule}')

        # init vqvae

        self.autoencoder = load_dualoctree(conf = vq_conf, ckpt = opt.vq_ckpt, opt = opt)

        ######## END: Define Networks ########

    def optimizer_initialize(self, opt):

        if self.stage_flag == "lr":
            self.set_requires_grad([
                self.df.unet_hr
            ], False)
        elif self.stage_flag == "hr":
            self.set_requires_grad([
                self.df.unet_lr
            ], False)
        
        if self.isTrain:

            # initialize optimizers
            self.optimizer = optim.AdamW([p for p in self.df.parameters() if p.requires_grad == True], lr=opt.lr)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1000, 0.9)

            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

            self.print_networks(verbose=False)

        if opt.ckpt is None and os.path.exists(os.path.join(opt.logs_dir, opt.name, "ckpt/df_steps-latest.pth")):
            opt.ckpt = os.path.join(opt.logs_dir, opt.name, "ckpt/df_steps-latest.pth")
        
        if opt.ckpt is not None:
            if self.stage_flag == "lr":
                load_options = ["unet_lr"]
            elif self.stage_flag == "hr":
                load_options = ["unet_lr", "unet_hr"]
            if self.isTrain:
                load_options.append("opt")
            self.load_ckpt(opt.ckpt, self.df, self.ema_df, load_options)

        if opt.pretrain_ckpt is not None:
            self.load_ckpt(opt.pretrain_ckpt, self.df, self.ema_df, load_options=["unet_lr"])
                
        trainable_params_num = 0
        for m in [self.df]:
            trainable_params_num += sum([p.numel() for p in m.parameters() if p.requires_grad == True])
        print("Trainable_params: ", trainable_params_num)

        # for distributed training
        if self.opt.distributed:
            self.make_distributed(opt)
            self.df_module = self.df.module
            self.autoencoder_module = self.autoencoder.module

        else:
            self.df_module = self.df
            self.autoencoder_module = self.autoencoder

    def reset_parameters(self):
        self.ema_df.load_state_dict(self.df.state_dict())

    def make_distributed(self, opt):
        self.df = nn.parallel.DistributedDataParallel(
            self.df,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        if opt.sync_bn:
            self.autoencoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.autoencoder)
        self.autoencoder = nn.parallel.DistributedDataParallel(
            self.autoencoder,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

    ############################ START: init diffusion params ############################

    def batch_to_cuda(self, batch):
        def points2octree(points):
            octree = ocnn.octree.Octree(depth = self.input_depth, full_depth = self.full_depth)
            octree.build_octree(points)
            return octree

        if self.load_pointcloud:
            # points = [pts.cuda(non_blocking=True) for pts in batch['points']]
            points = [pts['points'].cuda(non_blocking=True) for pts in batch['curve']]
            octrees = [points2octree(pts) for pts in points]
            octree = ocnn.octree.merge_octrees(octrees)
            octree.construct_all_neigh()
            batch['octree_in'] = octree
            batch['split_small'] = octree2split_small(batch['octree_in'], self.full_depth)

            skeletons = []
            for pts in points:
                point_data = pts.points
                point_tensor = torch.tensor(point_data, dtype=torch.float32).cuda()
                num_points = point_tensor.size(0)
                indices = torch.linspace(0, num_points - 1, 512).round().long()
                point_tensor = point_tensor[indices]
                skeletons.append(point_tensor)
            batch['skeleton_gt'] = torch.stack(skeletons)

        if self.load_mask:
            context = torch.stack([torch.tensor(mask, dtype=torch.float32).unsqueeze(0) for mask in batch['context']])  # B×1×H×W×D
            batch['context'] = context.cuda(non_blocking=True)

        batch['label'] = batch['label'].cuda()
        if self.load_octree:
            batch['octree_in'] = batch['octree_in'].cuda()
            batch['split_small'] = octree2split_small(batch['octree_in'], self.full_depth)
            # batch['split_large'] = self.octree2split_large(batch['octree_in'])
        elif self.load_split_small:
            batch['split_small'] = batch['split_small'].cuda()
            batch['octree_in'] = split2octree_small(batch['split_small'], self.input_depth, self.full_depth)

    def set_input(self, input=None):
        self.batch_to_cuda(input)
        self.split_small = input['split_small']
        # self.split_large = input['split_large']
        self.octree_in = input['octree_in']
        self.batch_size = self.octree_in.batch_size
        self.context = input['context']
        self.skeleton_gt = input['skeleton_gt']

        # # 查看真实输入形式
        # octree_small = split2octree_small(self.split_small, self.octree_depth, self.full_depth)
        # self.export_octree(octree_small, depth=self.small_depth,
        #                    save_dir='/home/data/liangzhichao/Code/octfusion-main/logs/skeleton6_rifle_union/union_2t_test_lr2e-4/results_skeleton6_rifle/latent',
        #                    index=1)

        if self.enable_label:
            self.label = input['label']
        else:
            self.label = None

    def switch_train(self):
        self.df.train()

    def switch_eval(self):
        self.df.eval()

    def calc_skel_loss(self, input_data, doctree_in, batch_id, unet_type, unet_lr, df_type="x0", context=None):
        times = torch.zeros(
            (self.batch_size,), device=self.device).float().uniform_(0, 1)

        noise = torch.randn_like(input_data)

        noise_level = self.log_snr(times)
        alpha, sigma = log_snr_to_alpha_sigma(noise_level)
        batch_alpha = right_pad_dims_to(input_data, alpha[batch_id])
        batch_sigma = right_pad_dims_to(input_data, sigma[batch_id])
        noised_data = batch_alpha * input_data + batch_sigma * noise

        output = self.df(unet_type=unet_type, x=noised_data, doctree=doctree_in, unet_lr=unet_lr, timesteps=noise_level,
                         label=self.label, context=context)

        if df_type == "x0":
            mse_loss = F.mse_loss(output, input_data)
            mae_loss = F.l1_loss(output, input_data)
            combined_loss = 0.7 * mse_loss + 0.3 * mae_loss
            return combined_loss

            # return F.mse_loss(output, input_data)

        elif df_type == "eps":
            # x_start = (noised_data - output * batch_sigma) / batch_alpha.clamp(min=1e-8)
            # self.output = self.autoencoder_module.decode_code(x_start, doctree_in)
            # # 从 octree_out 获取 batch_id
            # batch_id = self.output['octree_out'].batch_id(depth=8, nempty=True)
            # # 初始化一个列表来存储每个曲线的信息
            # curves = []
            #
            # for b in range(self.batch_size):
            #     # 获取属于当前 batch 的 signal
            #     signal_for_batch = self.output['signal'][batch_id == b]  # 选择与当前 batch_id 匹配的点
            #
            #     if signal_for_batch.size(0) > 0:  # 确保有数据
            #         n = signal_for_batch.size(0)
            #         # 使用线性插值将其转换为 512 x 4 的形状
            #         target_indices = torch.linspace(0, n - 1, steps=self.curvesize)
            #         target_indices = target_indices.long()  # 将索引转换为整数
            #         # 将 `signal_for_batch` 的维度调整为适当的形式
            #         interpolated_curve = signal_for_batch[target_indices.clamp(0, n - 1)]  # 确保范围在有效索引内
            #     else:
            #         # 如果没有数据则用零填充
            #         interpolated_curve = torch.zeros((self.curvesize, 4), device=self.device)
            #     curves.append(interpolated_curve)
            #     # 将所有的曲线合并成一个 tensor
            # skel_input = torch.stack(curves)  # shape: (batch_size, 512, 4)
            #
            # skel_pointnet_output = self.skel_pointnet(skel_input)
            # skel_loss = self.skel_pointnet.compute_loss_pre(self.skeleton_gt, skel_pointnet_output)

            # save_dir = '/home/data/liangzhichao/Code/octfusion-main/logs/skeleton4_rifle_union/union_2t_test_lr2e-4/results_skeleton4_rifle/latent'
            # os.makedirs(save_dir, exist_ok=True)  # 创建目录（如果不存在的话）
            # output_dict = {
            #     'skeleton_gt': self.skeleton_gt.detach().cpu().numpy(),
            #     'skel_pointnet_output': skel_pointnet_output.detach().cpu().numpy()
            # }
            # # 生成文件名（可根据需要自定义）
            # mat_file_path = os.path.join(save_dir, 'skeleton_data.mat')
            # sio.savemat(mat_file_path, output_dict)

            return F.mse_loss(output, noise)
        else:
            raise ValueError(f'invalid loss type {df_type}')


    def calc_loss(self, input_data, doctree_in, batch_id, unet_type, unet_lr, df_type="x0", context=None):
        times = torch.zeros(
            (self.batch_size,), device=self.device).float().uniform_(0, 1)
        
        noise = torch.randn_like(input_data)

        noise_level = self.log_snr(times)
        alpha, sigma = log_snr_to_alpha_sigma(noise_level)
        batch_alpha = right_pad_dims_to(input_data, alpha[batch_id])
        batch_sigma = right_pad_dims_to(input_data, sigma[batch_id])
        noised_data = batch_alpha * input_data + batch_sigma * noise

        # for name, param in self.df.unet_lr.named_parameters():
        #     print(f"{name} requires_grad: {param.requires_grad}")

        output = self.df(unet_type=unet_type, x=noised_data, doctree=doctree_in, unet_lr=unet_lr, timesteps=noise_level, label=self.label, context=context)

        # # Save input_data and output to .mat files
        # save_dir = "/home/data/liangzhichao/Code/octfusion-main/logs/skeleton4_union/union_2t_test_lr2e-4/results_skeleton4/latent/"
        # os.makedirs(save_dir, exist_ok=True)
        # input_data_path = os.path.join(save_dir, f"input_data.mat")
        # noised_data_path = os.path.join(save_dir, f"noised_data.mat")
        # output_path = os.path.join(save_dir, f"output.mat")
        # sio.savemat(input_data_path, {'input_data': input_data.detach().cpu().numpy()})
        # sio.savemat(noised_data_path, {'noised_data': noised_data.detach().cpu().numpy()})
        # sio.savemat(output_path, {'output': output.detach().cpu().numpy()})
        
        if df_type == "x0":
            return F.mse_loss(output, input_data)
        elif df_type == "eps":
            # x_start = (noised_data - output * batch_sigma) / batch_alpha.clamp(min=1e-8)
            # self.output = self.autoencoder_module.decode_code(x_start, doctree_in)
            # self.get_sdfs(self.output['neural_mpu'], self.batch_size, bbox = None)
            # self.export_mesh(save_dir = "/home/data/liangzhichao/Code/octfusion-main/logs/rifle_union/union_2t_test_lr2e-4/results_rifle/latent/output/", index = 0)
            #
            # self.output = self.autoencoder_module.decode_code(noised_data, doctree_in)
            # self.get_sdfs(self.output['neural_mpu'], self.batch_size, bbox = None)
            # self.export_mesh(save_dir = "/home/data/liangzhichao/Code/octfusion-main/logs/rifle_union/union_2t_test_lr2e-4/results_rifle/latent/input/", index = 2)
            return F.mse_loss(output, noise)
        else:
            raise ValueError(f'invalid loss type {df_type}')
        
    def forward(self):

        self.df.train()

        c = None        

        self.df_hr_loss = torch.tensor(0., device=self.device)
        self.df_lr_loss = torch.tensor(0., device=self.device)

        if self.stage_flag == "lr":
            # self.df_lr_loss = self.forward_lr(split_small)
            batch_id = torch.arange(0, self.batch_size, device=self.device).long()
            self.df_lr_loss = self.calc_loss(self.split_small, None, batch_id, "lr", None, self.df_type[0], self.context)
            
        elif self.stage_flag == "hr":
            with torch.no_grad():
                self.input_data, self.doctree_in = self.autoencoder_module.extract_code(self.octree_in)
            # self.df_hr_loss = self.forward_hr(self.input_data, self.small_depth, "hr", self.df_module.unet_lr)
            self.df_hr_loss = self.calc_loss(self.input_data, self.doctree_in, self.doctree_in.batch_id(self.small_depth), "hr", self.df_module.unet_lr, self.df_type[1], self.context)

        self.loss = self.df_lr_loss + self.df_hr_loss

    def get_sampling_timesteps(self, batch, device, steps):
        times = torch.linspace(1., 0., steps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    def get_sampling_context(self, mask_dir, batch_size, device):
        if self.isTrain:
            # 从文本文件读取文件名
            with open('/mnt/gemlab_data_2/User_database/liangzhichao/octfusion_dataset_4/filelist/test.txt', 'r') as f:
                file_names = [line.strip() for line in f.readlines()]
            selected_mask_files = [f'{name}.mat' for name in file_names]
            # 过滤出存在的矩阵文件
            existing_mask_files = [f for f in selected_mask_files if os.path.isfile(os.path.join(mask_dir, f))]
            if not existing_mask_files:
                raise FileNotFoundError("No mask files found in the specified directory.")

            selected_mask_files = random.sample(existing_mask_files, batch_size) if len(
                existing_mask_files) >= batch_size else existing_mask_files
        else:
            # mask_dir = "/mnt/gemlab_data_2/User_database/liangzhichao/QC_mask_mat"
            mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.mat')]
            if not mask_files:
                raise FileNotFoundError("No mask files found in the specified directory.")
            selected_mask_files = random.sample(mask_files, batch_size) if len(mask_files) >= batch_size else mask_files

        # selected_mask_files = ['PA1598_09189.mat']
        # mask_dir = '/mnt/gemlab_data_2/User_database/liangzhichao/QC_mask_mat/'
        # selected_mask_files = ['PA1537_CT_Bowel_Small_2_mask.mat']
        masks = []
        for mask_file in selected_mask_files:
            mask_path = os.path.join(mask_dir, mask_file)
            mask_data = sio.loadmat(mask_path)['mask']  # 假设在 .mat 文件中键为 'mask'
            mask_tensor = torch.tensor(mask_data, dtype=torch.float32).unsqueeze(0)  # 添加批量维度
            masks.append(mask_tensor)
        context = torch.stack(masks).to(device)
        return context, mask_file

    # def get_sampling_context(self, mask_dir, batch_size, device):
    #     mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.mat')]
    #     if not mask_files:
    #         raise FileNotFoundError("No mask files found in the specified directory.")
    #
    #     # Select `batch_size` number of random mask files
    #     selected_mask_files = random.sample(mask_files, batch_size) if len(mask_files) >= batch_size else mask_files
    #     # selected_mask_files = ['PA1478_04118.mat']
    #     # Load the masks and create a tensor
    #     masks = []
    #     for mask_file in selected_mask_files:
    #         mask_path = os.path.join(mask_dir, mask_file)
    #         mask_data = sio.loadmat(mask_path)['mask']  # Assumes 'mask' is the key in the .mat file
    #         mask_tensor = torch.tensor(mask_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    #         masks.append(mask_tensor)
    #     # Stack all masks to create a B×1×H×W×D tensor
    #     context = torch.stack(masks).to(device)
    #
    #     return context

    @torch.no_grad()
    def sample_loop(self, doctree_lr = None, ema=False, shape=None, ddim_steps=200, label=None, cond=False, cond_dir=None, unet_type="lr", unet_lr=None, df_type="x0", truncated_index=0.0):
        batch_size = self.vq_conf.data.test.batch_size

        time_pairs = self.get_sampling_timesteps(
            batch_size, device=self.device, steps=ddim_steps)

        if cond:
            context_pairs, mask_filename = self.get_sampling_context(mask_dir=cond_dir, batch_size=batch_size, device=self.device)
        else:
            context_pairs = None

        noised_data = torch.randn(shape, device = self.device)

        x_start = None

        time_iter = tqdm(time_pairs, desc='small sampling loop time step')

        for t, t_next in time_iter:

            log_snr = self.log_snr(t)
            log_snr_next = self.log_snr(t_next)
            noise_cond = log_snr

            # for module in self.ema_df.modules():
            #     if isinstance(module, nn.BatchNorm3d):
            #         print(f"Running mean: {module.running_mean}, Running var: {module.running_var}")

            if ema:
                output = self.ema_df(unet_type=unet_type, x=noised_data, doctree=doctree_lr,  timesteps=noise_cond, unet_lr=unet_lr, x_self_cond=x_start, label=label, context=context_pairs)
            else:
                output = self.df(unet_type=unet_type, x=noised_data, doctree=doctree_lr,  timesteps=noise_cond, unet_lr=unet_lr, x_self_cond=x_start, label=label, context=context_pairs)

            # save_dir = "/home/data/liangzhichao/Code/octfusion-main/logs/skeleton4_union/union_2t_test_lr2e-4/results_skeleton4/latent"
            # os.makedirs(save_dir, exist_ok=True)
            # input_data_path = os.path.join(save_dir, f"sample_output_train.mat")
            # sio.savemat(input_data_path, {'output': output.detach().cpu().numpy()})

            if t[0] < truncated_index and unet_type == "lr":
                output.sign_()

            if df_type == "x0":
                x_start = output
                padded_log_snr, padded_log_snr_next = map(
                    partial(right_pad_dims_to, noised_data), (log_snr, log_snr_next))

                alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
                alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

                c = -expm1(padded_log_snr - padded_log_snr_next)
                mean = alpha_next * (noised_data * (1 - c) / alpha + c * output)
                variance = (sigma_next ** 2) * c
                noise = torch.where(
                    # rearrange(t_next > truncated_index, 'b -> b 1 1 1 1'),
                    right_pad_dims_to(noised_data, t_next > truncated_index),
                    torch.randn_like(noised_data),
                    torch.zeros_like(noised_data)
                )
                noised_data = mean + torch.sqrt(variance) * noise
            elif df_type == "eps":
                alpha, sigma = log_snr_to_alpha_sigma(log_snr)
                alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)
                alpha, sigma, alpha_next, sigma_next = alpha[0], sigma[0], alpha_next[0], sigma_next[0]
                x_start = (noised_data - output * sigma) / alpha.clamp(min=1e-8)
                noised_data = x_start * alpha_next + output * sigma_next
        
        return noised_data, context_pairs, mask_filename
    
    @torch.no_grad()
    def sample(self, split_small = None, category = 'airplane', prefix = 'results', ema = False, ddim_steps=200, clean = False, save_index = 0, cond=False, cond_dir=None):

        if ema:
            self.ema_df.eval()
        else:
            self.df.eval()
        
        batch_size = self.vq_conf.data.test.batch_size
        if self.enable_label:
            label = torch.ones(batch_size).to(self.device) * category_5_to_label[category]
            label = label.long()
        else:
            label = None
            
        save_dir = os.path.join(self.opt.logs_dir, self.opt.name, f"{prefix}_{category}")
        
        if split_small == None:
            # seed_everything(self.opt.seed + save_index)
            split_small, mask, mask_filename = self.sample_loop(doctree_lr=None, ema=ema, shape=(batch_size, *self.z_shape), ddim_steps=ddim_steps, label=label, cond=cond,
                                                 cond_dir=cond_dir, unet_type="lr", unet_lr=None, df_type=self.df_type[0], truncated_index=TRUNCATED_TIME)

        octree_small = split2octree_small(split_small, self.octree_depth, self.full_depth)

        self.export_octree(octree_small, depth = self.small_depth, save_dir = os.path.join(save_dir, "octree"), index = save_index)
        # for i in range(batch_size):
        #     save_path = os.path.join(save_dir, "splits_small", f"{save_index}.pth")
        #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #     torch.save(split_small[i].unsqueeze(0), save_path)
        self.export_mask(mask, save_dir = os.path.join(save_dir, "mask"), index = save_index)

        # latent_save_dir = os.path.join(save_dir, "latent")
        # os.makedirs(latent_save_dir, exist_ok=True)
        # sio.savemat(os.path.join(latent_save_dir, f"{save_index}.mat"), {'split_small': split_small.cpu().numpy()})

        if self.stage_flag == "lr":
            return
        
        doctree_small = dual_octree.DualOctree(octree_small)
        doctree_small.post_processing_for_docnn()

        doctree_small_num = doctree_small.total_num
        
        # seed_everything(self.opt.seed)
        samples, _, _ = self.sample_loop(doctree_lr=doctree_small, shape=(doctree_small_num, self.code_channel), ema=ema, ddim_steps=ddim_steps, label=label,
                                      cond=cond, cond_dir=cond_dir,unet_type="hr", unet_lr=self.ema_df.unet_lr, df_type=self.df_type[1])

        print(samples.max())
        print(samples.min())
        print(samples.mean())
        print(samples.std())

        self.output = self.autoencoder_module.decode_code(samples, doctree_small)

        # batch_id = self.output['octree_out'].batch_id(depth=8, nempty=True)
        # curves = []
        # for b in range(self.batch_size):
        #     # 获取属于当前 batch 的 signal
        #     signal_for_batch = self.output['signal'][batch_id == b]  # 选择与当前 batch_id 匹配的点
        #     if signal_for_batch.size(0) > 0:  # 确保有数据
        #         n = signal_for_batch.size(0)
        #         # 使用线性插值将其转换为 512 x 4 的形状
        #         target_indices = torch.linspace(0, n - 1, steps=self.curvesize)
        #         target_indices = target_indices.long()  # 将索引转换为整数
        #         # 将 `signal_for_batch` 的维度调整为适当的形式
        #         interpolated_curve = signal_for_batch[target_indices.clamp(0, n - 1)]  # 确保范围在有效索引内
        #     else:
        #         # 如果没有数据则用零填充
        #         interpolated_curve = torch.zeros((self.curvesize, 4), device=self.device)
        #     curves.append(interpolated_curve)
        # skel_input = torch.stack(curves)  # shape: (batch_size, 512, 4)
        # skeleton_pre = self.skel_pointnet(skel_input)
        # skeleton_mat_path = os.path.join(save_dir, f'skeleton_{mask_filename[:-4]}.mat')
        # sio.savemat(skeleton_mat_path, {'skeleton': skeleton_pre.cpu().detach().numpy()})

        # # decode z
        # # self.output = self.autoencoder_module.decode_code(samples, doctree_small)
        self.get_sdfs(self.output['neural_mpu'], batch_size, bbox = None)
        self.export_mesh(save_dir = save_dir, index = save_index, clean = clean)

        signal = self.output['signal'].cpu().numpy()  # Ensure it's on CPU and convert to numpy array
        signal_mat_path = os.path.join(save_dir, 'skeleton', f'signal_{mask_filename[:-4]}_{save_index}.mat')
        sio.savemat(signal_mat_path, {'signal': signal})

        # skeleton = self.output['ounet_out']['signal'].cpu().numpy()  # Ensure it's on CPU and convert to numpy array
        # skeleton_mat_path = os.path.join(save_dir, f'skeleton_{mask_filename[:-4]}.mat')
        # sio.savemat(skeleton_mat_path, {'skeleton': skeleton})

    def export_octree(self, octree, depth, save_dir = None, index = 0):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except FileExistsError:
            pass

        batch_id = octree.batch_id(depth = depth, nempty = False)
        data = torch.ones((len(batch_id), 1), device = self.device)
        data = octree2voxel(data = data, octree = octree, depth = depth, nempty = False)
        data = data.permute(0,4,1,2,3).contiguous()

        batch_size = octree.batch_size

        for i in range(batch_size):
            voxel = data[i].squeeze().cpu().numpy()
            mat_path = os.path.join(save_dir, f'{index + i}.mat')
            sio.savemat(mat_path, {'data': voxel})

        # for i in tqdm(range(batch_size)):
        #     voxel = data[i].squeeze().cpu().numpy()
        #     mesh = voxel2mesh(voxel)
        #     if batch_size == 1:
        #         mesh.export(os.path.join(save_dir, f'{index}.obj'))
        #     else:
        #         mesh.export(os.path.join(save_dir, f'{index + i}.obj'))


    def export_mask(self, context = 0, save_dir='', index=0):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except FileExistsError:
            pass

        for i, mask_tensor in enumerate(context):
            mask_save_path = os.path.join(save_dir, f"{index}_{i}.mat")
            # Convert mask_tensor back to numpy and save as .mat file
            mask_data_np = mask_tensor.squeeze(0).cpu().numpy()
            sio.savemat(mask_save_path, {'mask': mask_data_np})


    def get_sdfs(self, neural_mpu, batch_size, bbox):
        # bbox used for marching cubes
        if bbox is not None:
            self.bbmin, self.bbmax = bbox[:3], bbox[3:]
        else:
            sdf_scale = self.solver.sdf_scale
            self.bbmin, self.bbmax = -sdf_scale, sdf_scale    # sdf_scale = 0.9

        self.sdfs = calc_sdf(neural_mpu, batch_size, size = self.solver.resolution, bbmin = self.bbmin, bbmax = self.bbmax)

    # def export_mesh(self, save_dir, index = 0, level = 0, clean = False):
    #     try:
    #         os.makedirs(save_dir, exist_ok=True)
    #     except FileExistsError:
    #         pass
    #     ngen = self.sdfs.shape[0]
    #     size = self.solver.resolution
    #     mesh_scale=self.vq_conf.data.test.point_scale
    #     for i in range(ngen):
    #         filename = os.path.join(save_dir, f'{index + i}.obj')
    #         if ngen == 1:
    #             filename = os.path.join(save_dir, f'{index}.obj')
    #         sdf_value = self.sdfs[i].cpu().numpy()
    #         vtx, faces = np.zeros((0, 3)), np.zeros((0, 3))
    #         try:
    #             vtx, faces, _, _ = skimage.measure.marching_cubes(sdf_value, level)
    #         except:
    #             pass
    #         if vtx.size == 0 or faces.size == 0:
    #             print('Warning from marching cubes: Empty mesh!')
    #             return
    #         vtx = vtx * ((self.bbmax - self.bbmin) / size) + self.bbmin   # [0,sz]->[bbmin,bbmax]  把vertex放缩到[bbmin, bbmax]之间
    #         vtx = vtx * mesh_scale
    #         mesh = trimesh.Trimesh(vtx, faces)  # 利用Trimesh创建mesh并存储为obj文件。
    #         if clean:
    #             components = mesh.split(only_watertight=False)
    #             bbox = []
    #             for c in components:
    #                 bbmin = c.vertices.min(0)
    #                 bbmax = c.vertices.max(0)
    #                 bbox.append((bbmax - bbmin).max())
    #             max_component = np.argmax(bbox)
    #             mesh = components[max_component]
    #         mesh.export(filename)

    def export_mesh(self, save_dir, index=0, level=0, clean=False):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except FileExistsError:
            pass

        ngen = self.sdfs.shape[0]
        size = self.solver.resolution
        mesh_scale = self.vq_conf.data.test.point_scale

        for i in range(ngen):
            # 定义文件名
            filename_obj = os.path.join(os.path.join(save_dir, "sdf"), f'{index + i}.obj')
            filename_mat = os.path.join(os.path.join(save_dir, "sdf"), f'{index + i}.mat')

            if ngen == 1:
                filename_obj = os.path.join(os.path.join(save_dir, "sdf"), f'{index}.obj')
                filename_mat = os.path.join(os.path.join(save_dir, "sdf"), f'{index}.mat')

            # 获取SDF值并转换为numpy数组
            sdf_value = self.sdfs[i].cpu().numpy()

            # # 使用 marching cubes 获取顶点和面
            # vtx, faces = np.zeros((0, 3)), np.zeros((0, 3))
            # try:
            #     vtx, faces, _, _ = skimage.measure.marching_cubes(sdf_value, level)
            # except:
            #     pass
            #
            # if vtx.size == 0 or faces.size == 0:
            #     print('Warning from marching cubes: Empty mesh!')
            #     return
            #
            # # 对顶点进行放缩
            # vtx = vtx * ((self.bbmax - self.bbmin) / size) + self.bbmin
            # vtx = vtx * mesh_scale
            #
            # # 创建网格并保存为.obj文件
            # mesh = trimesh.Trimesh(vtx, faces)
            # if clean:
            #     components = mesh.split(only_watertight=False)
            #     bbox = []
            #     for c in components:
            #         bbmin = c.vertices.min(0)
            #         bbmax = c.vertices.max(0)
            #         bbox.append((bbmax - bbmin).max())
            #     max_component = np.argmax(bbox)
            #     mesh = components[max_component]
            #
            # # 导出.obj文件
            # mesh.export(filename_obj)

            # 保存sdf_value到.mat文件
            sio.savemat(filename_mat, {'sdfs': sdf_value})


    def backward(self):

        self.loss.backward()

    def update_EMA(self):
        update_moving_average(self.ema_df, self.df, self.ema_updater)
        # 确保 BatchNorm3d 的参数一致
        for ema_module, df_module in zip(self.ema_df.modules(), self.df.modules()):
            if isinstance(ema_module, nn.BatchNorm3d) and isinstance(df_module, nn.BatchNorm3d):
                ema_module.running_mean.copy_(df_module.running_mean)
                ema_module.running_var.copy_(df_module.running_var)
                ema_module.num_batches_tracked.copy_(df_module.num_batches_tracked)

    def optimize_parameters(self):

        # self.set_requires_grad([self.df.unet_hr], requires_grad=True)

        self.forward()
        assert not torch.isnan(self.loss).any()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
        self.update_EMA()

    def get_current_errors(self):

        ret = OrderedDict([
            ('loss', self.loss.data),
            ('lr', self.optimizer.param_groups[0]['lr']),
        ])

        if hasattr(self, 'loss_gamma'):
            ret['gamma'] = self.loss_gamma.data

        return ret

    def save(self, label, global_iter):

        state_dict = {
            'df_unet_lr': self.df_module.unet_lr.state_dict(),
            'ema_df_unet_lr': self.ema_df.unet_lr.state_dict(),
            'opt': self.optimizer.state_dict(),
            'global_step': global_iter,
        }
        if self.stage_flag == "hr":
            state_dict['df_unet_hr'] = self.df_module.unet_hr.state_dict()
            state_dict['ema_df_unet_hr'] = self.ema_df.unet_hr.state_dict()

        save_filename = 'df_%s.pth' % (label)
        save_path = os.path.join(self.opt.ckpt_dir, save_filename)

        ckpts = os.listdir(self.opt.ckpt_dir)
        ckpts = [ck for ck in ckpts if ck!='df_steps-latest.pth']
        ckpts.sort(key=lambda x: int(x[9:-4]))
        if len(ckpts) > self.opt.ckpt_num:
            for ckpt in ckpts[:-self.opt.ckpt_num]:
                os.remove(os.path.join(self.opt.ckpt_dir, ckpt))

        torch.save(state_dict, save_path)

    def load_ckpt(self, ckpt, df, ema_df, load_options=[]):
        map_fn = lambda storage, loc: storage
        if type(ckpt) == str:
            state_dict = torch.load(ckpt, map_location=map_fn)
        else:
            state_dict = ckpt

        if "unet_lr" in load_options and "df_unet_lr" in state_dict:
            df.unet_lr.load_state_dict(state_dict['df_unet_lr'])
            ema_df.unet_lr.load_state_dict(state_dict['ema_df_unet_lr'])
            print(colored('[*] weight successfully load unet_lr from: %s' % ckpt, 'blue'))
        if "unet_hr" in load_options and "df_unet_hr" in state_dict:
            df.unet_hr.load_state_dict(state_dict['df_unet_hr'])
            ema_df.unet_hr.load_state_dict(state_dict['ema_df_unet_hr'])
            print(colored('[*] weight successfully load unet_hr from: %s' % ckpt, 'blue'))

        if "opt" in load_options and "opt" in state_dict:
            self.start_iter = state_dict['global_step']
            print(colored('[*] training start from: %d' % self.start_iter, 'green'))
            self.optimizer.load_state_dict(state_dict['opt'])
            print(colored('[*] optimizer successfully restored from: %s' % ckpt, 'blue'))
