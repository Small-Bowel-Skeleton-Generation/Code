# Reference: diffusion is borrowed from the LDM repo: https://github.com/CompVis/latent-diffusion
# Specifically, functions from: https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py

import os
import copy
import scipy.io as sio
import numpy as np
from omegaconf import OmegaConf
from termcolor import colored, cprint
from einops import repeat
import scipy.ndimage as ndimage
from tqdm import tqdm
import random
import ocnn
from ocnn.octree import Octree, Points

import torch
import torch.nn.functional as F
from torch import nn, optim

from models.base_model import BaseModel
from models.networks.diffusion_networks.graph_unet_union import UNet3DModel
from models.model_utils import load_dualoctree, set_requires_grad
from models.networks.diffusion_networks.ldm_diffusion_util import *
from models.networks.dualoctree_networks import dual_octree

from utils.distributed import get_rank
from utils.util_dualoctree import calc_sdf, octree2split_small, split2octree_small
from utils.util import seed_everything

# Import skeleton refiner
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))
from skeleton_refine import refine_signal_to_skeleton

class TreeDiffuisonModel(BaseModel):
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.is_train = opt.mode == "train"
        self.device = opt.device
        self.start_iter = opt.start_iter

        self._initialize_configs(opt)
        self._initialize_networks(opt)
        self._initialize_optimizers(opt)
        self._load_checkpoints(opt)
        self._finalize_setup(opt)

    def _initialize_configs(self, opt):
        assert opt.df_cfg is not None, "Diffusion model config must be provided."
        assert opt.vq_cfg is not None, "VAE model config must be provided."

        df_conf = OmegaConf.load(opt.df_cfg)
        vq_conf = OmegaConf.load(opt.vq_cfg)
        
        self.batch_size = opt.batch_size
        vq_conf.data.train.batch_size = self.batch_size

        self.vq_config = vq_conf
        self.df_config = df_conf
        self.solver_config = vq_conf.solver

        self.input_depth = vq_conf.model.depth
        self.octree_depth = vq_conf.model.depth_stop
        self.small_depth = 6
        self.large_depth = 8
        self.full_depth = vq_conf.model.full_depth

        self.load_octree = vq_conf.data.train.load_octree
        self.load_pointcloud = vq_conf.data.train.load_pointcloud
        self.load_split_small = vq_conf.data.train.load_split_small
        self.load_mask = vq_conf.data.train.load_mask
        self.load_graph = vq_conf.data.train.load_graph

        df_model_params = df_conf.model.params
        self.conditioning_key = df_model_params.conditioning_key
        self.num_timesteps = df_model_params.timesteps
        self.enable_label = "num_classes" in df_conf.unet.params
        self.df_type = df_conf.unet.params.df_type
        self.stage_flag = opt.stage_flag

        self.noise_schedule = "linear"
        if self.noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif self.noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'Invalid noise schedule {self.noise_schedule}')

    def _initialize_networks(self, opt):
        unet_params = self.df_config.unet.params
        self.df = UNet3DModel(self.stage_flag, **unet_params).to(self.device)

        # record z_shape
        self.split_channel = 8
        self.code_channel = self.vq_config.model.embed_dim
        z_sp_dim = 2 ** self.full_depth
        self.z_shape = (self.split_channel, z_sp_dim, z_sp_dim, z_sp_dim)

        self.ema_df = copy.deepcopy(self.df)
        self.ema_df.to(self.device)
        if self.is_train:
            self.ema_rate = opt.ema_rate
            self.ema_updater = EMA(self.ema_rate)
            self.reset_ema()
            set_requires_grad(self.ema_df, False)

        self.autoencoder = load_dualoctree(conf=self.vq_config, ckpt=opt.vq_ckpt, opt=opt)
        set_requires_grad(self.autoencoder, False)
        self.autoencoder.eval()

    def _initialize_optimizers(self, opt):
        if self.stage_flag == "lr":
            set_requires_grad(self.df.unet_hr, False)
        elif self.stage_flag == "hr":
            set_requires_grad(self.df.unet_lr, False)

        if self.is_train:
            self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.df.parameters()), lr=opt.lr)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1000, 0.9)
            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]
            self.print_networks(verbose=False)

    def _load_checkpoints(self, opt):
        if opt.ckpt is None and os.path.exists(os.path.join(opt.logs_dir, opt.name, "ckpt/df_steps-latest.pth")):
            opt.ckpt = os.path.join(opt.logs_dir, opt.name, "ckpt/df_steps-latest.pth")

        if opt.ckpt is not None:
            load_options = []
            if self.stage_flag == "lr":
                load_options.append("unet_lr")
            elif self.stage_flag == "hr":
                load_options.extend(["unet_lr", "unet_hr"])
            
            if self.is_train:
                load_options.append("opt")
            self.load_ckpt(opt.ckpt, self.df, self.ema_df, load_options)

        if opt.pretrain_ckpt is not None:
            self.load_ckpt(opt.pretrain_ckpt, self.df, self.ema_df, load_options=["unet_lr"])

    def _finalize_setup(self, opt):
        trainable_params_num = sum(p.numel() for p in self.df.parameters() if p.requires_grad)
        cprint(f"Trainable parameters: {trainable_params_num}", 'cyan')

        if self.opt.distributed:
            self._setup_distributed(opt)
            self.df_module = self.df.module
            self.autoencoder_module = self.autoencoder.module
        else:
            self.df_module = self.df
            self.autoencoder_module = self.autoencoder

    def reset_ema(self):
        self.ema_df.load_state_dict(self.df.state_dict())

    def _setup_distributed(self, opt):
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

    def batch_to_cuda(self, batch):
        def points2octree(points):
            octree = ocnn.octree.Octree(depth=self.input_depth, full_depth=self.full_depth)
            octree.build_octree(points)
            return octree

        if self.load_pointcloud:
            points = [pts['points'].to(self.device, non_blocking=True) for pts in batch['curve']]
            octrees = [points2octree(pts) for pts in points]
            octree = ocnn.octree.merge_octrees(octrees)
            octree.construct_all_neigh()
            batch['octree_in'] = octree
            batch['split_small'] = octree2split_small(batch['octree_in'], self.full_depth)

        if self.load_mask:
            context = torch.stack(
                [torch.tensor(mask, dtype=torch.float32).unsqueeze(0) for mask in batch['context']])
            batch['context'] = context.to(self.device, non_blocking=True)

        if self.load_graph:
            points = [gra.to(self.device, non_blocking=True) for gra in batch['graph_inf']]
            octrees = [points2octree(pts) for pts in points]
            octree = ocnn.octree.merge_octrees(octrees)
            octree.construct_all_neigh()
            batch['octree_graph'] = octree

        if 'label' in batch:
            batch['label'] = batch['label'].to(self.device)
        if self.load_octree:
            batch['octree_in'] = batch['octree_in'].to(self.device)
            batch['split_small'] = octree2split_small(batch['octree_in'], self.full_depth)
        elif self.load_split_small:
            batch['split_small'] = batch['split_small'].to(self.device)
            batch['octree_in'] = split2octree_small(batch['split_small'], self.input_depth, self.full_depth)

    def set_input(self, data):
        self.batch_to_cuda(data)
        self.split_small = data['split_small']
        self.octree_in = data['octree_in']
        self.batch_size = self.octree_in.batch_size
        self.context = data.get('context')
        self.octree_graph = data.get('octree_graph')

        if self.enable_label:
            self.label = data.get('label')
        else:
            self.label = None

    def switch_to_train(self):
        self.df.train()

    def switch_to_eval(self):
        self.df.eval()

    def _calculate_loss(self, input_data, doctree_in, batch_id, unet_type, unet_lr, df_type="x0", context=None, octree_graph=None):
        times = torch.zeros((self.batch_size,), device=self.device).float().uniform_(0, 1)
        noise = torch.randn_like(input_data)

        noise_level = self.log_snr(times)
        alpha, sigma = log_snr_to_alpha_sigma(noise_level)
        
        batch_alpha = right_pad_dims_to(input_data, alpha[batch_id])
        batch_sigma = right_pad_dims_to(input_data, sigma[batch_id])
        noised_data = batch_alpha * input_data + batch_sigma * noise

        output = self.df(unet_type=unet_type, x=noised_data, doctree=doctree_in, unet_lr=unet_lr, timesteps=noise_level,
                         label=self.label, context=context, graph_octree=octree_graph)

        if df_type == "x0":
            return F.mse_loss(output, input_data)
        elif df_type == "eps":
            return F.mse_loss(output, noise)
        else:
            raise ValueError(f'Invalid loss type {df_type}')

    def forward(self):
        self.switch_to_train()

        self.df_hr_loss = torch.tensor(0., device=self.device)
        self.df_lr_loss = torch.tensor(0., device=self.device)

        if self.stage_flag == "lr":
            batch_id = torch.arange(0, self.batch_size, device=self.device).long()
            self.df_lr_loss = self._calculate_loss(self.split_small, None, batch_id, "lr", None, self.df_type[0],
                                                   self.context, self.octree_graph)
        elif self.stage_flag == "hr":
            with torch.no_grad():
                self.input_data, self.doctree_in = self.autoencoder_module.extract_code(self.octree_in)
            
            batch_id = self.doctree_in.batch_id(self.small_depth)
            self.df_hr_loss = self._calculate_loss(self.input_data, self.doctree_in, batch_id, "hr", 
                                                   self.df_module.unet_lr, self.df_type[1], self.context, self.octree_graph)

        self.loss = self.df_lr_loss + self.df_hr_loss

    def get_sampling_timesteps(self, batch_size, device, steps):
        times = torch.linspace(1., 0., steps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch_size)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        return times.unbind(dim=-1)

    def get_graph_inf(self, device, graph_files, graph_dir):
        def curve2octree(points):
            octree = ocnn.octree.Octree(depth=self.input_depth, full_depth=self.full_depth)
            octree.build_octree(points)
            return octree

        curve_batch = []
        for graph_file in graph_files:
            graph_path = os.path.join(graph_dir, graph_file)
            if not os.path.exists(graph_path):
                cprint(f"Graph file not found: {graph_path}", 'yellow')
                continue
            
            data = sio.loadmat(graph_path)
            curve = data['new_curve']
            points = torch.from_numpy(curve[:, 0:3]).float().to(device)
            features = torch.from_numpy(curve[:, 3:7]).float().to(device)

            pts = Points(points=points, features=features)
            pts.clip(min=-1, max=1)
            curve_batch.append(pts)

        if not curve_batch:
            return None

        octrees = [curve2octree(pts) for pts in curve_batch]
        octree = ocnn.octree.merge_octrees(octrees)
        octree.construct_all_neigh()
        return octree

    def get_sampling_context(self, mask_dir, batch_size, device, file_list_path=None, sample_i=0):
        if file_list_path and os.path.exists(file_list_path):
            with open(file_list_path, 'r') as f:
                file_names = [line.strip() for line in f.readlines()]
            mask_files = [f'{name}.mat' for name in file_names]
            existing_mask_files = [f for f in mask_files if os.path.isfile(os.path.join(mask_dir, f))]
        else:
            existing_mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.mat')]

        if not existing_mask_files:
            raise FileNotFoundError(f"No mask files found in {mask_dir}.")

        start_index = sample_i * batch_size
        end_index = (sample_i + 1) * batch_size
        
        if start_index >= len(existing_mask_files):
            selected_mask_files = random.sample(existing_mask_files, min(batch_size, len(existing_mask_files)))
        else:
            selected_mask_files = existing_mask_files[start_index:end_index]

        masks = []
        for mask_file in selected_mask_files:
            mask_path = os.path.join(mask_dir, mask_file)
            mask_data = sio.loadmat(mask_path)['mask']
            mask_data = ndimage.distance_transform_edt(mask_data)
            mask_tensor = torch.tensor(mask_data, dtype=torch.float32).unsqueeze(0)
            masks.append(mask_tensor)
        
        context = torch.stack(masks).to(device)
        return context, selected_mask_files

    def convert_octree_to_points(self, octree, signal):
        depth = octree.depth
        batch_size = octree.batch_size
        displacement = signal[:, :3]

        x, y, z, _ = octree.xyzb(depth, nempty=True)
        xyz = torch.stack([x, y, z], dim=1) + 0.5 + displacement
        xyz = xyz / (2 ** (depth - 1)) - 1.0

        point_cloud = torch.cat([xyz, signal[:, 3:]], dim=1)
        batch_id = octree.batch_id(depth, nempty=True)
        points_per_batch = [torch.sum(batch_id == i) for i in range(batch_size)]
        return torch.split(point_cloud, points_per_batch)

    @torch.no_grad()
    def _sample_loop(self, shape, ema=False, ddim_steps=200, label=None, cond_info=None,
                     unet_type="lr", unet_lr=None, df_type="x0", truncated_index=0.0, doctree=None):
        
        # For HR graph sampling, the input shape is (N_nodes, C), but timesteps/labels are per-batch.
        # Use doctree.batch_size to drive the time schedule when doctree is provided.
        if doctree is not None:
            batch_size = doctree.batch_size
        else:
            batch_size = shape[0]
        time_pairs = self.get_sampling_timesteps(batch_size, device=self.device, steps=ddim_steps)
        
        context, octree_graph = None, None
        if cond_info:
            context = cond_info.get('context')
            octree_graph = cond_info.get('octree_graph')

        noised_data = torch.randn(shape, device=self.device)
        x_start = None

        time_iter = tqdm(time_pairs, desc=f'{unet_type} sampling loop')

        for t, t_next in time_iter:
            log_snr_t = self.log_snr(t)
            
            model = self.ema_df if ema else self.df
            output = model(unet_type=unet_type, x=noised_data, doctree=doctree, timesteps=log_snr_t,
                           unet_lr=unet_lr, x_self_cond=x_start, label=label, context=context,
                           graph_octree=octree_graph)

            if t[0] < truncated_index and unet_type == "lr":
                output.sign_()

            if df_type == "x0":
                x_start = output
            elif df_type == "eps":
                alpha_t, sigma_t = log_snr_to_alpha_sigma(log_snr_t)
                # For HR, alphas/sigmas should be scalars to broadcast over all nodes.
                if unet_type == "hr":
                    alpha_t, sigma_t = alpha_t[0], sigma_t[0]
                x_start = (noised_data - output * sigma_t) / alpha_t.clamp(min=1e-8)
            
            log_snr_next = self.log_snr(t_next)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)
            if unet_type == "hr":
                alpha_next, sigma_next = alpha_next[0], sigma_next[0]
            noised_data = x_start * alpha_next + output * sigma_next

        return noised_data

    @torch.no_grad()
    def sample(self, category='airplane', prefix='results', ema=True, ddim_steps=200, clean=False,
               save_index=0, cond=False, cond_dir=None, graph_dir=None, file_list_path=None, iter_i=0):

        model = self.ema_df if ema else self.df
        model.eval()

        batch_size = self.vq_config.data.test.batch_size
        label = None
        if self.enable_label:
            # This needs a mapping from category string to label index
            # label = category_to_label(category) 
            label = torch.randint(0, self.df_config.unet.params.num_classes, (batch_size,), device=self.device)

        save_dir = os.path.join(self.opt.logs_dir, self.opt.name, f"{prefix}_{category}")
        os.makedirs(save_dir, exist_ok=True)

        cond_info = None
        mask_filenames = None
        if cond:
            context, mask_filenames = self.get_sampling_context(
                mask_dir=cond_dir, batch_size=batch_size, device=self.device, 
                file_list_path=file_list_path, sample_i=iter_i
            )
            octree_graph = self.get_graph_inf(self.device, mask_filenames, graph_dir) if graph_dir else None
            cond_info = {'context': context, 'octree_graph': octree_graph}

        # Low-resolution sampling
        split_small = self._sample_loop(
            shape=(batch_size, *self.z_shape), ema=ema, ddim_steps=ddim_steps, label=label,
            cond_info=cond_info, unet_type="lr", df_type=self.df_type[0], truncated_index=0.0, doctree=None
        )

        # High-resolution sampling
        octree_in = split2octree_small(split_small, self.input_depth, self.full_depth)
        doctree_in = dual_octree.DualOctree(octree_in)
        doctree_in.post_processing_for_docnn()
        
        # For HR sampling, pass the LR UNet module (not its output) so HR UNet can call forward_as_middle internally
        unet_lr_module = self.df_module.unet_lr
        
        # Get the total number of nodes for HR sampling
        # doctree_in.total_num gives the total number of nodes at the current depth
        hr_num_nodes = doctree_in.total_num
        hr_shape_tuple = (hr_num_nodes, self.code_channel)
        
        sampled_code = self._sample_loop(
            shape=hr_shape_tuple, ema=ema, ddim_steps=ddim_steps, label=label,
            cond_info=cond_info, unet_type="hr", unet_lr=unet_lr_module, df_type=self.df_type[1], doctree=doctree_in
        )

        # Decode and save - decode_code returns a dictionary
        output = self.autoencoder_module.decode_code(sampled_code, doctree_in)
        octree_out = output['octree_out']
        signal = output['signal']
        
        for i in range(batch_size):
            idx = save_index + i
            mesh_save_dir = os.path.join(save_dir, f"{idx:04d}")
            os.makedirs(mesh_save_dir, exist_ok=True)

            if mask_filenames:
                sio.savemat(os.path.join(mesh_save_dir, 'mask.mat'), {'mask': cond_info['context'][i].cpu().numpy()})

            points = self.convert_octree_to_points(octree_out, signal)
            signal_np = points[i].cpu().numpy()
            sio.savemat(os.path.join(mesh_save_dir, 'signal.mat'), {'signal': signal_np})

            try:
                num_pts = signal_np.shape[0]
                cprint(f"Processing skeleton refinement for {num_pts} points", 'cyan')
                
                # Add downsampling for large point clouds to speed up processing
                if num_pts > 20000:
                    # Downsample to every 3rd point for very large clouds
                    signal_np = signal_np[::3]
                    num_pts = signal_np.shape[0]
                    cprint(f"Downsampled to {num_pts} points for faster processing", 'yellow')
                elif num_pts > 10000:
                    # Downsample to every 2nd point for moderately large clouds
                    signal_np = signal_np[::2]
                    num_pts = signal_np.shape[0]
                    cprint(f"Downsampled to {num_pts} points for faster processing", 'yellow')
                
                order = np.arange(num_pts).reshape(-1, 1)
                signal_aug = np.concatenate([signal_np[:, :3], order], axis=1)
                # Optimize parameters for faster processing
                skeleton_curve = refine_signal_to_skeleton(
                    signal_aug,
                    max_edge_dist=0.2,
                    recon_threshold=0.1,
                    min_nodes=10,  # Reduced from default 20
                    max_dist=0.4
                )
                sio.savemat(os.path.join(mesh_save_dir, 'skeleton.mat'), {'curve': skeleton_curve})
                cprint(f"Skeleton refinement completed, output curve has {len(skeleton_curve)} points", 'green')
            except Exception as e:
                cprint(f"Skeleton refinement failed for sample {idx}: {e}", 'red')

            # bbox = self.batch.get('bbox', [None]*batch_size)[i]
            # sdfs = self.get_sdfs(self.autoencoder_module.neural_mp, 1, bbox)
            # self.export_mesh(sdfs, mesh_save_dir, index=0, clean=clean)

    def get_sdfs(self, neural_mpu, batch_size, bbox):
        if bbox is not None:
            bbmin, bbmax = bbox[:3], bbox[3:]
        else:
            sdf_scale = self.solver_config.sdf_scale
            bbmin, bbmax = -sdf_scale, sdf_scale
        
        self.bbmin, self.bbmax = bbmin, bbmax
        return calc_sdf(neural_mpu, batch_size, size=self.solver_config.resolution, bbmin=bbmin, bbmax=bbmax)

    def export_mesh(self, sdfs, save_dir, index=0, level=0, clean=False):
        os.makedirs(save_dir, exist_ok=True)
        resolution = self.solver_config.resolution
        mesh_scale = self.vq_config.data.test.point_scale

        for i in range(sdfs.shape[0]):
            filename = os.path.join(save_dir, f'{index + i}.obj')
            sdf_values = sdfs[i].cpu().numpy()
            
            try:
                vtx, faces, _, _ = skimage.measure.marching_cubes(sdf_values, level)
            except (ValueError, RuntimeError) as e:
                cprint(f"Marching cubes failed: {e}", 'red')
                continue

            if vtx.size == 0 or faces.size == 0:
                cprint('Warning from marching cubes: Empty mesh!', 'yellow')
                continue
            
            vtx = vtx * ((self.bbmax - self.bbmin) / resolution) + self.bbmin
            vtx *= mesh_scale
            mesh = trimesh.Trimesh(vertices=vtx, faces=faces)

            if clean:
                components = mesh.split(only_watertight=False)
                if components:
                    bounding_box_sizes = [(c.vertices.max(0) - c.vertices.min(0)).max() for c in components]
                    mesh = components[np.argmax(bounding_box_sizes)]
            
            mesh.export(filename)

    def backward(self):
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
        if self.is_train:
            # Update EMA weights for the diffusion model
            self.ema_updater.update_model_average(self.ema_df, self.df)

    def get_current_errors(self):
        return {
            'loss': self.loss.item(),
            'df_lr_loss': self.df_lr_loss.item(),
            'df_hr_loss': self.df_hr_loss.item(),
            'lr': self.optimizer.param_groups[0]['lr'],
        }

    def save(self, label, global_iter):
        state_dict = {
            'df': self.df_module.state_dict(),
            'ema_df': self.ema_df.state_dict(),
            'opt': self.optimizer.state_dict(),
            'global_step': global_iter,
        }

        save_filename = f'df_{label}.pth'
        save_path = os.path.join(self.opt.ckpt_dir, save_filename)
        torch.save(state_dict, save_path)

    def load_ckpt(self, ckpt_path, df, ema_df, load_options):
        map_location = lambda storage, loc: storage
        if isinstance(ckpt_path, str):
            state_dict = torch.load(ckpt_path, map_location=map_location)
        else:
            state_dict = ckpt_path

        def load_part(model, state_dict_part):
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict_part.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        df_state_dict = state_dict.get('df', state_dict)
        ema_df_state_dict = state_dict.get('ema_df', state_dict)

        if "unet_lr" in load_options:
            # Try new format: nested dict with 'unet_lr' key
            if 'unet_lr' in df_state_dict:
                load_part(df.unet_lr, df_state_dict['unet_lr'])
                if 'unet_lr' in ema_df_state_dict:
                    load_part(ema_df.unet_lr, ema_df_state_dict['unet_lr'])
                cprint(f"[*] UNet-LR weights loaded from: {ckpt_path}", 'blue')
            # Fallback to old format: flat dict with 'df_unet_lr' key
            elif 'df_unet_lr' in state_dict:
                load_part(df.unet_lr, state_dict['df_unet_lr'])
                if 'ema_df_unet_lr' in state_dict:
                    load_part(ema_df.unet_lr, state_dict['ema_df_unet_lr'])
                cprint(f"[*] UNet-LR weights (old format) loaded from: {ckpt_path}", 'blue')

        if "unet_hr" in load_options:
            # Try new format
            if 'unet_hr' in df_state_dict:
                load_part(df.unet_hr, df_state_dict['unet_hr'])
                if 'unet_hr' in ema_df_state_dict:
                    load_part(ema_df.unet_hr, ema_df_state_dict['unet_hr'])
                cprint(f"[*] UNet-HR weights loaded from: {ckpt_path}", 'blue')
            # Fallback to old format
            elif 'df_unet_hr' in state_dict:
                load_part(df.unet_hr, state_dict['df_unet_hr'])
                if 'ema_df_unet_hr' in state_dict:
                    load_part(ema_df.unet_hr, state_dict['ema_df_unet_hr'])
                cprint(f"[*] UNet-HR weights (old format) loaded from: {ckpt_path}", 'blue')

        if "opt" in load_options and self.is_train and 'opt' in state_dict:
            self.start_iter = state_dict.get('global_step', 0)
            self.optimizer.load_state_dict(state_dict['opt'])
            cprint(f"[*] Optimizer state restored from: {ckpt_path}", 'blue')
