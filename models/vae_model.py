import os
import copy
import scipy.io
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from omegaconf import OmegaConf
from termcolor import colored, cprint
import ocnn
from ocnn.octree import Octree
import trimesh
import skimage.measure

from models.base_model import BaseModel
from models.model_utils import load_dualoctree, set_requires_grad
from models.networks.dualoctree_networks import loss as dualoctree_loss
from utils.util_dualoctree import calc_sdf
from utils.distributed import get_rank

class VAEModel(BaseModel):
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.is_train = opt.mode == 'train'
        self.device = opt.device
        self.start_iter = opt.start_iter

        self._initialize_config(opt)
        self._initialize_vae_model(opt)
        self._initialize_optimizers(opt)
        self._load_checkpoint(opt)
        self._finalize_model_setup(opt)

    def _initialize_config(self, opt):
        assert opt.vq_cfg is not None, "VAE config path must be provided."
        self.vae_config = OmegaConf.load(opt.vq_cfg)
        self.solver_config = self.vae_config.solver
        self.data_config = self.vae_config.data.train
        
        self.input_depth = self.vae_config.model.depth
        self.full_depth = self.vae_config.model.full_depth

    def _initialize_vae_model(self, opt):
        self.vae_model = load_dualoctree(conf=self.vae_config, ckpt=opt.vq_ckpt, opt=opt)
        if self.is_train:
            self.vae_model.train()
            set_requires_grad(self.vae_model, True)

    def _initialize_optimizers(self, opt):
        if self.is_train:
            self.optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.vae_model.parameters()),
                lr=opt.lr
            )
            
            def poly_lr_scheduler(epoch, lr_power=0.9):
                return (1 - epoch / opt.epochs) ** lr_power

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, poly_lr_scheduler)
            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]
            self.print_networks(verbose=False)

    def _load_checkpoint(self, opt):
        ckpt_path = opt.ckpt
        if ckpt_path is None and self.is_train:
            latest_ckpt_path = os.path.join(opt.logs_dir, opt.name, "ckpt", "vae_steps-latest.pth")
            if os.path.exists(latest_ckpt_path):
                ckpt_path = latest_ckpt_path
        
        if ckpt_path is not None:
            self.load_ckpt(ckpt_path, self.vae_model, load_opt=self.is_train)
            if self.is_train:
                self.optimizers = [self.optimizer]

    def _finalize_model_setup(self, opt):
        trainable_params_num = sum(p.numel() for p in self.vae_model.parameters() if p.requires_grad)
        cprint(f"Trainable parameters: {trainable_params_num}", 'cyan')

        if opt.distributed:
            self._setup_distributed(opt)
            self.vae_model_module = self.vae_model.module
        else:
            self.vae_model_module = self.vae_model

    def _setup_distributed(self, opt):
        if opt.sync_bn:
            self.vae_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.vae_model)
        self.vae_model = nn.parallel.DistributedDataParallel(
            self.vae_model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

    def batch_to_cuda(self, batch):
        def points_to_octree(points):
            octree = Octree(depth=self.input_depth, full_depth=self.full_depth)
            octree.build_octree(points)
            return octree

        if self.data_config.load_pointcloud:
            points = [pts['points'].to(self.device, non_blocking=True) for pts in batch['curve']]
            octrees = [points_to_octree(pts) for pts in points]
            octree = ocnn.octree.merge_octrees(octrees)
            octree.construct_all_neigh()
            batch['octree_in'] = octree

        if self.data_config.load_octree:
            batch['octree_in'] = batch['octree_in'].to(self.device)

        for key in ['pos', 'sdf', 'grad']:
            if key in batch:
                batch[key] = batch[key].to(self.device)
        if 'pos' in batch:
            batch['pos'].requires_grad = True

    def set_input(self, data):
        self.batch_to_cuda(data)
        self.octree_in = data['octree_in']
        self.octree_gt = copy.deepcopy(self.octree_in)
        self.batch = data
        self.batch_size = self.octree_in.batch_size

    def switch_to_train(self):
        self.vae_model.train()

    def switch_to_eval(self):
        self.vae_model.eval()

    def get_loss_function(self):
        loss_name = self.vae_config.loss.name.lower()
        if loss_name == 'geometry':
            return dualoctree_loss.geometry_loss
        elif loss_name == 'color':
            return dualoctree_loss.geometry_color_loss
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

    def forward(self):
        self.switch_to_train()
        model_out = self.vae_model_module(self.octree_in, self.octree_gt, self.batch['pos'])
        loss_func = self.get_loss_function()
        output = loss_func(self.batch, model_out, self.vae_config.loss.loss_type, kl_weight=self.vae_config.loss.kl_weight)
        
        losses = [val for key, val in output.items() if 'loss' in key]
        output['loss'] = torch.sum(torch.stack(losses))
        
        self.loss = output['loss']
        self.output = output

    @torch.no_grad()
    def inference(self, save_folder="results_vae"):
        self.switch_to_eval()
        output = self.vae_model.forward(octree_in=self.batch['octree_in'], evaluate=True)
        
        filename = os.path.splitext(self.batch['filename'][0])[0]
        save_dir = os.path.join(self.opt.logs_dir, self.opt.name, save_folder, filename)
        os.makedirs(save_dir, exist_ok=True)

        self._save_output_signal(output, save_dir)
        self._generate_and_export_mesh(output, save_dir)
        self._export_input_pointcloud(save_dir)

    def _save_output_signal(self, output, save_dir):
        signal = self.convert_octree_to_points(output['octree_out'], output['signal'])
        signal_np = signal[0].cpu().numpy()
        signal_mat_path = os.path.join(save_dir, 'signal.mat')
        scipy.io.savemat(signal_mat_path, {'signal': signal_np})

    def _generate_and_export_mesh(self, output, save_dir):
        bbox = self.batch['bbox'][0].numpy() if 'bbox' in self.batch else None
        sdfs = self._calculate_sdfs(output['neural_mpu'], self.batch_size, bbox)
        self.export_mesh_from_sdfs(sdfs, save_dir, index=0)

    def _export_input_pointcloud(self, save_dir):
        input_points = self.batch['curve'][0]['points'].cpu().numpy()
        pointcloud = trimesh.PointCloud(vertices=input_points)
        pointcloud.export(os.path.join(save_dir, 'input.ply'))

    def _calculate_sdfs(self, neural_mpu, batch_size, bbox):
        if bbox is not None:
            bbmin, bbmax = bbox[:3], bbox[3:]
        else:
            sdf_scale = self.solver_config.sdf_scale
            bbmin, bbmax = -sdf_scale, sdf_scale
        
        self.bbmin, self.bbmax = bbmin, bbmax
        return calc_sdf(neural_mpu, batch_size, size=self.solver_config.resolution, bbmin=bbmin, bbmax=bbmax)

    def export_mesh_from_sdfs(self, sdfs, save_dir, index=0, level=0, clean=False):
        os.makedirs(save_dir, exist_ok=True)
        num_meshes = sdfs.shape[0]
        resolution = self.solver_config.resolution
        mesh_scale = self.vae_config.data.test.point_scale

        for i in range(num_meshes):
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
                mesh = self._clean_mesh(mesh)
            
            mesh.export(filename)

    def _clean_mesh(self, mesh):
        components = mesh.split(only_watertight=False)
        if not components:
            return mesh
        
        bounding_box_sizes = [(c.vertices.max(0) - c.vertices.min(0)).max() for c in components]
        largest_component = components[np.argmax(bounding_box_sizes)]
        return largest_component

    def export_pointcloud(self, save_dir, octree, index=0, depth=None):
        os.makedirs(save_dir, exist_ok=True)
        points_list = []
        depth_range = range(octree.full_depth, octree.depth + 1) if depth is None else [depth]

        for d in depth_range:
            if octree.nnum[d] > 0:
                keys = octree.keys[d].cpu().numpy()
                size = 2 ** d
                
                x = (keys >> 0) & (size - 1)
                y = (keys >> d) & (size - 1)
                z = (keys >> (2 * d)) & (size - 1)

                points = np.stack([x, y, z], axis=1).astype(np.float32)
                points = (points / size) * 2 - 1
                points_list.append(points)

        if not points_list:
            cprint("No points found to export.", 'yellow')
            return

        points = np.vstack(points_list)
        filename = os.path.join(save_dir, f'{index}.ply')
        pointcloud = trimesh.PointCloud(vertices=points)
        pointcloud.export(filename)

    def convert_octree_to_points(self, octree, signal):
        depth = octree.depth
        batch_size = octree.batch_size
        displacement = signal[:, :3]

        x, y, z, _ = octree.xyzb(depth, nempty=True)
        xyz = torch.stack([x, y, z], dim=1) + 0.5 + displacement
        xyz = xyz / (2 ** (depth - 1)) - 1.0

        point_cloud = torch.cat([xyz, signal[:, 3:]], dim=1)
        batch_ids = octree.batch_id(depth, nempty=True)
        points_per_batch = [torch.sum(batch_ids == i) for i in range(batch_size)]
        
        return torch.split(point_cloud, points_per_batch)

    def backward(self):
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def get_current_errors(self):
        errors = OrderedDict([
            ('loss', self.loss.item()),
            ('lr', self.optimizer.param_groups[0]['lr']),
        ])
        errors.update({k: v.item() for k, v in self.output.items() if 'loss' in k})
        return errors

    def save(self, label, global_iter):
        state_dict = {
            'autoencoder': self.vae_model_module.state_dict(),
            'opt': self.optimizer.state_dict(),
            'global_step': global_iter,
        }

        save_filename = f'vae_{label}.pth'
        save_path = os.path.join(self.opt.ckpt_dir, save_filename)

        self._manage_checkpoints()
        torch.save(state_dict, save_path)

    def _manage_checkpoints(self):
        ckpts = [f for f in os.listdir(self.opt.ckpt_dir) if f.endswith('.pth') and 'latest' not in f]
        ckpts.sort(key=lambda x: int(x.split('')[-1].split('.')[0]))
        
        if len(ckpts) > self.opt.ckpt_num:
            for ckpt_to_remove in ckpts[:-self.opt.ckpt_num]:
                os.remove(os.path.join(self.opt.ckpt_dir, ckpt_to_remove))

    def load_ckpt(self, ckpt_path, model, load_opt=False):
        map_location = lambda storage, loc: storage
        if isinstance(ckpt_path, str):
            state_dict = torch.load(ckpt_path, map_location=map_location)
        else:
            state_dict = ckpt_path

        model.load_state_dict(state_dict['autoencoder'])
        cprint(f'[*] Weights successfully loaded from: {ckpt_path}', 'blue')

        if load_opt and self.is_train:
            self.start_iter = state_dict.get('global_step', 0)
            self.optimizer.load_state_dict(state_dict['opt'])
            cprint(f'[*] Training state (optimizer, step) restored from: {ckpt_path}', 'blue')


