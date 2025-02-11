# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import ocnn
import torch
import numpy as np
import copy
import scipy
import scipy.ndimage as ndimage

from ocnn.octree import Octree, Points
from solver import Dataset
from .utils import collate_func
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
from scipy.ndimage import gaussian_filter, zoom
import cupy as cp  # 替换 numpy 为 cupy
from skimage import measure
import random


class TransformShape:

    def __init__(self, flags):
        self.flags = flags

        self.depth = flags.depth
        self.full_depth = flags.full_depth

        self.point_sample_num = flags.point_sample_num
        self.point_scale = flags.point_scale
        self.noise_std = 0.005

    def points2octree(self, points: Points):
        octree = Octree(self.depth, self.full_depth)
        octree.build_octree(points)
        return octree

    # def process_points_cloud(self, sample):
    #     # get the input
    #     points, normals = sample['points'], sample['normals']
    #     points = points / self.point_scale    # scale to [-1.0, 1.0]
    #
    #     # transform points to octree
    #     points_gt = Points(points = torch.from_numpy(points).float(), normals = torch.from_numpy(normals).float())
    #     if self.flags.load_color:
    #         points_gt.features = torch.from_numpy(sample['colors']).float()
    #     points_gt.clip(min=-1, max=1)
    #
    #     return {'points': points_gt}

    def process_points_cloud(self, sample):
        # get the input
        points, normals, features = sample['points'], sample['normals'], sample['features']
        points = points / self.point_scale    # scale to [-1.0, 1.0]

        # transform points to octree
        points_gt = Points(points = torch.from_numpy(points).float(), normals = torch.from_numpy(normals).float(), features = torch.from_numpy(features).float())
        if self.flags.load_color:
            points_gt.features = torch.from_numpy(sample['colors']).float()
        points_gt.clip(min=-1, max=1)

        return {'points': points_gt}

    def sample_sdf(self, sample):     # 这里加载的sdf的坐标也都是在[-1,1]范围内的。
        sdf = sample['sdf']
        grad = sample['grad']
        points = sample['points'] / self.point_scale    # to [-1, 1]

        rand_idx = np.random.choice(points.shape[0], size=self.point_sample_num)
        points = torch.from_numpy(points[rand_idx]).float()
        sdf = torch.from_numpy(sdf[rand_idx]).float()
        grad = torch.from_numpy(grad[rand_idx]).float()
        return {'pos': points, 'sdf': sdf, 'grad': grad}

    def sample_on_surface(self, points, normals):
        rand_idx = np.random.choice(points.shape[0], size=self.point_sample_num)
        xyz = torch.from_numpy(points[rand_idx]).float()
        grad = torch.from_numpy(normals[rand_idx]).float()
        sdf = torch.zeros(self.point_sample_num)
        return {'pos': xyz, 'sdf': sdf, 'grad': grad}

    def sample_off_surface(self, xyz):
        xyz = xyz / self.point_scale    # to [-1, 1]

        rand_idx = np.random.choice(xyz.shape[0], size=self.point_sample_num)
        xyz = torch.from_numpy(xyz[rand_idx]).float()
        # grad = torch.zeros(self.sample_number, 3)    # dummy grads
        grad = xyz / (xyz.norm(p=2, dim=1, keepdim=True) + 1.0e-6)
        sdf = -1 * torch.ones(self.point_sample_num)    # dummy sdfs
        return {'pos': xyz, 'sdf': sdf, 'grad': grad}

    def __call__(self, sample, idx):
        output = {}

        if self.flags.load_octree:
            output['octree_in'] = sample['octree_in']

        if self.flags.load_pointcloud:
            output = self.process_points_cloud(sample['point_cloud'])

        if self.flags.load_curve:
            output['curve'] = self.process_points_cloud(sample['curve'])

        if self.flags.load_split_small:
            output['split_small'] = sample['split_small']

        if self.flags.load_split_large:
            output['split_large'] = sample['split_large']

        if self.flags.load_mask:
            output['context'] = sample['context']

        # sample ground truth sdfs
        if self.flags.load_sdf:
            sdf_samples = self.sample_sdf(sample['sdf'])
            output.update(sdf_samples)

        # sample on surface points and off surface points
        if self.flags.sample_surf_points:
            on_surf = self.sample_on_surface(sample['points'], sample['normals'])
            off_surf = self.sample_off_surface(sample['sdf']['points'])    # TODO
            sdf_samples = {
                    'pos': torch.cat([on_surf['pos'], off_surf['pos']], dim=0),
                    'grad': torch.cat([on_surf['grad'], off_surf['grad']], dim=0),
                    'sdf': torch.cat([on_surf['sdf'], off_surf['sdf']], dim=0)}
            output.update(sdf_samples)

        return output


class ReadFile:
    def __init__(self, flags):
        self.load_octree = flags.load_octree
        self.load_pointcloud = flags.load_pointcloud
        self.load_curve = flags.load_curve
        self.load_split_small = flags.load_split_small
        self.load_split_large = flags.load_split_large
        self.load_occu = flags.load_occu
        self.load_sdf = flags.load_sdf
        self.load_color = flags.load_color
        self.load_mask = flags.load_mask
        self.cond_dir = flags.cond_dir

    def add_random_dents_and_bumps(self, tubular_mask, num_dents, dent_radius_range, bump_radius_range, grid_size):
        verts, faces, _, _ = measure.marching_cubes(tubular_mask, 0.5)
        verts = np.array(verts)  # 使用 NumPy 数组
        mask_with_dents_and_bumps = tubular_mask.copy()
        X, Y, Z = np.meshgrid(np.arange(grid_size[0]), np.arange(grid_size[1]), np.arange(grid_size[2]), indexing="ij")

        for _ in range(num_dents):
            center = verts[np.random.randint(len(verts))]
            if np.random.rand() > 0.5:
                radius = np.random.uniform(dent_radius_range[0], dent_radius_range[1])
                mask_with_dents_and_bumps = self.apply_spherical_shape(mask_with_dents_and_bumps, X, Y, Z, center,
                                                                       -radius)
            else:
                radius = np.random.uniform(bump_radius_range[0], bump_radius_range[1])
                mask_with_dents_and_bumps = self.apply_spherical_shape(mask_with_dents_and_bumps, X, Y, Z, center,
                                                                       radius)

        return mask_with_dents_and_bumps

    def apply_spherical_shape(self, mask, X, Y, Z, center, radius):
        distances = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2)
        if radius < 0:
            mask[distances <= abs(radius)] = False
        else:
            mask[distances <= radius] = True
        return mask

    def process_mask(self, mask, sigma, scale_factor=0.5):
        if sigma > 1:
            mask = gaussian_filter(mask.astype(float), sigma=sigma)
        # 将平滑后的掩码阈值化为布尔掩码
        mask_show = mask > 0.5
        # 确保输入是布尔类型
        mask_show = mask_show.astype(float)
        # 降低分辨率：使用最近邻插值
        low_res_mask = zoom(mask_show, zoom=scale_factor, order=0)  # order=0 表示最近邻插值
        # 将低分辨率掩码放大到原始大小：使用最近邻插值
        rough_mask = zoom(low_res_mask, zoom=(1 / scale_factor, 1 / scale_factor, 1 / scale_factor),
                          order=0)  # 放大比例是缩小的倒数
        # 将放大后的掩码阈值化为布尔类型
        final_mask = rough_mask > 0.5
        # 确保最终掩码的形状与输入掩码相同
        if final_mask.shape != mask.shape:
            final_mask = np.resize(final_mask, mask.shape)
        return final_mask

    def __call__(self, filename):
        output = {}
        # filename = '/mnt/gemlab_data_2/User_database/liangzhichao/octfusion_dataset_5/dataset/PA1478_04110'

        if self.load_octree:
            octree_path = os.path.join(filename, 'octree.pth')
            raw = torch.load(octree_path)
            octree_in = raw['octree_in']
            output['octree_in'] = octree_in

        if self.load_pointcloud:
            filename_pc = os.path.join(filename, 'pointcloud.npz')
            raw = np.load(filename_pc)

            point_cloud = {'points': raw['points'], 'normals': raw['normals'], 'features': raw['orders']}
            point_cloud['features'] = np.expand_dims(point_cloud['features'], axis=1)

            # filename_pc = os.path.join(filename, 'pointcloud.npz')
            # raw = np.load(filename_pc)
            # point_cloud = {'points': raw['points'], 'normals': raw['normals']}
            # order = np.linspace(-1, 1, raw['points'].shape[0], dtype=np.float32)  # Generate order info in [0, 1]
            # point_cloud['features']  = np.expand_dims(order, axis=1)

            if self.load_color:
                filename_color = os.path.join(filename, 'color.npz')
                raw = np.load(filename_color)
                point_cloud['colors'] = raw['colors']
            else:
                point_cloud['colors'] = None

            output['point_cloud'] = point_cloud

        if self.load_curve:
            filename_cu = os.path.join(filename, 'curve.npz')
            raw = np.load(filename_cu)
            curve = {'points': raw['points'], 'normals': raw['normals'], 'features': raw['orders']}
            curve['features'] = np.expand_dims(curve['features'], axis=1)
            output['curve'] = curve


        if self.load_split_small:
            filename_split_small = os.path.join(filename, 'split_small.pth')
            raw = torch.load(filename_split_small, map_location = 'cpu')
            output['split_small'] = raw

        if self.load_split_large:
            filename_split_large = os.path.join(filename, 'split_large.pth')
            try:
                raw = torch.load(filename_split_large, map_location = 'cpu')
            except:
                print('Error!!')
                print(filename)
            output['split_large'] = raw

        if self.load_occu:
            filename_occu = os.path.join(filename, 'points.npz')
            raw = np.load(filename_occu)
            occu = {'points': raw['points'], 'occupancies': raw['occupancies']}
            output['occu'] = occu

        if self.load_sdf:
            filename_ = filename.replace('octfusion_dataset_4', 'octfusion_dataset_2')
            filename_sdf = os.path.join(filename_, 'sdf.npz')
            raw = np.load(filename_sdf)
            sdf = {'points': raw['points'], 'grad': raw['grad'], 'sdf': raw['sdf']}
            output['sdf'] = sdf

        if self.load_mask:
            mask_filename = self.cond_dir
            # Construct the mask file path based on the filename
            base_name = os.path.basename(filename)
            mask_path = os.path.join(mask_filename, f"{base_name}.mat")
            mask_data = scipy.io.loadmat(mask_path)
            output['context'] = mask_data['mask']

            # # 设置参数
            # num_dents = 10  # 随机突起和凹陷的数量
            # dent_radius_range = (1.0, 3.0)  # 突起和凹陷的半径范围
            # bump_radius_range = (3.0, 5.0)  # 凹陷的半径范围
            # grid_size = output['context'].shape  # 获取掩码的形状
            # # 添加随机突起和凹陷
            # output['context'] = self.add_random_dents_and_bumps(output['context'], num_dents, dent_radius_range,
            #                                                bump_radius_range, grid_size)
            #
            # # 随机选择sigma
            # sigma = random.choice([1, 2, 3, 4])
            # # sigma = 2
            # # 随机选择scale_factor
            # scale_factor = random.uniform(0.7, 0.9)
            # output['context'] = self.process_mask(output['context'], sigma, scale_factor)

        # save_path = '/home/data/liangzhichao/Code/octfusion-main/data/'
        # mat_file_name = os.path.join(save_path, os.path.splitext(os.path.basename(filename))[0] + '.mat')
        # data_to_save = {
        #     'points': point_cloud['points'],
        #     'mask': output['context']
        # }
        # scipy.io.savemat(mat_file_name, data_to_save)

        return output


def get_shapenet_dataset(flags):
    transform = TransformShape(flags)
    read_file = ReadFile(flags)
    dataset = Dataset(flags.location, flags.filelist, transform,
                                        read_file=read_file, in_memory=flags.in_memory)
    return dataset, collate_func
