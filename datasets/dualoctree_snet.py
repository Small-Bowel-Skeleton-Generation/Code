import os
import random
import scipy
import scipy.io
import scipy.ndimage as ndimage
import torch
import numpy as np
from skimage import measure
from scipy.ndimage import gaussian_filter, zoom

from ocnn.octree import Octree, Points
from solver import Dataset
from .utils import collate_func
from discrete_frenet_solver import solve_frenet_frame


class GeometryTransform:
    """Transform geometry data including point clouds and octrees."""
    
    def __init__(self, flags):
        """Initialize geometry transform with configuration flags.
        
        Args:
            flags: Configuration object containing transform parameters
        """
        self.flags = flags
        self.depth = flags.depth
        self.full_depth = flags.full_depth
        self.point_sample_num = flags.point_sample_num
        self.point_scale = flags.point_scale
        self.noise_std = 0.005

    def points_to_octree(self, points: Points) -> Octree:
        """Convert points to octree structure.
        
        Args:
            points: Input point cloud
            
        Returns:
            Generated octree structure
        """
        octree = Octree(self.depth, self.full_depth)
        octree.build_octree(points)
        return octree

    def process_point_cloud(self, sample):
        """Process point cloud data by scaling and converting to tensors.
        
        Args:
            sample: Dictionary containing points, normals, and features
            
        Returns:
            Dictionary with processed point cloud data
        """
        points = sample['points'] / self.point_scale  # Scale to [-1.0, 1.0]
        normals = sample['normals']
        features = sample['features']

        # Convert to torch tensors
        points_gt = Points(
            points=torch.from_numpy(points).float(),
            normals=torch.from_numpy(normals).float(),
            features=torch.from_numpy(features).float()
        )
        
        if self.flags.load_color:
            points_gt.features = torch.from_numpy(sample['colors']).float()
        
        points_gt.clip(min=-1, max=1)
        return {'points': points_gt}

    def sample_sdf(self, sample):
        """Sample SDF values and gradients at random points.
        
        Args:
            sample: Dictionary containing SDF data
            
        Returns:
            Dictionary with sampled SDF data
        """
        sdf = sample['sdf']
        grad = sample['grad']
        points = sample['points'] / self.point_scale  # Scale to [-1, 1]

        rand_idx = np.random.choice(points.shape[0], size=self.point_sample_num)
        points = torch.from_numpy(points[rand_idx]).float()
        sdf = torch.from_numpy(sdf[rand_idx]).float()
        grad = torch.from_numpy(grad[rand_idx]).float()
        
        return {'pos': points, 'sdf': sdf, 'grad': grad}

    def sample_on_surface(self, points, normals):
        """Sample points on the surface with zero SDF values.
        
        Args:
            points: Surface points
            normals: Surface normals
            
        Returns:
            Dictionary with sampled surface data
        """
        rand_idx = np.random.choice(points.shape[0], size=self.point_sample_num)
        xyz = torch.from_numpy(points[rand_idx]).float()
        grad = torch.from_numpy(normals[rand_idx]).float()
        sdf = torch.zeros(self.point_sample_num)
        
        return {'pos': xyz, 'sdf': sdf, 'grad': grad}

    def sample_off_surface(self, xyz):
        """Sample points off the surface with negative SDF values.
        
        Args:
            xyz: Input point coordinates
            
        Returns:
            Dictionary with sampled off-surface data
        """
        xyz = xyz / self.point_scale  # Scale to [-1, 1]

        rand_idx = np.random.choice(xyz.shape[0], size=self.point_sample_num)
        xyz = torch.from_numpy(xyz[rand_idx]).float()
        grad = xyz / (xyz.norm(p=2, dim=1, keepdim=True) + 1.0e-6)
        sdf = -1 * torch.ones(self.point_sample_num)  # Negative SDF for off-surface
        
        return {'pos': xyz, 'sdf': sdf, 'grad': grad}

    def __call__(self, sample, idx):
        """Transform sample data based on configuration flags.
        
        Args:
            sample: Input sample data
            idx: Sample index
            
        Returns:
            Transformed sample data
        """
        output = {}

        if self.flags.load_octree:
            output['octree_in'] = sample['octree_in']

        if self.flags.load_pointcloud:
            output = self.process_point_cloud(sample['point_cloud'])

        if self.flags.load_curve:
            output['curve'] = self.process_point_cloud(sample['curve'])

        if self.flags.load_split_small:
            output['split_small'] = sample['split_small']

        if self.flags.load_split_large:
            output['split_large'] = sample['split_large']

        if self.flags.load_mask:
            output['context'] = sample['context']

        if self.flags.load_graph:
            output['graph_inf'] = sample['graph_inf']

        # Sample ground truth SDF values
        if self.flags.load_sdf:
            sdf_samples = self.sample_sdf(sample['sdf'])
            output.update(sdf_samples)

        # Sample both on-surface and off-surface points
        if self.flags.sample_surf_points:
            on_surf = self.sample_on_surface(sample['points'], sample['normals'])
            off_surf = self.sample_off_surface(sample['sdf']['points'])
            
            sdf_samples = {
                'pos': torch.cat([on_surf['pos'], off_surf['pos']], dim=0),
                'grad': torch.cat([on_surf['grad'], off_surf['grad']], dim=0),
                'sdf': torch.cat([on_surf['sdf'], off_surf['sdf']], dim=0)
            }
            output.update(sdf_samples)

        return output


class DataFileReader:
    """Read and process various data file formats."""
    
    def __init__(self, flags):
        """Initialize file reader with configuration flags.
        
        Args:
            flags: Configuration object containing file reading parameters
        """
        self.load_octree = flags.load_octree
        self.load_pointcloud = flags.load_pointcloud
        self.load_curve = flags.load_curve
        self.load_split_small = flags.load_split_small
        self.load_split_large = flags.load_split_large
        self.load_occu = flags.load_occu
        self.load_sdf = flags.load_sdf
        self.load_color = flags.load_color
        self.load_mask = flags.load_mask
        self.load_graph = flags.load_graph
        self.cond_dir = flags.cond_dir
        self.data_augmentation = flags.data_augmentation

    def add_random_surface_features(self, tubular_mask, num_features, 
                                   dent_radius_range, bump_radius_range, grid_size):
        """Add random surface features (dents and bumps) to mask.
        
        Args:
            tubular_mask: Input mask
            num_features: Number of features to add
            dent_radius_range: Range for dent radii
            bump_radius_range: Range for bump radii
            grid_size: Size of the grid
            
        Returns:
            Modified mask with surface features
        """
        verts, _, _, _ = measure.marching_cubes(tubular_mask, 0.5)
        verts = np.array(verts)
        mask_with_features = tubular_mask.copy()
        
        X, Y, Z = np.meshgrid(
            np.arange(grid_size[0]), 
            np.arange(grid_size[1]), 
            np.arange(grid_size[2]), 
            indexing="ij"
        )

        for _ in range(num_features):
            center = verts[np.random.randint(len(verts))]
            
            if np.random.rand() > 0.5:
                # Add dent
                radius = np.random.uniform(dent_radius_range[0], dent_radius_range[1])
                mask_with_features = self._apply_spherical_shape(
                    mask_with_features, X, Y, Z, center, -radius
                )
            else:
                # Add bump
                radius = np.random.uniform(bump_radius_range[0], bump_radius_range[1])
                mask_with_features = self._apply_spherical_shape(
                    mask_with_features, X, Y, Z, center, radius
                )

        return mask_with_features

    def _apply_spherical_shape(self, mask, X, Y, Z, center, radius):
        """Apply spherical modification to mask.
        
        Args:
            mask: Input mask
            X, Y, Z: Coordinate grids
            center: Center of the sphere
            radius: Radius (negative for dent, positive for bump)
            
        Returns:
            Modified mask
        """
        distances = np.sqrt(
            (X - center[0]) ** 2 + 
            (Y - center[1]) ** 2 + 
            (Z - center[2]) ** 2
        )
        
        if radius < 0:
            mask[distances <= abs(radius)] = False
        else:
            mask[distances <= radius] = True
            
        return mask

    def process_mask_with_noise(self, mask, sigma, scale_factor=0.5):
        """Process mask by adding smoothing and resolution changes.
        
        Args:
            mask: Input mask
            sigma: Gaussian filter sigma
            scale_factor: Scale factor for resolution reduction
            
        Returns:
            Processed mask
        """
        if sigma > 1:
            mask = gaussian_filter(mask.astype(float), sigma=sigma)
        
        # Threshold smoothed mask to boolean
        mask_binary = mask > 0.5
        mask_float = mask_binary.astype(float)
        
        # Reduce resolution using nearest neighbor interpolation
        low_res_mask = zoom(mask_float, zoom=scale_factor, order=0)
        
        # Upscale back to original size
        upscale_factor = 1 / scale_factor
        rough_mask = zoom(
            low_res_mask, 
            zoom=(upscale_factor, upscale_factor, upscale_factor),
            order=0
        )
        
        # Convert back to boolean
        final_mask = rough_mask > 0.5
        
        # Ensure shape consistency
        if final_mask.shape != mask.shape:
            final_mask = np.resize(final_mask, mask.shape)
            
        return final_mask

    def _load_octree_data(self, filename):
        """Load octree data from file."""
        octree_path = os.path.join(filename, 'octree.pth')
        raw = torch.load(octree_path)
        return raw['octree_in']

    def _load_pointcloud_data(self, filename):
        """Load point cloud data from file."""
        filename_pc = os.path.join(filename, 'pointcloud.npz')
        raw = np.load(filename_pc)

        point_cloud = {
            'points': raw['points'],
            'normals': raw['normals'],
            'features': raw['orders']
        }
        point_cloud['features'] = np.expand_dims(point_cloud['features'], axis=1)

        if self.load_color:
            filename_color = os.path.join(filename, 'color.npz')
            raw_color = np.load(filename_color)
            point_cloud['colors'] = raw_color['colors']
        else:
            point_cloud['colors'] = None

        return point_cloud

    def _load_curve_data(self, filename):
        """Load curve data from file."""
        filename_curve = os.path.join(filename, 'curve.npz')
        raw = np.load(filename_curve)
        
        curve = {
            'points': raw['points'],
            'normals': raw['normals'],
            'features': raw['orders']
        }
        curve['features'] = np.expand_dims(curve['features'], axis=1)
        return curve

    def _load_split_data(self, filename, split_type):
        """Load split data (small or large) from file."""
        filename_split = os.path.join(filename, f'split_{split_type}.pth')
        
        try:
            raw = torch.load(filename_split, map_location='cpu')
            return raw
        except Exception as e:
            print(f'Error loading {filename_split}: {e}')
            print(f'Filename: {filename}')
            return None

    def _load_sdf_data(self, filename):
        """Load SDF data from file."""
        # Handle dataset path replacement
        filename_sdf_dir = filename.replace('octfusion_dataset_4', 'octfusion_dataset_2')
        filename_sdf = os.path.join(filename_sdf_dir, 'sdf.npz')
        
        raw = np.load(filename_sdf)
        return {
            'points': raw['points'],
            'grad': raw['grad'],
            'sdf': raw['sdf']
        }

    def _load_mask_data(self, filename):
        """Load mask data from file."""
        base_name = os.path.basename(filename)
        mask_path = os.path.join(self.cond_dir, f"{base_name}.mat")
        
        mask_data = scipy.io.loadmat(mask_path)
        mask = mask_data['mask']
        
        # Apply distance transform weighting
        weighted_mask = ndimage.distance_transform_edt(mask)
        return weighted_mask

    def _load_graph_data(self, filename):
        """Load graph data from file."""
        condition_dir = '/home/data/liangzhichao/Code/upsample-clean-master/curve_condition/'
        base_name = os.path.basename(filename)
        cond_path = os.path.join(condition_dir, f"{base_name}.mat")
        
        cond_data = scipy.io.loadmat(cond_path)
        
        if 'new_curve' not in cond_data:
            raise ValueError(f"Missing 'new_curve' in {cond_path}")

        feature_data = cond_data['new_curve']
        points = feature_data[:, 0:3]
        features = feature_data[:, 3:7]
        
        graph_inf = Points(
            points=torch.from_numpy(points).float(),
            features=torch.from_numpy(features).float()
        )
        graph_inf.clip(min=-1, max=1)  # Limit point cloud range
        return graph_inf

    def _apply_data_augmentation(self, output):
        """Apply data augmentation by randomly permuting and flipping axes."""
        axis_permutation = np.random.permutation(3)
        flips = np.random.choice([-1, 1], 3)

        # Apply to point cloud data
        if 'point_cloud' in output and output['point_cloud'] is not None:
            points = output['point_cloud']['points']
            output['point_cloud']['points'] = points[:, axis_permutation] * flips
            normals = output['point_cloud']['normals']
            output['point_cloud']['normals'] = normals[:, axis_permutation] * flips

        # Apply to curve data
        if 'curve' in output and output['curve'] is not None:
            points = output['curve']['points']
            output['curve']['points'] = points[:, axis_permutation] * flips
            normals = output['curve']['normals']
            output['curve']['normals'] = normals[:, axis_permutation] * flips

        # Apply to context mask
        if 'context' in output and output['context'] is not None:
            mask = output['context']
            mask = np.transpose(mask, axes=axis_permutation)
            for i in range(3):
                if flips[i] == -1:
                    mask = np.flip(mask, axis=i)
            # Ensure array is contiguous in memory after transformations
            output['context'] = mask.copy()

    def __call__(self, filename):
        """Read and process data files.
        
        Args:
            filename: Base filename/directory path
            
        Returns:
            Dictionary containing loaded data
        """
        output = {}

        if self.load_octree:
            output['octree_in'] = self._load_octree_data(filename)

        if self.load_pointcloud:
            output['point_cloud'] = self._load_pointcloud_data(filename)

        if self.load_curve:
            output['curve'] = self._load_curve_data(filename)

        if self.load_split_small:
            output['split_small'] = self._load_split_data(filename, 'small')

        if self.load_split_large:
            output['split_large'] = self._load_split_data(filename, 'large')

        if self.load_occu:
            filename_occu = os.path.join(filename, 'points.npz')
            raw = np.load(filename_occu)
            output['occu'] = {
                'points': raw['points'],
                'occupancies': raw['occupancies']
            }

        if self.load_sdf:
            output['sdf'] = self._load_sdf_data(filename)

        if self.load_mask:
            output['context'] = self._load_mask_data(filename)

        if self.load_graph:
            output['graph_inf'] = self._load_graph_data(filename)

        if self.data_augmentation:
            self._apply_data_augmentation(output)

        return output


def get_shapenet_dataset(flags):
    """Create ShapeNet dataset with transforms and file reader.
    
    Args:
        flags: Configuration flags
        
    Returns:
        Tuple of (dataset, collate_function)
    """
    transform = GeometryTransform(flags)
    read_file = DataFileReader(flags)
    dataset = Dataset(
        flags.location, 
        flags.filelist, 
        transform,
        read_file=read_file, 
        in_memory=flags.in_memory
    )
    return dataset, collate_func
