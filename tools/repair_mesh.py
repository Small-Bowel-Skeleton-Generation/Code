import os
import torch
import ocnn
import logging
import argparse
import scipy
from scipy.interpolate import interp1d
import numpy as np
from tqdm import tqdm
from discrete_frenet_solver import solve_frenet_frame

# Suppress trimesh logging, which can be verbose
logging.getLogger("trimesh").setLevel(logging.ERROR)


def ensure_folder_exists(filepaths: list):
    """
    Ensures that the directory for each given filepath exists.
    If it does not exist, it will be created.
    """
    for filepath in filepaths:
        folder = os.path.dirname(filepath)
        if not os.path.exists(folder):
            os.makedirs(folder)


def get_filenames(root_folder, list_filename):
    """
    Reads a list of filenames from a specified file.
    """
    filelist_path = os.path.join(root_folder, 'filelist', list_filename)
    with open(filelist_path, 'r') as fid:
        lines = fid.readlines()
    filenames = [line.strip().split()[0] for line in lines]
    return filenames


def process_curve_data(args):
    """
    Reads 3D curve data from .mat files, interpolates to a specified number of points,
    normalizes coordinates to the range [-0.5, 0.5], centers the data, and calculates
    the Frenet frame (tangents, normals, and binormals) for each point on the curve.
    """
    print('-> Processing 3D curve data...')

    output_folder = os.path.join(args.root_folder, 'dataset')
    os.makedirs(output_folder, exist_ok=True)

    filenames = get_filenames(args.root_folder, 'all.txt')

    for filename in tqdm(filenames, ncols=80):
        mat_filepath = os.path.join(args.curve_data_folder, filename + '.mat')
        output_filepath = os.path.join(output_folder, filename, 'curve.npz')

        ensure_folder_exists([output_filepath])
        if os.path.exists(output_filepath):
            continue

        data = scipy.io.loadmat(mat_filepath)
        curve_points = np.array(data['smoothedCurve'])

        t_original = np.linspace(0, 1, len(curve_points))
        t_target = np.linspace(0, 1, args.num_samples)
        interpolator = interp1d(t_original, curve_points, axis=0, kind='linear')
        interpolated_points = interpolator(t_target)

        min_bound = np.min(interpolated_points, axis=0)
        max_bound = np.max(interpolated_points, axis=0)
        center = (min_bound + max_bound) / 2
        scale = (max_bound - min_bound).max()
        normalized_points = (interpolated_points - center) / scale
        normalized_points *= 0.9

        orders = np.linspace(-1, 1, normalized_points.shape[0], dtype=np.float32)

        T, N, B = solve_frenet_frame(normalized_points)

        np.savez(output_filepath,
                 points=normalized_points.astype(np.float16),
                 tangents=T.astype(np.float16),
                 normals=N.astype(np.float16),
                 binormals=B.astype(np.float16),
                 orders=orders.astype(np.float16))


def sample_sdf_from_curve(args):
    """
    Samples ground-truth SDF values for training. This function uses the processed
    curve data and pre-computed SDFs to generate training samples.
    """
    print('-> Sampling SDFs from the ground truth.')

    depth, full_depth = 6, 4
    sample_num = 4  # Number of samples in each octree node

    filenames = get_filenames(args.root_folder, 'all.txt')
    for i in tqdm(range(args.start, args.end), ncols=80):
        if i >= len(filenames):
            break
        filename = filenames[i]
        dataset_folder = os.path.join(args.root_folder, 'dataset')
        sdf_filepath = os.path.join(args.root_folder, 'sdf', filename + '.npy')
        curve_filepath = os.path.join(dataset_folder, filename, 'curve.npz')
        output_filepath = os.path.join(dataset_folder, filename, 'sdf.npz')

        ensure_folder_exists([output_filepath])
        if os.path.exists(output_filepath):
            continue

        if not os.path.exists(curve_filepath) or not os.path.exists(sdf_filepath):
            print(f"Warning: Missing data for {filename}. Skipping.")
            continue

        curve_data = np.load(curve_filepath)
        sdf = torch.from_numpy(np.load(sdf_filepath))
        points = curve_data['points'].astype(np.float32) / args.shape_scale
        normals = curve_data['normals'].astype(np.float32)

        ocnn_points = ocnn.octree.Points(torch.from_numpy(points), torch.from_numpy(normals))
        octree = ocnn.octree.Octree(depth=depth, full_depth=full_depth)
        octree.build_octree(ocnn_points)

        xyzs, sdfs = [], []
        for d in range(full_depth, depth + 1):
            xyzb = octree.xyzb(d)
            x, y, z, b = xyzb
            xyz = torch.stack([x, y, z], dim=1).float()
            xyz = xyz.unsqueeze(1) + torch.rand(xyz.shape[0], sample_num, 3)
            xyz = xyz.view(-1, 3)
            xyz = xyz * (args.sdf_size / 2 ** d)
            xyzs.append(xyz)

        if xyzs:
            all_xyz = torch.cat(xyzs, dim=0).numpy()
            np.savez(output_filepath, sampled_points=all_xyz)
        else:
            np.savez(output_filepath)


def main():
    parser = argparse.ArgumentParser(description="Preprocess curve data for diffusion models.")
    parser.add_argument('--run', type=str, required=True,
                        choices=['process_curve', 'sample_sdf', 'generate_dataset'],
                        help='The task to run.')
    parser.add_argument('--root_folder', type=str, required=True,
                        help='Root directory for the dataset.')
    parser.add_argument('--curve_data_folder', type=str, required=True,
                        help='Directory containing the raw curve .mat files.')
    parser.add_argument('--num_samples', type=int, default=5000,
                        help='Number of points to interpolate for each curve.')
    parser.add_argument('--start', type=int, default=0,
                        help='Start index for processing files.')
    parser.add_argument('--end', type=int, default=100000,
                        help='End index for processing files.')
    parser.add_argument('--sdf_size', type=int, default=128,
                        help='Resolution of the SDF grid.')
    parser.add_argument('--shape_scale', type=float, default=0.5,
                        help='Scale factor for shapes.')
    args = parser.parse_args()

    if args.run == 'process_curve':
        process_curve_data(args)
    elif args.run == 'sample_sdf':
        sample_sdf_from_curve(args)
    elif args.run == 'generate_dataset':
        print("Running full dataset generation pipeline...")
        print("\nStep 1: Processing curve data...")
        process_curve_data(args)
        print("\nStep 2: Sampling SDF values from curves...")
        sample_sdf_from_curve(args)
        print("\nDataset generation complete.")


if __name__ == '__main__':
    main()


# python e:\phdplat\Code\Tree-diffuison-update\tools\repair_mesh.py --run generate_dataset --root_folder /mnt/gemlab_data_2/User_database/liangzhichao/octfusion_dataset_6 --curve_data_folder /home/data/liangzhichao/Data/skeleton_point4