import os
import argparse
import scipy.io
import scipy.interpolate
import cupy as cp
from skimage import measure
from scipy.ndimage import gaussian_filter, zoom
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def set_gpu_device(gpu_id):
    """Sets the GPU device to be used by CuPy."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

def split_curve_at_center(curve, threshold_distance_range):
    """
    Splits a 3D curve by removing a segment around its center point.

    Args:
        curve (cp.ndarray): A cupy array of shape (N, 3) representing the curve.
        threshold_distance_range (list): A list [min, max] for the random distance threshold.

    Returns:
        cp.ndarray: The modified curve with a segment removed.
    """
    center_point = cp.mean(curve, axis=0)
    distances_to_center = cp.linalg.norm(curve - center_point, axis=1)
    center_idx = int(cp.argmin(distances_to_center))

    points_to_remove = [center_idx]
    threshold = threshold_distance_range[0] + cp.random.rand() * (threshold_distance_range[1] - threshold_distance_range[0])

    # Expand removal towards the start of the curve
    for i in range(center_idx - 1, -1, -1):
        if cp.linalg.norm(curve[i + 1] - curve[i]) < threshold:
            points_to_remove.append(i)
        else:
            break
    
    # Expand removal towards the end of the curve
    for i in range(center_idx + 1, len(curve)):
        if cp.linalg.norm(curve[i - 1] - curve[i]) < threshold:
            points_to_remove.append(i)
        else:
            break
    
    points_to_remove.sort()
    
    mask = cp.ones(len(curve), dtype=bool)
    mask[points_to_remove] = False
    
    return curve[mask]


def add_dents_and_bumps(tubular_mask, num_dents, dent_radius_range, bump_radius_range, grid_size):
    """
    Adds random spherical dents and bumps to the surface of a binary mask.

    Args:
        tubular_mask (cp.ndarray): The input binary mask.
        num_dents (int): The number of modifications to apply.
        dent_radius_range (tuple): (min, max) radius for dents.
        bump_radius_range (tuple): (min, max) radius for bumps.
        grid_size (tuple): The dimensions of the grid (e.g., (128, 128, 128)).

    Returns:
        cp.ndarray: The mask with added dents and bumps.
    """
    try:
        verts, _, _, _ = measure.marching_cubes(tubular_mask.get(), 0.5)
    except (ValueError, RuntimeError):
        # Marching cubes can fail if the surface is not found or is ambiguous.
        print("Warning: Marching cubes failed. Skipping dents and bumps.")
        return tubular_mask

    if len(verts) == 0:
        return tubular_mask

    verts = cp.array(verts)
    modified_mask = tubular_mask.copy()
    X, Y, Z = cp.meshgrid(cp.arange(grid_size[0]), cp.arange(grid_size[1]), cp.arange(grid_size[2]), indexing="ij")

    for _ in range(num_dents):
        center = verts[cp.random.randint(len(verts))]
        is_bump = cp.random.rand() > 0.5
        
        if is_bump:
            radius = cp.random.uniform(bump_radius_range[0], bump_radius_range[1])
            modified_mask = apply_spherical_modification(modified_mask, X, Y, Z, center, radius, add=True)
        else:
            radius = cp.random.uniform(dent_radius_range[0], dent_radius_range[1])
            modified_mask = apply_spherical_modification(modified_mask, X, Y, Z, center, radius, add=False)

    return modified_mask


def apply_spherical_modification(mask, X, Y, Z, center, radius, add=True):
    """Helper function to add (bump) or remove (dent) a sphere from a mask."""
    distances = cp.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
    if add:
        mask[distances <= radius] = True
    else:
        mask[distances <= radius] = False
    return mask


def create_tubular_mask(curve, min_radius, max_radius, grid_size, max_radius_change,
                        xyz_deformation_min, xyz_deformation_max):
    """
    Generates a tubular mask around a curve with varying, anisotropically scaled radii.
    """
    num_interp_points = 2048
    t = cp.linspace(0, 1, curve.shape[0])
    t_interp = cp.linspace(0, 1, num_interp_points)

    interpolator = scipy.interpolate.interp1d(cp.asnumpy(t), cp.asnumpy(curve), kind='linear', axis=0)
    interpolated_curve = cp.array(interpolator(cp.asnumpy(t_interp)))

    min_vals, max_vals = cp.min(interpolated_curve, axis=0), cp.max(interpolated_curve, axis=0)
    center, range_vals = (max_vals + min_vals) / 2, max_vals - min_vals
    normalized_curve = (interpolated_curve - center) / (cp.max(range_vals) / 2) * 0.8

    grid_points = cp.linspace(-1, 1, grid_size[0])
    X, Y, Z = cp.meshgrid(grid_points, grid_points, grid_points, indexing='ij')
    distance_field = cp.full(grid_size, cp.inf)

    scale_factor = cp.max(range_vals) / 2
    adj_min_radius = min_radius / scale_factor
    adj_max_radius = max_radius / scale_factor
    adj_max_change = max_radius_change / scale_factor

    num_points = normalized_curve.shape[0]
    smooth_radii = cp.zeros(num_points)
    smooth_radii[0] = cp.random.uniform(adj_min_radius, adj_max_radius)

    for i in range(1, num_points):
        change = (cp.random.rand() - 0.5) * 2 * adj_max_change
        potential_radius = cp.clip(smooth_radii[i-1] + change, adj_min_radius, adj_max_radius)
        smooth_radii[i] = potential_radius

    deformation_matrices = [create_anisotropic_deformation_matrix(xyz_deformation_min, xyz_deformation_max) for _ in range(num_points)]

    for i in range(num_points):
        radius = smooth_radii[i]
        point = normalized_curve[i, :]
        deformation_matrix = deformation_matrices[i]
        distances = compute_anisotropic_distances(X, Y, Z, point, deformation_matrix)
        distance_field = cp.minimum(distance_field, distances - radius)

    return distance_field <= 0


def create_anisotropic_deformation_matrix(deformation_min, deformation_max):
    """Creates a random rotation and scaling matrix for anisotropic deformation."""
    scale_x = cp.random.uniform(deformation_min[0], deformation_max[0])
    scale_y = cp.random.uniform(deformation_min[1], deformation_max[1])
    scale_z = cp.random.uniform(deformation_min[2], deformation_max[2])
    
    rotation_matrix = create_random_rotation_matrix()
    scale_matrix = cp.diag(cp.array([scale_x, scale_y, scale_z]))
    return rotation_matrix @ scale_matrix


def create_random_rotation_matrix():
    """Generates a random 3D rotation matrix."""
    u = cp.random.randn(3)
    u /= cp.linalg.norm(u)
    theta = cp.random.uniform(-cp.pi / 12, cp.pi / 12)
    c, s = cp.cos(theta), cp.sin(theta)
    C = 1 - c
    R = cp.array([
        [c + u[0]**2 * C, u[0] * u[1] * C - u[2] * s, u[0] * u[2] * C + u[1] * s],
        [u[1] * u[0] * C + u[2] * s, c + u[1]**2 * C, u[1] * u[2] * C - u[0] * s],
        [u[2] * u[0] * C - u[1] * s, u[2] * u[1] * C + u[0] * s, c + u[2]**2 * C]
    ])
    return R


def compute_anisotropic_distances(X, Y, Z, point, deformation_matrix):
    """Computes distances from grid points to a curve point with anisotropic deformation."""
    centered_points = cp.stack((X - point[0], Y - point[1], Z - point[2]), axis=-1)
    deformed_points = centered_points @ deformation_matrix.T
    return cp.sqrt(cp.sum(deformed_points**2, axis=-1))


def postprocess_mask(mask, sigma, scale_factor):
    """
    Applies smoothing and resolution changes to the mask to create a rougher appearance.
    """
    smoothed_mask = gaussian_filter(cp.asnumpy(mask).astype(float), sigma=sigma)
    thresholded_mask = (smoothed_mask > 0.5).astype(float)
    
    low_res_mask = zoom(thresholded_mask, zoom=scale_factor, order=0)  # Downsample
    rough_mask = zoom(low_res_mask, zoom=1/scale_factor, order=0)   # Upsample
    
    return rough_mask > 0.5


def process_mat_file(mat_filename, args):
    """
    Processes a single .mat file: loads a curve, generates a complex 3D mask, and saves it.
    """
    input_path = os.path.join(args.input_folder, mat_filename)
    base_name = os.path.splitext(mat_filename)[0]
    output_path = os.path.join(args.output_folder, f'{base_name}.mat')

    if os.path.exists(output_path):
        return f"Skipped (exists): {mat_filename}"

    try:
        data = scipy.io.loadmat(input_path)
        if 'smoothedCurve' not in data:
            return f"Failed (no 'smoothedCurve'): {mat_filename}"
        
        curve = cp.array(data['smoothedCurve'])
        
        if args.split_curve:
            curve = split_curve_at_center(curve, args.split_threshold)

        mask = create_tubular_mask(
            curve,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
            grid_size=tuple(args.grid_size),
            max_radius_change=args.max_radius_change,
            xyz_deformation_min=tuple(args.deformation_min),
            xyz_deformation_max=tuple(args.deformation_max)
        )

        if args.num_dents > 0:
            mask = add_dents_and_bumps(
                mask,
                num_dents=args.num_dents,
                dent_radius_range=tuple(args.dent_radius),
                bump_radius_range=tuple(args.bump_radius),
                grid_size=tuple(args.grid_size)
            )
        
        final_mask = postprocess_mask(mask, sigma=args.sigma, scale_factor=args.scale_factor)
        
        scipy.io.savemat(output_path, {'mask': cp.asnumpy(final_mask)})
        return f"Processed: {mat_filename}"

    except Exception as e:
        return f"Failed ({e}): {mat_filename}"


def main():
    parser = argparse.ArgumentParser(description="Generate 3D masks from curve data.")
    parser.add_argument('--input_folder', type=str, required=True, help='Folder containing input .mat files.')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save the generated mask files.')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use.')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of parallel processes to use.')
    
    # Curve processing parameters
    parser.add_argument('--split_curve', action='store_true', help='Enable splitting the curve at the center.')
    parser.add_argument('--split_threshold', type=float, nargs=2, default=[1.0, 2.0], help='Min and max distance for curve splitting.')

    # Mask generation parameters
    parser.add_argument('--grid_size', type=int, nargs=3, default=[128, 128, 128], help='Dimensions of the mask grid.')
    parser.add_argument('--min_radius', type=float, default=4.0, help='Minimum radius of the tube.')
    parser.add_argument('--max_radius', type=float, default=5.0, help='Maximum radius of the tube.')
    parser.add_argument('--max_radius_change', type=float, default=2.0, help='Max change in radius between adjacent points.')
    parser.add_argument('--deformation_min', type=float, nargs=3, default=[1.0, 0.6, 1.0], help='Min anisotropic deformation scales (x,y,z).')
    parser.add_argument('--deformation_max', type=float, nargs=3, default=[1.2, 0.8, 1.2], help='Max anisotropic deformation scales (x,y,z).')

    # Dents and bumps parameters
    parser.add_argument('--num_dents', type=int, default=5, help='Number of dents and bumps to add.')
    parser.add_argument('--dent_radius', type=float, nargs=2, default=[2.0, 4.0], help='Min and max radius for dents.')
    parser.add_argument('--bump_radius', type=float, nargs=2, default=[3.0, 5.0], help='Min and max radius for bumps.')

    # Post-processing parameters
    parser.add_argument('--sigma', type=float, default=2.0, help='Sigma for Gaussian filter.')
    parser.add_argument('--scale_factor', type=float, default=0.5, help='Scale factor for down/upsampling.')

    args = parser.parse_args()

    set_gpu_device(args.gpu_id)
    os.makedirs(args.output_folder, exist_ok=True)

    mat_files = [f for f in os.listdir(args.input_folder) if f.endswith('.mat')]

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_mat_file, f, args): f for f in mat_files}
        
        progress_bar = tqdm(as_completed(futures), total=len(mat_files), desc="Processing files")
        for future in progress_bar:
            result = future.result()
            progress_bar.set_postfix_str(result)

    print("\nAll files processed.")

if __name__ == '__main__':
    main()


# python e:\phdplat\Code\Tree-diffuison-update\tools\create_mask_from_curve.py --input_folder /home/data/liangzhichao/Data/skeleton_point5/ --output_folder /mnt/gemlab_data_2/User_database/liangzhichao/simulated_mask_data_9/ --gpu_id 0 --num_workers 8 --split_curve