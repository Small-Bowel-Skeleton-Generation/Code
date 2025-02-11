import os
import numpy as np
import scipy.io
import scipy.ndimage
import scipy.interpolate
import scipy.spatial
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm  # 用于显示进度条
import time  # 用于计时
import cupy as cp  # 替换 numpy 为 cupy
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.ndimage import gaussian_filter, zoom
from skimage.transform import resize

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def split_curve_at_center(curve, threshold_distance):
    """
    curve - N×3 矩阵，表示 3D 曲线
    threshold_distance - 列表，表示查找的距离阈值范围 [min, max]
    curve_output - 分割并修改后的曲线
    """
    center_point = cp.mean(curve, axis=0)  # 计算曲线的中心点
    distances_to_center = cp.linalg.norm(curve - center_point, axis=1)
    min_idx = int(cp.argmin(distances_to_center))  # 将索引转换为整数
    points_to_remove = [min_idx]
    threshold_distance_val = threshold_distance[0] + cp.random.rand() * (threshold_distance[1] - threshold_distance[0])
    for i in range(min_idx - 1, -1, -1):
        if cp.linalg.norm(curve[i + 1, :] - curve[i, :]) < threshold_distance_val:
            points_to_remove.append(i)
        else:
            break  # 停止移除

    for i in range(min_idx + 1, len(curve)):
        if cp.linalg.norm(curve[i - 1, :] - curve[i, :]) < threshold_distance_val:
            points_to_remove.append(i)
        else:
            break  # 停止移除

    if points_to_remove[0] > 0:
        curve1 = curve[:points_to_remove[0], :]
    else:
        curve1 = cp.empty((0, 3))

    if points_to_remove[-1] < len(curve) - 1:
        curve2 = curve[points_to_remove[-1] + 1:, :]
    else:
        curve2 = cp.empty((0, 3))
    curve_output = cp.vstack((curve1, curve2))
    return curve_output


def add_random_dents_and_bumps(tubular_mask, num_dents, dent_radius_range, bump_radius_range, grid_size):
    verts, faces, _, _ = measure.marching_cubes(tubular_mask.get(), 0.5)
    verts = cp.array(verts)
    mask_with_dents_and_bumps = tubular_mask.copy()
    X, Y, Z = cp.meshgrid(cp.arange(grid_size[0]), cp.arange(grid_size[1]), cp.arange(grid_size[2]), indexing="ij")

    for _ in range(num_dents):
        center = verts[cp.random.randint(len(verts))]
        if cp.random.rand() > 0.5:
            radius = cp.random.uniform(dent_radius_range[0], dent_radius_range[1])
            mask_with_dents_and_bumps = apply_spherical_shape(mask_with_dents_and_bumps, X, Y, Z, center, -radius)
        else:
            radius = cp.random.uniform(bump_radius_range[0], bump_radius_range[1])
            mask_with_dents_and_bumps = apply_spherical_shape(mask_with_dents_and_bumps, X, Y, Z, center, radius)

    return mask_with_dents_and_bumps


def apply_spherical_shape(mask, X, Y, Z, center, radius):
    distances = cp.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
    if radius < 0:
        mask[distances <= abs(radius)] = False
    else:
        mask[distances <= radius] = True
    return mask


# def generate_flattened_tubular_mask(curve, min_radius, max_radius, grid_size, max_change, xyz_deformation_min,
#                                     xyz_deformation_max):
#     num_interp_points = 512
#     t = cp.linspace(0, 1, curve.shape[0])
#     t_interp = cp.linspace(0, 1, num_interp_points)
#
#     # 曲线插值
#     interpolator = scipy.interpolate.interp1d(cp.asnumpy(t), cp.asnumpy(curve), kind='linear', axis=0)
#     interpolated_curve = cp.array(interpolator(cp.asnumpy(t_interp)))
#
#     # 数据缩放到 [-1, 1]
#     min_vals = cp.min(interpolated_curve, axis=0)
#     max_vals = cp.max(interpolated_curve, axis=0)
#     center = (max_vals + min_vals) / 2
#     range_vals = (max_vals - min_vals)
#     normalized_curve = (interpolated_curve - center) / (cp.max(range_vals) / 2) * 0.8
#
#     # 定义目标网格（128x128x128）中的坐标
#     grid_points = cp.linspace(-1, 1, grid_size[0])
#     X, Y, Z = cp.meshgrid(grid_points, grid_points, grid_points, indexing='ij')
#
#     # 初始化距离场
#     distance_field = cp.inf * cp.ones(grid_size)
#
#     # 调整半径，根据曲线的缩放比例缩放原始半径
#     adjusted_min_radius = min_radius / (cp.max(range_vals) / 2)
#     adjusted_max_radius = max_radius / (cp.max(range_vals) / 2)
#
#     # 计算每个点的随机半径和形变矩阵
#     num_points = normalized_curve.shape[0]
#     random_radii = cp.random.uniform(adjusted_min_radius, adjusted_max_radius, num_points)
#     deformation_matrices = [generate_anisotropic_deformation_matrix(xyz_deformation_min, xyz_deformation_max) for _ in
#                             range(num_points)]
#
#     # 计算距离场并生成 mask
#     for i in range(num_points):
#         radius = random_radii[i]
#         current_point = normalized_curve[i, :]
#         deformation_matrix = deformation_matrices[i]
#
#         # 计算每个网格点到当前曲线点的距离
#         distances = compute_anisotropic_distances(X, Y, Z, current_point, deformation_matrix)
#
#         # 更新距离场，将每个点的最短距离和半径相比较
#         distance_field = cp.minimum(distance_field, distances - radius)
#
#     # 使用调整后的半径生成 mask
#     tubular_mask = distance_field <= 0
#     return tubular_mask


def generate_flattened_tubular_mask(curve, min_radius, max_radius, grid_size=(128, 128, 128), max_change=2, xyz_deformation_min=(1.0, 0.5, 1.0), xyz_deformation_max=(1.2, 0.7, 1.2)):
    num_interp_points = 512
    t = cp.linspace(0, 1, curve.shape[0])
    t_interp = cp.linspace(0, 1, num_interp_points)

    # 曲线插值
    interpolator = scipy.interpolate.interp1d(cp.asnumpy(t), cp.asnumpy(curve), kind='linear', axis=0)
    interpolated_curve = cp.array(interpolator(cp.asnumpy(t_interp)))

    # 数据缩放到 [-1, 1]
    min_vals = cp.min(interpolated_curve, axis=0)
    max_vals = cp.max(interpolated_curve, axis=0)
    center = (max_vals + min_vals) / 2
    range_vals = (max_vals - min_vals)
    normalized_curve = (interpolated_curve - center) / (cp.max(range_vals) / 2) * 0.8

    # 定义目标网格（128x128x128）中的坐标
    grid_points = cp.linspace(-1, 1, grid_size[0])
    X, Y, Z = cp.meshgrid(grid_points, grid_points, grid_points, indexing='ij')

    # 初始化距离场
    distance_field = cp.inf * cp.ones(grid_size)

    # 调整半径，根据曲线的缩放比例缩放原始半径
    adjusted_min_radius = min_radius / (cp.max(range_vals) / 2)
    adjusted_max_radius = max_radius / (cp.max(range_vals) / 2)
    adjusted_max_change = max_change / (cp.max(range_vals) / 2)

    # 计算每个点的目标半径
    num_points = normalized_curve.shape[0]
    target_radii = cp.random.uniform(adjusted_min_radius, adjusted_max_radius, num_points)

    # 生成平滑半径，保证相邻点半径变化不超过 max_change
    smooth_radii = cp.zeros(num_points)
    smooth_radii[0] = target_radii[0] + (cp.random.rand() - 0.5) * adjusted_max_change  # 第一个半径

    for i in range(1, num_points):
        # 计算相邻点的目标半径
        target_radius = smooth_radii[i-1]

        # 计算可能的新半径
        potential_radius = smooth_radii[i-1] + (cp.random.rand() - 0.5) * 2 * adjusted_max_change  # 相邻半径变化限制

        # 确保半径在 min_radius 和 max_radius 之间
        potential_radius = cp.clip(potential_radius, adjusted_min_radius, adjusted_max_radius)

        # 如果超过目标半径，调整至最大允许范围
        if cp.abs(potential_radius - target_radius) > adjusted_max_change:
            potential_radius = target_radius + cp.sign(potential_radius - target_radius) * adjusted_max_change

        # 更新平滑半径
        smooth_radii[i] = potential_radius

    # 计算每个点的随机半径和形变矩阵
    deformation_matrices = [generate_anisotropic_deformation_matrix(xyz_deformation_min, xyz_deformation_max) for _ in range(num_points)]

    # 计算距离场并生成 mask
    for i in range(num_points):
        radius = smooth_radii[i]
        current_point = normalized_curve[i, :]
        deformation_matrix = deformation_matrices[i]

        # 计算每个网格点到当前曲线点的距离
        distances = compute_anisotropic_distances(X, Y, Z, current_point, deformation_matrix)

        # 更新距离场，将每个点的最短距离和半径相比较
        distance_field = cp.minimum(distance_field, distances - radius)

    # 使用调整后的半径生成 mask
    tubular_mask = distance_field <= 0
    return tubular_mask


def generate_smooth_random_radii(num_points, min_radius, max_radius, max_change, curve, surface_center):
    smooth_radii = cp.zeros(num_points)

    # 计算每个点到中心的距离
    distances_to_center = cp.linalg.norm(curve - surface_center, axis=1)

    # 归一化到 [0, 1] 范围
    normalized_distances = (distances_to_center - cp.min(distances_to_center)) / (
            cp.max(distances_to_center) - cp.min(distances_to_center)
    )

    # 计算目标半径
    target_radii = min_radius + (max_radius - min_radius) * normalized_distances

    # 初始化第一个点的半径
    smooth_radii[0] = target_radii[0] + (cp.random.rand() - 0.5) * max_change

    # 平滑生成剩余点的半径
    for i in range(1, num_points):
        potential_radius = smooth_radii[i - 1] + (cp.random.rand() - 0.5) * 2 * max_change
        potential_radius = cp.clip(potential_radius, min_radius, max_radius)

        # 调整到目标半径附近
        if cp.abs(potential_radius - target_radii[i]) > max_change:
            potential_radius = target_radii[i] + cp.sign(potential_radius - target_radii[i]) * max_change

        smooth_radii[i] = potential_radius

    return smooth_radii


def generate_anisotropic_deformation_matrix(deformation_min, deformation_max):
    scale_x = deformation_min[0] + (deformation_max[0] - deformation_min[0]) * cp.random.rand()
    scale_y = deformation_min[1] + (deformation_max[1] - deformation_min[1]) * cp.random.rand()
    scale_z = deformation_min[2] + (deformation_max[2] - deformation_min[2]) * cp.random.rand()

    rotation_matrix = random_rotation_matrix()
    scale_matrix = cp.diag(cp.array([scale_x, scale_y, scale_z]))
    return rotation_matrix @ scale_matrix


def random_rotation_matrix():
    u = cp.random.randn(3)
    u /= cp.linalg.norm(u)
    theta = (2 * cp.random.rand() - 1) * cp.pi / 12
    c, s = cp.cos(theta), cp.sin(theta)
    C = 1 - c
    R = cp.array([
        [c + u[0]**2 * C, u[0] * u[1] * C - u[2] * s, u[0] * u[2] * C + u[1] * s],
        [u[1] * u[0] * C + u[2] * s, c + u[1]**2 * C, u[1] * u[2] * C - u[0] * s],
        [u[2] * u[0] * C - u[1] * s, u[2] * u[1] * C + u[0] * s, c + u[2]**2 * C]
    ])
    return R


def compute_anisotropic_distances(X, Y, Z, current_point, deformation_matrix):
    centered_points = cp.stack((X - current_point[0], Y - current_point[1], Z - current_point[2]), axis=-1)
    deformed_points = centered_points @ deformation_matrix.T
    distances = cp.sqrt(cp.sum(deformed_points**2, axis=-1))
    return distances


def process_mask(mask, sigma, scale_factor=0.5):
    smoothed_mask = gaussian_filter(cp.asnumpy(mask).astype(float), sigma=sigma)
    # 将平滑后的掩码阈值化为布尔掩码
    mask_show = smoothed_mask > 0.5
    # 确保输入是布尔类型
    mask_show = mask_show.astype(float)
    # 降低分辨率：使用最近邻插值
    low_res_mask = zoom(mask_show, zoom=scale_factor, order=0)  # order=0 表示最近邻插值
    # 将低分辨率掩码放大到原始大小：使用最近邻插值
    rough_mask = zoom(low_res_mask, zoom=1/scale_factor, order=0)  # 放大比例是缩小的倒数
    # 将放大后的掩码阈值化为布尔类型
    final_mask = rough_mask > 0.5
    return final_mask


def process_file(k):
    """
    处理单个文件的函数，生成 mask 并保存。
    """
    mat_file_path = os.path.join(input_folder, mat_files[k])
    # 确保文件名和输出路径一致
    file_base_name = os.path.splitext(mat_files[k])[0]
    mat_output_path = os.path.join(output_folder, f'{file_base_name}.mat')
    # 检查是否已经存在同名文件
    if os.path.exists(mat_output_path):
        print(f"File already exists: {mat_output_path}. Skipping.")
        return

    data = scipy.io.loadmat(mat_file_path)

    # 确保 'smoothedCurve' 存在
    if 'smoothedCurve' in data:
        curve = cp.array(data['smoothedCurve'])  # 将曲线转为 cupy 矩阵
    else:
        raise ValueError(f'The variable "smoothedCurve" not found in file {mat_files[k]}.')

    # 断开曲线
    threshold_distance = [5, 8]
    curve = split_curve_at_center(curve, threshold_distance)

    # 生成管状 mask
    final_mask = generate_flattened_tubular_mask(curve, min_radius=6, max_radius=10, grid_size=(128, 128, 128), max_change=2, xyz_deformation_min=(1.0, 0.6, 1.0), xyz_deformation_max=(1.2, 0.8, 1.2))
    # visualize_mask(mask, 'Tubular Mask', slice_index=64)
    # 添加突起和凹陷
    # mask_with_dents_and_bumps = add_random_dents_and_bumps(mask, num_dents=10, dent_radius_range=(3, 5), bump_radius_range=(4, 6), grid_size=(128, 128, 128))
    # final_mask = process_mask(mask_with_dents_and_bumps, sigma=2, scale_factor=0.5)
    # 获取文件名并去掉末尾的 .mat 扩展名
    file_base_name = os.path.splitext(mat_files[k])[0]

    # 保存生成的 mask 为 .mat 文件
    mat_filename = os.path.join(output_folder, f'{file_base_name}.mat')
    scipy.io.savemat(mat_filename, {'mask': cp.asnumpy(final_mask)})  # 将 cupy 矩阵转为 numpy 保存


# 定义输入和输出文件夹
input_folder = r'/home/data/liangzhichao/Data/skeleton_point3/'
output_folder = r'/mnt/gemlab_data_2/User_database/liangzhichao/simulated_mask_data_7/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取所有 .mat 文件列表
mat_files = [f for f in os.listdir(input_folder) if f.endswith('.mat')]

# 设置并行处理
num_workers = 10  # 手动设置为合适的进程数量
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = {executor.submit(process_file, k): k for k in range(len(mat_files))}

    # 使用 tqdm 显示进度条
    for future in tqdm(as_completed(futures), total=len(mat_files), desc="Processing files", unit="file"):
        future.result()  # 确保处理过程中捕获异常
print("All files processed.")




# # 取消多进程
# input_folder = r'/home/data/liangzhichao/Data/skeleton_point2/'
# output_folder = r'/mnt/gemlab_data_2/User_database/liangzhichao/simulated_mask_data_3/'
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# # 获取所有 .mat 文件列表
# mat_files = [f for f in os.listdir(input_folder) if f.endswith('.mat')]
#
# # 单进程按顺序处理每个文件
# for k in tqdm(range(len(mat_files)), desc="Processing files", unit="file"):
#     try:
#         process_file(k)  # 调用文件处理函数
#     except Exception as e:
#         print(f"Error processing file {mat_files[k]}: {e}")
#
# print("All files processed.")
