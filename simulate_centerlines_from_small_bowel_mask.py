import os
import numpy as np
import scipy
from scipy.io import loadmat
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor


def read_and_resize_mat(mat_filepath):
    # 加载 .mat 文件
    data = loadmat(mat_filepath)
    mask3D = data['maskHull']
    return mask3D


def extract_inner_space(available_space, min_distance):
    # 计算每个 True 点到最近 False 点的距离
    distance_map = distance_transform_edt(available_space)
    # 提取距离大于等于 min_distance 的空间
    inner_space = distance_map >= min_distance
    return inner_space


def find_corners_and_sample(inner_space, max_distance):
    # 找到标记为 True 的点
    points = np.argwhere(inner_space)
    if points.size == 0:
        raise ValueError("No available points in the inner space.")

    # 找到内空间的左上角和右下角
    min_corner = np.min(points, axis=0)
    max_corner = np.max(points, axis=0)

    # 在左上角附近采样点
    left_top_point = points[np.argmax(points[:, 1] + points[:, 2])]
    right_bottom_point = points[np.argmin(points[:, 1] + points[:, 2])]

    def sample_near_point(base_point):
        while True:
            sampled_point = base_point + (np.random.rand(3) * 2 - 1) * max_distance
            sampled_point = np.clip(sampled_point, min_corner, max_corner)
            sampled_point = sampled_point.astype(int)
            if inner_space[tuple(sampled_point)]:
                return sampled_point

    # 生成起点和终点
    start_point = sample_near_point(left_top_point)
    end_point = sample_near_point(right_bottom_point)
    return start_point, end_point


def update_available_space(inner_space, point, min_self_distance):
    """
    更新可用空间：排除距离给定点 `point` 小于 `min_self_distance` 的区域
    """
    point_int = np.round(point).astype(int)
    if not (0 <= point_int[0] < inner_space.shape[0] and
            0 <= point_int[1] < inner_space.shape[1] and
            0 <= point_int[2] < inner_space.shape[2]):
        raise ValueError(f"Point {point_int} is out of bounds for the mask dimensions.")

    temp_space = np.zeros_like(inner_space, dtype=bool)
    temp_space[tuple(point_int)] = True
    distance_map = distance_transform_edt(~temp_space)
    updated_space = (distance_map >= min_self_distance) & inner_space
    return updated_space


def visualize_temp_space(temp_space, base_point):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    true_points = np.argwhere(temp_space)
    base_point = np.array(base_point).reshape(1, -1)
    # 显示可用空间的点（True 的点）
    ax.scatter(true_points[:, 0], true_points[:, 1], true_points[:, 2], c='lightgrey', s=1, alpha=0.5,
               label='Available Space (True)')
    # 显示 base_point
    ax.scatter(base_point[:, 0], base_point[:, 1], base_point[:, 2], c='red', s=50, label='Base Point')
    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("Visualization of temp_space with Base Point")
    plt.show()

def check_x_line_above(curve, prev_point, new_point):
    """
    检查新点与上一点的连线在 X 轴方向的上方是否存在曲线的连线。
    """
    x_line_above = False
    for i in range(len(curve) - 1):
        curve_start = curve[i]
        curve_end = curve[i + 1]
        # 检查 curve 的连线是否在 prev_point 和 new_point 的 X 范围内
        if prev_point[0] < curve_start[0] < new_point[0] or prev_point[0] < curve_end[0] < new_point[0]:
            x_line_above = True
            break
    return x_line_above

def sample_point_in_distance_range(available_space, base_point, curve, min_dist=7, max_dist=14, min_distance_to_boundary=6):
    """
    在距离 `base_point` 的规定范围内采样一个点。
    优先检查新点与上一点的连线在 X 轴方向是否存在曲线连线。
    """
    temp_space = np.zeros_like(available_space, dtype=bool)
    base_point_int = np.round(base_point).astype(int)  # 将浮点数坐标转换为整数坐标
    temp_space[tuple(base_point_int)] = True  # 设置为 True，表示起始点

    # 计算距离并筛选在范围内的候选点
    distance_map = distance_transform_edt(~temp_space)
    within_range = (distance_map >= min_dist) & (distance_map <= max_dist) & available_space
    candidates = np.argwhere(within_range)

    if len(candidates) == 0:
        print(f"No points found in distance range [{min_dist}, {max_dist}] from point {base_point}")
        return None

    # 随机选择候选点
    selected_index = np.random.choice(len(candidates))
    candidate = candidates[selected_index] + np.random.uniform(-0.5, 0.5, size=3)

    # 确保 `curve` 是 NumPy 数组
    curve = np.array(curve)

    # 检查新点与上一点的连线是否在 X 轴方向上方存在曲线
    if len(curve) > 1:
        prev_point = curve[-1]
        x_line_above = check_x_line_above(curve, prev_point, candidate)

        if not x_line_above:
            # 在候选点中找到与新点的 y 和 z 坐标相同，且 x 坐标最大的点
            candidate_yz = np.round(candidate[1:3])  # 获取新点的 y, z 坐标
            yz_matches = candidates[
                (np.round(candidates[:, 1]) == candidate_yz[0]) & (np.round(candidates[:, 2]) == candidate_yz[1])
                ]
            if len(yz_matches) > 0:
                # 选择 x 坐标最大的点
                max_x_candidate = yz_matches[np.argmax(yz_matches[:, 0])]
                return max_x_candidate + np.random.uniform(-0.5, 0.5, size=3)
            else:
                print(
                    f"No candidates match the y and z coordinates of the candidate {candidate}. Using the original candidate.")
                return candidate

    return candidate

def plot_curve_3d(curve_points, inner_space=None, start_point=None, end_point=None):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制可用空间（如果提供了）
    if inner_space is not None:
        points = np.argwhere(inner_space)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='lightgrey', s=1, alpha=0.1, label='Inner Space')

    # 绘制曲线点
    curve_points = np.array(curve_points)
    ax.plot(curve_points[:, 0], curve_points[:, 1], curve_points[:, 2], color='blue', marker='o', label='Curve')

    # 绘制起点和终点（如果提供了）
    if start_point is not None:
        ax.scatter(*start_point, c='red', s=50, label='Start Point')
    if end_point is not None:
        ax.scatter(*end_point, c='green', s=50, label='End Point')

    # 设置轴标签和图例
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("Generated 3D Curve")
    plt.show()

def rank_of_curve(curve):
    """
    计算曲线的秩
    """
    if len(curve) < 2:
        return 0
    differences = curve[1:] - curve[0]
    rank = np.linalg.matrix_rank(differences)
    return rank

def gauss_linking_integral(curve):
    """
    检查曲线是否打结（高斯链接积分）
    """
    N = len(curve)
    linking_number = 0
    for i in range(N - 1):
        for j in range(i + 1, N - 1):
            r1, r2 = curve[i], curve[i + 1]
            r3, r4 = curve[j], curve[j + 1]
            dr1 = r2 - r1
            dr2 = r4 - r3
            r13 = r1 - r3
            r24 = r2 - r4
            r14 = r1 - r4
            r23 = r2 - r3
            norm_r13 = np.linalg.norm(r13)
            norm_r14 = np.linalg.norm(r14)
            norm_r23 = np.linalg.norm(r23)
            norm_r24 = np.linalg.norm(r24)
            # 避免零距离
            if norm_r13 * norm_r14 * norm_r23 * norm_r24 < 1e-6:
                continue
            # 计算高斯链接积分的微分
            dLk = np.dot(np.cross(dr1, dr2), r13) / (norm_r13 * norm_r14 * norm_r23 * norm_r24)
            linking_number += dLk
    # 检查链接数是否显著偏离 0
    return abs(linking_number) > 1e-3

def generate_curve(start_point, end_point, initial_available_space, max_length, min_self_distance, max_steps):
    curve_points = [start_point]  # 曲线点集合
    current_point = start_point  # 当前点
    current_length = 0  # 当前曲线长度
    available_space = initial_available_space.copy()
    available_space = update_available_space(available_space, end_point, min_self_distance)
    step_count = 0  # 当前步数

    while current_length < max_length:
        curve = np.array(curve_points)
        sampling_space = sample_point_in_distance_range(available_space, current_point, curve, min_dist=7, max_dist=14)
        if sampling_space is None:
            break

        # 采样新点并更新曲线
        new_point = sampling_space
        curve_points.append(new_point)
        available_space = update_available_space(available_space, current_point, min_self_distance)
        step_count += 1
        distance = np.linalg.norm(new_point - current_point)
        current_length += distance
        current_point = new_point

        # 检查步数和长度限制
        if step_count > max_steps or current_length > max_length:
            print("Step or length limit reached. Checking for a direct path to the endpoint.")

            # 尝试寻找直接路径
            if check_direct_path(current_point, end_point, available_space):
                direct_path = find_direct_path(current_point, end_point, available_space)
                curve_points.extend(direct_path)
                print("Direct path to endpoint found and added to the curve.")
                break
            else:
                print("No direct path found. Sampling points towards the endpoint.")

                # 在采样空间中优先采样距离终点最近的点
                while step_count < max_steps + 50 or current_length < max_length + 800:
                    sampling_space = sample_point_in_distance_range(available_space, current_point, curve, min_dist=7, max_dist=14)
                    if sampling_space is None:
                        print("No valid sampling space available.")
                        break

                    # 找到距离终点最近的点
                    distances_to_end = np.linalg.norm(sampling_space - end_point, axis=1)
                    closest_index = np.argmin(distances_to_end)
                    new_point = sampling_space[closest_index]

                    # 更新曲线
                    curve_points.append(new_point)
                    available_space = update_available_space(available_space, current_point, min_self_distance)
                    step_count += 1
                    distance = np.linalg.norm(new_point - current_point)
                    current_length += distance
                    current_point = new_point

                    # 如果当前点接近终点，则结束
                    if np.linalg.norm(end_point - current_point) < 40:
                        curve_points.append(end_point)
                        print("Reached endpoint while sampling.")
                        break
                break

    return np.array(curve_points), current_length


def check_direct_path(current_point, end_point, available_space):
    """
    检查当前点和终点之间是否存在直接连通的路径。
    """
    line_points = interpolate_line(current_point, end_point)
    for point in line_points:
        point_int = np.round(point).astype(int)
        if (
            point_int[0] < 0 or point_int[0] >= available_space.shape[0]
            or point_int[1] < 0 or point_int[1] >= available_space.shape[1]
            or point_int[2] < 0 or point_int[2] >= available_space.shape[2]
            or not available_space[tuple(point_int)]
        ):
            return False
    return True


def find_direct_path(current_point, end_point, available_space):
    """
    找到当前点和终点之间的直接路径。
    返回路径上的点列表。
    """
    line_points = interpolate_line(current_point, end_point)
    path = []
    for point in line_points:
        point_int = np.round(point).astype(int)
        if (
            0 <= point_int[0] < available_space.shape[0]
            and 0 <= point_int[1] < available_space.shape[1]
            and 0 <= point_int[2] < available_space.shape[2]
            and available_space[tuple(point_int)]
        ):
            path.append(point)
        else:
            break
    return path


def interpolate_line(start, end, num_points=100):
    """
    线性插值生成两点之间的路径。
    """
    return np.linspace(start, end, num_points)


# def generate_and_save_curves(mask_path, save_path, num_curves=100, max_length=5000, min_self_distance=15):
#     filenames = [f for f in os.listdir(mask_path) if f.endswith('.mat')]
#     for filename in filenames:
#         file_path = os.path.join(mask_path, filename)
#         available_space = read_and_resize_mat(file_path)
#         inner_space = extract_inner_space(available_space, min_distance=6)
#         file_prefix = filename[:6]
#         volume = np.sum(available_space)
#         max_steps = min(max(270, round(8.82e-5 * volume + 117.65)), 520)
#
#         for curve_index in range(num_curves):
#             save_filename = f"{file_prefix}_{curve_index + 1:05d}.mat"
#             save_filepath = os.path.join(save_path, save_filename)
#             if os.path.exists(save_filepath):
#                 print(f"File {save_filename} already exists. Skipping.")
#                 continue
#
#             valid_curve_generated = False
#             while not valid_curve_generated:
#                 # 生成起点和终点
#                 start_point, end_point = find_corners_and_sample(inner_space, max_distance=25)
#
#                 # 生成曲线
#                 curve_points, final_length = generate_curve(
#                     start_point, end_point, inner_space, max_length, min_self_distance, max_steps
#                 )
#
#                 # 检查曲线是否满足条件
#                 if len(curve_points) >= max_steps or final_length >= max_length:
#                     valid_curve_generated = True
#                     # 保存曲线到文件
#                     scipy.io.savemat(save_filepath, {'curve_points': curve_points})
#                     print(f"Saved curve {curve_index + 1} for {filename} to {save_filepath}.")
#                 else:
#                     print(
#                         f"Curve does not meet requirements (steps: {len(curve_points)}, length: {final_length}). Regenerating...")
#
#             scipy.io.savemat(save_filepath, {'curve_points': curve_points})
#             print(f"Saved curve {curve_index + 1} for {filename} to {save_filepath}.")
#
# # 定义路径
# mask_path = "/mnt/gemlab_data_2/User_database/liangzhichao/QC_maskHull/"
# save_path = "/mnt/gemlab_data_2/User_database/liangzhichao/skeleton_point/"
#
# # 生成并保存曲线
# generate_and_save_curves(mask_path, save_path, num_curves=100, max_length=5000, min_self_distance=15)

def process_file(file_path, save_path, num_curves, max_length, min_self_distance):
    available_space = read_and_resize_mat(file_path)
    inner_space = extract_inner_space(available_space, min_distance=6)
    volume = np.sum(available_space)
    max_steps = min(max(270, round(8.82e-5 * volume + 117.65)), 520)

    file_prefix = os.path.splitext(os.path.basename(file_path))[0][:6]

    for curve_index in range(num_curves):
        save_filename = f"{file_prefix}_{curve_index + 1:05d}.mat"
        save_filepath = os.path.join(save_path, save_filename)
        if os.path.exists(save_filepath):
            print(f"File {save_filename} already exists. Skipping.")
            continue

        valid_curve_generated = False
        while not valid_curve_generated:
            start_point, end_point = find_corners_and_sample(inner_space, max_distance=25)
            curve_points, final_length = generate_curve(
                start_point, end_point, inner_space, max_length, min_self_distance, max_steps
            )

            if len(curve_points) >= max_steps or final_length >= max_length:
                valid_curve_generated = True
                scipy.io.savemat(save_filepath, {'curve_points': curve_points})
                print(f"Saved curve {curve_index + 1} for {os.path.basename(file_path)} to {save_filepath}.")
            else:
                print(f"Curve does not meet requirements (steps: {len(curve_points)}, length: {final_length}). Regenerating...")


def generate_and_save_curves_parallel(mask_path, save_path, num_curves=100, max_length=5000, min_self_distance=15, max_workers=4):
    filenames = [os.path.join(mask_path, f) for f in os.listdir(mask_path) if f.endswith('.mat')]

    os.makedirs(save_path, exist_ok=True)

    # 使用并行进程处理文件
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_file, file_path, save_path, num_curves, max_length, min_self_distance, )
            for file_path in filenames
        ]
        for future in futures:
            future.result()

mask_path = "/mnt/gemlab_data_2/User_database/liangzhichao/QC_maskHull/"
save_path = "/mnt/gemlab_data_2/User_database/liangzhichao/skeleton_point/"
generate_and_save_curves_parallel(mask_path, save_path, num_curves=100, max_length=5000, min_self_distance=10, max_workers=4)