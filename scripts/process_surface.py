import numpy as np
from plyfile import PlyData
import h5py
from scipy.spatial import cKDTree
import torch
from collections import defaultdict, Counter
import os


def rbf(dist, d_min=0, d_max=20, d_count=16):
    d_mu = torch.linspace(d_min, d_max, d_count).reshape(1, 1, 1, -1).to(dist.device)
    d_sigma = (d_max - d_min) / d_count
    dist = dist[:, :, :, None]
    return torch.exp(-((dist - d_mu) ** 2) / (2 * d_sigma ** 2))


# ----------------- PLY文件加载模块 -----------------
def load_point_cloud(ply_path):
    """加载PLY文件并提取坐标和属性"""
    ply_data = PlyData.read(ply_path)
    vertex = ply_data['vertex'].data

    points = np.vstack([vertex['x'],
                        vertex['y'],
                        vertex['z']]).T
    attributes = np.vstack([vertex['charge'],
                            vertex['hbond'],
                            vertex['hphob'],
                            vertex['nx'],
                            vertex['ny'],
                            vertex['nz']]).T
    return points.astype(np.float32), attributes.astype(np.float32)


# ----------------- 八叉树压缩核心模块 -----------------
class OctreeNode:
    def __init__(self, center, size, depth=0):
        self.center = center
        self.size = size
        self.depth = depth
        self.points = []
        self.children = []


class Octree:
    def __init__(self, points, attributes, max_depth=6, max_points_per_node=4):  # 参数调整
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)

        root_size = np.max(max_coords - min_coords)
        root_center = (max_coords + min_coords) / 2
        self.root = OctreeNode(root_center, root_size)
        self.max_depth = max_depth
        self.max_points = max_points_per_node
        self.data = np.hstack((points, attributes))

        self._build_tree(self.root, self.data, 0)

    def _build_tree(self, node, points_data, depth):
        if depth >= self.max_depth or len(points_data) <= self.max_points:
            node.points = points_data
            return

        node.children = [None] * 8
        new_size = node.size / 2
        offsets = np.array([
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        ]) * new_size

        for i in range(8):
            child_center = node.center + offsets[i]
            child_node = OctreeNode(child_center, new_size, depth + 1)
            mask = self._points_in_node(points_data[:, :3], child_center, new_size)
            child_data = points_data[mask]

            if len(child_data) > 0:
                self._build_tree(child_node, child_data, depth + 1)
                node.children[i] = child_node

    def _points_in_node(self, points, center, size):
        half_size = size / 2
        return np.all(
            (points >= (center - half_size)) &
            (points <= (center + half_size)),
            axis=1
        )

    def compress(self):
        compressed = []
        self._collect_points(self.root, compressed)
        return np.array(compressed)

    def _collect_points(self, node, result):
        if not node.children:
            if len(node.points) > 0:
                avg_coords = np.mean(node.points[:, :3], axis=0)
                avg_attributes = np.mean(node.points[:, 3:], axis=0)
                result.append(np.hstack((avg_coords, avg_attributes)))
        else:
            for child in node.children:
                if child is not None:
                    self._collect_points(child, result)


# ----------------- 主处理流程 -----------------
def octree_compress(points, attributes):
    if points.shape[0] > 30000:
        many = 5
    else:
        many = 1
    # 按比例确定目标点数
    target_points = len(points) // 20

    # 固定一个较大的深度以确保充分分割
    max_depth = 6
    # 初始设定一个参数，后续迭代调整
    max_points_per_node = 20

    # 迭代调整 max_points_per_node 直到接近目标值
    for num in range(many):  # 限制迭代次数防止死循环
        octree = Octree(points, attributes,
                        max_depth=max_depth,
                        max_points_per_node=max_points_per_node)
        compressed = octree.compress()
        current_count = len(compressed)

        # 如果压缩后的点数比较接近目标，则退出循环
        if abs(current_count - target_points) / target_points < 0.2:
            break
        # 如果点数太多，说明分割太细（节点中点数太少），可以适当增大 max_points_per_node
        if current_count > target_points:
            max_points_per_node += 10
        # 如果点数太少，说明节点内聚集了过多点，尝试降低 max_points_per_node
        else:
            max_points_per_node = max(1, max_points_per_node - 1)

    return compressed[:, :3], compressed[:, 3:]


def read_hdf5(hdf5):
    batch = {}
    pdbid = []
    # 打开HDF文件
    with h5py.File(hdf5, 'r') as f:
        num = 0
        for group_name in f:
            num += 1
            pdbid.append(group_name)
            group = f[group_name]

            # 遍历组中的所有数据集
            for dataset_name in group:
                dataset = group[dataset_name]
                if dataset_name not in batch:
                    batch[dataset_name] = []
                if dataset_name == 'seq':
                    batch[dataset_name].append(dataset[()].decode('utf-8'))
                else:
                    batch[dataset_name].append(dataset[()])

    # print(batch)
    return batch, pdbid


def compute_edges(surface_coords, ca_coords, k=10):
    # 确保输入为numpy数组
    surface_coords = np.asarray(surface_coords)
    ca_coords = np.asarray(ca_coords)

    n_surface = surface_coords.shape[0]

    # 计算surface点的k-NN图
    surface_tree = cKDTree(surface_coords)
    knn_dist, knn_indices = surface_tree.query(surface_coords, k=k + 1)  # 包含自己

    # 去掉自己并展平
    sources = np.repeat(np.arange(n_surface), k)
    targets = knn_indices[:, 1:].flatten()  # 排除第一个点（自己）
    edge_index_surface = np.stack([sources, targets], axis=0)
    edge_features_surface = knn_dist[:, 1:].flatten()

    # 计算surface到最近CA的连接
    ca_tree = cKDTree(ca_coords)
    ca_dist, ca_indices = ca_tree.query(surface_coords, k=1)
    edge_index_inter = np.stack([np.arange(n_surface), ca_indices], axis=0)
    edge_features_all = ca_dist

    # 收集所有连接的CA索引
    connected_ca_indices = np.unique(ca_indices)

    # 表面特征
    tensor_dist = torch.from_numpy(edge_features_surface).float().reshape(edge_features_surface.shape[0], 1, 1)
    surface_features = rbf(tensor_dist).squeeze().numpy()

    # 表面与氨基酸特征
    tensor_dist = torch.from_numpy(edge_features_all).float().reshape(edge_features_all.shape[0], 1, 1)
    inter_features = rbf(tensor_dist).squeeze().numpy()

    return edge_index_surface, surface_features, edge_index_inter, inter_features, connected_ca_indices


def compute_local_axes(points):
    """
    利用 PCA 计算局部坐标系（主轴），返回排序后的特征向量矩阵（每列一个轴）
    """
    # 计算质心
    center = np.mean(points, axis=0)
    # 计算协方差矩阵
    cov = np.cov(points - center, rowvar=False)
    # 求特征值和特征向量（注意 np.linalg.eigh 返回的特征值是升序排列）
    eigvals, eigvecs = np.linalg.eigh(cov)
    # 将特征向量按照特征值从大到小排序
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    return eigvecs, center

def assign_sectors(normals, local_axes):
    """
    对每个法向量，根据在局部坐标系下的投影，将其划分到六个方向之一，并以整数形式返回标签：
        对于第1主轴：正 -> 0 ('front'), 负 -> 1 ('back')
        对于第2主轴：正 -> 2 ('right'), 负 -> 3 ('left')
        对于第3主轴：正 -> 4 ('up'),   负 -> 5 ('down')
    """
    labels = []
    for n in normals:
        # 在局部坐标系下投影：系数为每个主轴的内积
        proj = np.dot(local_axes.T, n)
        # 找到投影中绝对值最大的那一维
        idx = np.argmax(np.abs(proj))
        if idx == 0:
            label = 0 if proj[0] > 0 else 1
        elif idx == 1:
            label = 2 if proj[1] > 0 else 3
        else:
            label = 4 if proj[2] > 0 else 5
        labels.append(label)
    return labels


hd5f_path = '../test/whole.hdf5'
error_path = '../test/error_surface.txt'
fout = h5py.File(f'../test/all_structure_feature_test.hdf5', "w")
batch, names = read_hdf5(hd5f_path)
error_list = []
all_num = len(names)
num = 0

for i, j in zip(batch['five_coor'], names):
    num += 1
    try:
        ca = i[:, 1, :]
        surface_file = f'../test/surface/{j}_A.ply'

        points, attributes = load_point_cloud(surface_file)
        lenth = len(points)
        points, attributes = octree_compress(points, attributes)
        normals = attributes[:, 3:]  # 法向量

        print(f"surface points. before/after: {lenth}/{len(attributes)} compress rate: {len(points) / lenth:.1%} {num}/{all_num}  {j}")

        edge_index_surface, surface_features, edge_index_inter, inter_features, connected_ca_indices = compute_edges(surface_coords=points, ca_coords=ca, k=5)

        # 1. 计算局部坐标系（旋转等变）
        local_axes, center = compute_local_axes(points)

        # 2. 根据每个点在局部坐标系下的投影划分区域
        sector_labels = np.array(assign_sectors(points, local_axes))

        # 3. 把得到旋转等变的法向量
        normals = normals @ local_axes

        # 4. 计算表面点到质心的距离
        distances = np.linalg.norm(points - center, axis=1)

        # 提取表面点和CA的索引
        surface_indices = edge_index_inter[0, :]
        ca_indices = edge_index_inter[1, :]

        n_ca = ca.shape[0]
        # -------------------------------------

        # 1. 建立CA到表面点的映射
        ca_to_surface = defaultdict(list)
        for surf_idx, ca_idx in zip(surface_indices, ca_indices):
            ca_to_surface[ca_idx].append(int(surf_idx))  # 确保索引为整数

        # 2. 统计每个CA的标签众数
        ca_labels = {}
        for ca_idx, surf_indices in ca_to_surface.items():
            labels = sector_labels[surf_indices]  # 直接通过NumPy索引批量获取标签
            most_common = Counter(labels).most_common(1)
            ca_labels[ca_idx] = most_common[0][0] if most_common else 6

        # 3. 初始化全7数组（长度由用户提供或自动推断）
        ca_label_array = np.full(n_ca, 6, dtype=int)  # 使用用户提供的n_ca
        # ca_label_array = np.full(n_ca_auto, 6, dtype=int)  # 若自动推断

        # 4. 填充标签
        for ca_idx, label in ca_labels.items():
            ca_label_array[ca_idx] = label

        # 表面点：电性，氢键，疏水性，局部法向量*3，距离质心的距离
        distances = distances.reshape(-1, 1)
        surface_node_features = np.concatenate([attributes[:, :3], normals, distances], axis=1)

        group = fout.create_group(j)
        group.create_dataset("edge_index_surface", data=np.array(edge_index_surface), compression="lzf")
        group.create_dataset("surface_edge_features", data=np.array(surface_features), compression="lzf")
        group.create_dataset("surface_node_features", data=np.array(surface_node_features), compression="lzf")
        group.create_dataset("edge_index_inter", data=np.array(edge_index_inter), compression="lzf")
        group.create_dataset("inter_features", data=np.array(inter_features), compression="lzf")
        group.create_dataset("connected_ca_indices", data=np.array(connected_ca_indices), compression="lzf")
        group.create_dataset("surface_local_label", data=np.array(sector_labels), compression="lzf")
        group.create_dataset('aa_local_label', data=np.array(ca_label_array), compression="lzf")

    except Exception as e:
        print(f'{j} error！{e}')
        error_list.append([j, e])


if len(error_list) > 0:
    with open(error_path, 'w') as f:
        for i in error_list:
            f.write(f'{i[0]} {i[1]}\n')

