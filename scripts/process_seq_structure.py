from Bio.PDB import *
import numpy as np
import os
import json
import pandas as pd
import torch
import h5py
import esm
import re


aaname = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G',
          'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
          'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
          'TRP': 'W', 'TYR': 'Y'}


# 定义函数计算两个原子之间的距离
def calculate_distance(atom1, atom2):
    return np.linalg.norm(atom1 - atom2)


def extract_info_from_pdb(pdb_file):
    """
    得到序列，序列编号，原子坐标
    """

    pdbid, _ = os.path.basename(pdb_file).split('.')
    info = {}

    # 创建PDB解析器对象
    parser = PDBParser()

    # 解析PDB文件
    structure = parser.get_structure('STRUCTURE', pdb_file)

    # 获取模型列表中的第一个模型（PDB文件通常只有一个模型）
    model = structure[0]

    # 遍历模型中的所有链
    for chain in model:
        # 获取链中的所有残基
        residues = list(chain.get_residues())

        # 遍历残基，提取序列信息
        chain_sequence = ''
        chain_num = []
        atom_coordinates = []
        b_factor_chain = []

        for residue in residues:
            # 获取氨基酸的名称
            amino_acid = residue.get_resname()
            if amino_acid in aaname:
                amino_acid_id = aaname[amino_acid]
            else:
                amino_acid_id = 'X'  # 如果是非标准氨基酸，表示为'X'
            # 获取氨基酸的编号
            residue_number = str(residue.get_id()[1])
            residue_number_2 = str(residue.get_id()[2])
            residue_number = ''.join([residue_number, residue_number_2]).strip()
            # 将氨基酸名称和编号添加到序列列表中
            chain_sequence += amino_acid_id
            chain_num.append(residue_number)
            residue_coor = {}
            b_factor = 0
            for atom in residue:
                # 获取原子名称和坐标
                atom_name = atom.get_name()
                atom_coord = atom.get_coord()
                rounded_list = [round(num, 3) for num in atom_coord.tolist()]

                # residue_coor[atom_name] = atom_coord  # 数据是nparray类型

                residue_coor[atom_name] = rounded_list  # 数据是list
                b_factor = atom.bfactor

            atom_coordinates.append(residue_coor)
            b_factor_chain.append(b_factor)

        # 将链的序列信息添加到总序列列表中
        assert len(b_factor_chain) == len(chain_num)
        info = {'res_name': chain_sequence, 'res_num': chain_num, 'coor': atom_coordinates, 'b_factor': b_factor_chain}
        break

    return pdbid, info


def process_pdb_files(pdb_clean_path, out_path):
    files = os.listdir(pdb_clean_path)
    info_dict = {}
    num = 0
    for file in files:
        num += 1
        print(f'get pdb info {num} / {len(files)}  {file}')
        path = os.path.join(pdb_clean_path, file)
        pdbid, info = extract_info_from_pdb(path)
        info_dict[pdbid] = info

    out_info = os.path.join(out_path, 'pdb_info.json')

    with open(out_info, 'w') as jf:
        json.dump(info_dict, jf)


def mp_nerf_torch(a, b, c, l, theta, chi):  # 通过三个坐标（a、b、c）与旋转矩阵，输出处于另一个平面的d的坐标
    """ Custom Natural extension of Reference Frame.
        Inputs:
        * a: (batch, 3) or (3,). point(s) of the plane, not connected to d
        * b: (batch, 3) or (3,). point(s) of the plane, not connected to d
        * c: (batch, 3) or (3,). point(s) of the plane, connected to d
        * theta: (batch,) or (float).  angle(s) between b-c-d
        * chi: (batch,) or float. dihedral angle(s) between the a-b-c and b-c-d planes
        Outputs: d (batch, 3) or (float). the next point in the sequence, linked to c
    """
    if not ((-np.pi <= theta) * (theta <= np.pi)).all().item():
        raise ValueError(f"theta(s) must be in radians and in [-pi, pi]. theta(s) = {theta}")
    # calc vecs
    ba = b - a
    cb = c - b
    # calc rotation matrix. based on plane normals and normalized
    n_plane = torch.cross(ba, cb, dim=-1)
    n_plane_ = torch.cross(n_plane, cb, dim=-1)
    rotate = torch.stack([cb, n_plane_, n_plane], dim=-1)
    rotate = rotate / (torch.norm(rotate, dim=-2, keepdim=True) + 1e-5)
    # print (rotate.shape)
    # calc proto point, rotate. add (-1 for sidechainnet convention)
    # https://github.com/jonathanking/sidechainnet/issues/14
    d = torch.stack([-torch.cos(theta),
                     torch.sin(theta) * torch.cos(chi),
                     torch.sin(theta) * torch.sin(chi)], dim=-1).unsqueeze(-1)
    # extend base point, set length
    return c + l.unsqueeze(-1) * torch.matmul(rotate, d).squeeze()


def are_receptors_or_ligands(element1, element2, list1, list2):
    return (element1 in list1 and element2 in list1) or (element1 in list2 and element2 in list2)


def build_CB(Ncoor, CAcoor, Ccoor):
    nres = Ncoor.shape[0]
    l = torch.tensor(2.499, dtype=torch.float, device=Ncoor.device).repeat(nres)
    theta = torch.tensor(34.828 / 180.0 * 3.1415926, dtype=torch.float, device=Ncoor.device).repeat(nres)
    chi = torch.tensor(-122.611 / 180.0 * 3.1415926, dtype=torch.float, device=Ncoor.device).repeat(nres)
    cbcoor = mp_nerf_torch(Ncoor, CAcoor, Ccoor, l, theta, chi)
    return cbcoor[0, :]


def rigidFrom3Points_torch(x1, x2, x3):
    """
    通过三个点的坐标计算出了一个旋转矩阵
    """
    v1 = x3 - x2
    v2 = x1 - x2
    e1 = v1 / torch.linalg.norm(v1, dim=-1, keepdim=True)
    u2 = v2 - e1 * torch.sum(e1 * v2, dim=-1, keepdim=True)
    e2 = u2 / torch.linalg.norm(u2, dim=-1, keepdim=True)
    e3 = torch.cross(e1, e2)
    R = torch.stack((e1, e2, e3), axis=1)
    return R


def get_esm_list(data):
    esm_list = []
    num = 0
    length = len(data)
    for pdbid in data:
        print(f'processing esmlist {num}/{length}  {pdbid}')
        seq = data[pdbid]['res_name']
        esm_list.append([pdbid, seq])
        num += 1

    return esm_list


def process_hdf5(data, what):
    fout = h5py.File(f'../{what}/whole.hdf5', "w")
    count = 0
    error_list = []
    length = len(data)
    for pdbid in data:
        print(f'processing edge feature {count}/{length}  {pdbid}')

        resname = ''
        resnum = []
        b_factor = data[pdbid]['b_factor']
        five_coor = []
        whole_coors = []
        five_distance = []
        rotation = []
        edge_index = []
        try:
            chain_num = 0
            seq = data[pdbid]['res_name']
            coors = data[pdbid]['coor']
            for aa in seq:
                resname += aa
                resnum.append(chain_num)
                seq_coor = coors[chain_num]
                whole_coors.append(seq_coor)

                # N 原子 0
                n = seq_coor['N']
                n = torch.tensor(n, dtype=torch.float32)

                # CA 原子 1
                ca = seq_coor["CA"]
                ca = torch.tensor(ca, dtype=torch.float32)

                # C 原子 2
                c = seq_coor['C']
                c = torch.tensor(c, dtype=torch.float32)

                # O 原子 3
                o = seq_coor['O']
                o = torch.tensor(o, dtype=torch.float32)

                # CB 原子 4
                if 'CB' not in seq_coor:
                    cb = build_CB(n, ca, c)
                else:
                    cb = seq_coor['CB']
                    cb = torch.tensor(cb, dtype=torch.float32)

                # 整合五个原子坐标
                fivecoor = torch.stack([n, ca, c, o, cb], dim=0)
                five_coor.append(fivecoor)

                # 得到每个残基的旋转矩阵（旋转等变
                aa_rotation = rigidFrom3Points_torch(n, ca, c)
                rotation.append(aa_rotation)
                chain_num += 1

            for i in resnum:
                for j in resnum:
                    coor1 = five_coor[i][4, :]
                    coor2 = five_coor[j][4, :]
                    dist = calculate_distance(coor1, coor2)

                    # cb 链内10埃 链间12埃
                    if dist < 10 and dist != 0:
                        edge_index.append([i, j])

            for i in edge_index:
                i1 = i[0]
                i2 = i[1]
                coors1 = whole_coors[i1]
                # {"N": [73.227, 31.442, 101.41], "CA": [72.12, 30.453, 101.239], "C": [71.047, 30.634, 102.304], "O": [71.319, 30.52, 103.512]}
                coors2 = whole_coors[i2]

                # 将tensor堆叠成一个nx3的tensor
                tensors1 = [torch.tensor(value) for value in coors1.values()]
                result_tensor1 = torch.stack(tensors1)
                tensors2 = [torch.tensor(value) for value in coors2.values()]
                result_tensor2 = torch.stack(tensors2)

                # 计算成对距离
                edge_dist_feat = torch.cdist(result_tensor1, result_tensor2)

                # 将距离矩阵展平并排序
                sorted_distances, indices = torch.sort(torch.flatten(edge_dist_feat))

                # 获取最小的五个距离
                five_smallest_distances = sorted_distances[:5]

                # 如果需要，可以将这些值转换为Python列表
                five_smallest_distances_list = five_smallest_distances.tolist()
                five_distance.append(five_smallest_distances_list)

            group = fout.create_group(pdbid)

            # 写入数据
            group.create_dataset("resnum", data=np.array(resnum))
            group.create_dataset("plddt", data=np.array(b_factor))
            group.create_dataset("seq", data=resname, dtype=h5py.string_dtype(encoding='utf-8'))
            group.create_dataset("five_coor", data=np.array(five_coor))
            group.create_dataset("five_distance", data=np.array(five_distance))
            group.create_dataset("rotation", data=np.array(rotation))
            group.create_dataset("edge_index", data=np.array(edge_index))

            count += 1

        except Exception as e:
            error_list.append([pdbid, e])
            print(f'{pdbid} wrong')

    return error_list


def pdb_to_stride(pdb_path, stride):
    os.makedirs(stride, exist_ok=True)
    names = os.listdir(pdb_path)
    names = [i[:-4] for i in names]
    num = 0

    for name in names:
        #
        # 处理复合物的stride
        #
        input_file = os.path.join(pdb_path, f'{name}.pdb')
        out_file = os.path.join(stride, f'{name}.stride')
        len_name = len(names)
        command = f"stride {input_file} -f{out_file}"
        os.system(command)
        num += 1
        print(f'getting stride {num}/{len_name}')


amino_acid_dict = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
}


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


def rbf(dist, d_min=0, d_max=20, d_count=16):
    d_mu = torch.linspace(d_min, d_max, d_count).reshape(1, 1, 1, -1).to(dist.device)
    d_sigma = (d_max - d_min) / d_count
    dist = dist[:, :, :, None]

    return torch.exp(-((dist - d_mu) ** 2) / (2 * d_sigma ** 2))


def compute_edge_feat(five_atom_coords, five_dis, edge_idx, r):
    five_atom_coords = torch.from_numpy(five_atom_coords)
    five_dis = torch.from_numpy(five_dis)
    r = torch.from_numpy(r)
    list_whole = edge_idx.tolist()
    edge_idx = torch.tensor(list_whole).to(torch.long).t()

    src_idx, dst_idx = edge_idx[0], edge_idx[1]

    # Compute distance between each pair of 'true' atoms in neighboring residues.
    five_atom_coords_i, five_atom_coords_j = (
        five_atom_coords[src_idx],
        five_atom_coords[dst_idx],
    )

    five_dis = five_dis.unsqueeze(1)

    edge_dist_feat = torch.cdist(five_atom_coords_i, five_atom_coords_j)
    edge_dist_feat = torch.cat((edge_dist_feat, five_dis), dim=1)

    # Add small Noise
    edge_dist_feat = (edge_dist_feat + torch.randn_like(edge_dist_feat) * 0.02).clip(
        min=0.0
    )

    edge_dist_feat = rbf(edge_dist_feat)  # rbf把每个distance扩了16维
    edge_dist_feat = edge_dist_feat.view(len(edge_dist_feat), -1)  # nedge, 30*16
    # print(f'core_dis_feature  {edge_dist_feat.size()}')

    edge_angle_feat, edge_dir_fea = cal_angle_direction_feature(edge_idx, five_atom_coords, r)

    edge_feature = torch.cat([edge_dist_feat, edge_angle_feat, edge_dir_fea], dim=1)

    # torch.set_printoptions(threshold=float('inf'))

    # print(edge_feature.shape)

    return edge_idx, edge_feature


def cal_angle_direction_feature(index, five_coors, r):
    src_idx, dst_idx = index[0], index[1]

    # angle feature
    ri, rj = r[src_idx], r[dst_idx]
    ri_inv = torch.linalg.inv(ri)
    rij = ri_inv @ rj
    edge_angle_feat = rij.view(len(rij), -1)  # nedge, 3*3
    # print(f'angle_feature  {edge_angle_feat.shape}')

    # direction feature
    ca_i = five_coors[src_idx][:, 1:2, :]  # n, 1, 3
    five_j = five_coors[dst_idx]  # n, 5, 3
    vectors = five_j - ca_i  # 计算 ca_i 与 five_j 中每个原子的向量
    dir_feature = vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-6)  # 对每个向量进行归一化，以得到单位向量

    # dir_feature = torch.matmul(ri_inv, dir_feature.permute(0, 2, 1)).permute(0, 2, 1)
    dir_feature = torch.einsum('nij,nmj->nmi', ri_inv, dir_feature)

    edge_dir_fea = dir_feature.reshape(len(dir_feature), -1)  # nedge, 5*3
    # print(f'dir_feature  {edge_dir_fea.shape}')

    return edge_angle_feat, edge_dir_fea


def clean_pdb_filenames_strict(folder_path):
    """
    严格清理PDB文件名，只保留字母、数字和文件扩展名

    参数:
    folder_path (str): 存放PDB文件的文件夹路径
    """
    # 只保留字母、数字和文件扩展名中的点
    pattern = r'[^a-zA-Z0-9]'

    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)

        if os.path.isfile(old_path):
            name, ext = os.path.splitext(filename)

            # 去除所有非字母数字字符
            cleaned_name = re.sub(pattern, '', name)

            new_filename = cleaned_name + ext
            new_path = os.path.join(folder_path, new_filename)

            if new_filename != filename:
                try:
                    os.rename(old_path, new_path)
                    print(f"rename: {filename} -> {new_filename}")
                except OSError as e:
                    print(f"rename error {filename}: {e}")


what = 'test'
root_path = f'../{what}'
pdb_clean = os.path.join(root_path, 'pdb')
print("cleaning file name......")
clean_pdb_filenames_strict(pdb_clean)
process_pdb_files(pdb_clean, root_path)
file_path = f'../{what}/pdb_info.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# 用于计算esm特征
esmlist = get_esm_list(data)
with open(f'../{what}/seq_list_for_esm.json', 'w') as f:
    json.dump(esmlist, f)

# 用于计算边特征
list_result = process_hdf5(data, what)
if len(list_result) > 0:
    with open(f'../{what}/error.txt', 'w') as f:
        for i in list_result[:-1]:
            i = str(i)
            f.write(i + '\n')
        for i in list_result[-1:]:
            i = str(i)
            f.write(i)

pdb_path = f'../{what}/pdb'
stride = f'../{what}/stride'
pdb_to_stride(pdb_path, stride)
hdf5_file_path = f'../{what}/whole.hdf5'
batch, pdbid = read_hdf5(hdf5_file_path)
edge_feature = f'../{what}/whole_edge_feature.hdf5'
fout = h5py.File(edge_feature, "w")
esm_out = f'../{what}/seq_list_for_esm.json'
node_feature = f'../{what}/esm_feature'
os.makedirs(node_feature, exist_ok=True)
solubility_file = f'../{what}/test.txt'

"""
edge feature
"""

count = 0
lenth = len(pdbid)

for i, j, k, m, id, chain_seq, plddt in zip(batch['five_coor'], batch['five_distance'], batch['edge_index'], batch['rotation'], pdbid, batch['seq'], batch['plddt']):

    solubility = np.array([0])

    with open(solubility_file, 'r') as f:
        for line in f.readlines():
            if line.startswith(id):
                solubility = float(line.strip().split('_')[-1])
                solubility = np.array([solubility])
                break



    # 计算seq_array
    seq = np.array([amino_acid_dict.get(aa, 20) for aa in chain_seq])

    # 得到sasa面积
    sasa = []
    sasa_path = os.path.join(f'../{what}/stride', f'{id}.stride')
    with open(sasa_path, 'r') as f:
        start = False
        lines = f.readlines()
        for line in lines:
            if start:
                line_info = line.split(' ')
                line_info = [o for o in line_info if o != '']
                _sasa = float(line_info[-2])
                sasa.append(_sasa)
            if '|---Residue---|    |--Structure--|   |-Phi-|   |-Psi-|  |-Area-|' in line:
                start = True

    count += 1
    print(f"save edge feature （{count} / {lenth}）  {id}")

    edge_idx, edge_fea = compute_edge_feat(i, j, k, m)
    group = fout.create_group(id)

    group.create_dataset("sasa", data=np.array(sasa), compression="lzf")
    group.create_dataset("plddt", data=np.array(plddt), compression="lzf")
    group.create_dataset("edge_fea", data=np.array(edge_fea), compression="lzf")
    group.create_dataset("edge_index", data=np.array(edge_idx), compression="lzf")
    group.create_dataset("solubility", data=np.array(solubility), compression="lzf")
    group.create_dataset("seq", data=np.array(seq), compression="lzf")


"""
node feature
"""

print('loading ESM2-3B')
model_name = "esm2_t36_3B_UR50D"
model_pretrain, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
batch_converter = alphabet.get_batch_converter()
model_pretrain.eval()


def get_embed(datatmp_list, save_path):
    """
    datatmp_list:[[id1, seq1],
                [id2, seq2],
                ...
                [idn, seqn]]  # 一个列表，包含【序列的名称（1a4y_A）, 序列（DILPCVPFSVAKSVKS...LYLGRMFS）】
    """
    len_list = len(datatmp_list)
    num = 0
    for x in datatmp_list:
        batch_labels, batch_strs, batch_tokens = batch_converter([x])
        with torch.no_grad():
            results = model_pretrain(batch_tokens, repr_layers=[36], return_contacts=True)
        token_representations = results["representations"][36]
        for i, (id, seq) in enumerate([x]):
            num += 1
            seq_representation = token_representations[i, 1: len(seq) + 1]  # .mean(0)
            embedding = os.path.join(save_path, f'{id}.pt')
            torch.save(seq_representation, embedding)
            print(f'node_feature {num} / {len_list} {id}')


with open(esm_out, 'r') as f:
    datatmp_list = json.load(f)

get_embed(datatmp_list, node_feature)

all_second_structure = {'Coil': 0, '310Helix': 1, 'Turn': 2, 'Strand': 3, 'Bridge': 4, 'AlphaHelix': 5, 'PiHelix': 6}
ss_num = 0

# 打开现有hdf5文件（追加模式）
with h5py.File(edge_feature, "a") as fout:
    # 获取所有group名称
    group_names = list(fout.keys())
    lenth = len(group_names)

    count = 0
    for id in group_names:
        # 获取对应group
        group = fout[id]

        # 解析ss数据
        ss = []
        ss_path = os.path.join(f'../{what}/stride', f'{id}.stride')

        try:
            with open(ss_path, 'r') as f:
                start = False
                lines = f.readlines()
                for line in lines:
                    if start:
                        line_info = line.split(' ')
                        line_info = [o for o in line_info if o != '']
                        if len(line_info) >= 6:  # 确保有足够的数据列
                            _ss = line_info[-5]
                            if _ss not in all_second_structure:
                                all_second_structure[_ss] = ss_num
                                ss_num += 1
                            ss.append(all_second_structure[_ss])
                    if '|---Residue---|    |--Structure--|   |-Phi-|   |-Psi-|  |-Area-|' in line:
                        start = True

            # 将ss数据写入当前group
            if "second_structure" in group:
                del group["second_structure"]  # 如果已存在则先删除
            group.create_dataset("second_structure", data=np.array(ss), compression="lzf")

        except Exception as e:
            print(f"stride append {id} wrong: {str(e)}")
            continue

        count += 1
        print(f"stride append （{count} / {lenth}）  {id}")