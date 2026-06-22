import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
from SOL_GNN_NEW import QJGNN
import argparse
from argparse import RawTextHelpFormatter
import warnings
from glob import glob

warnings.filterwarnings('ignore')


def augment_graph(labels, edge_index):
    N = len(labels)

    # Step 1: 调整原始边索引（后移1位）
    new_edge_index = edge_index + 1  # 新增1个全局节点在开头

    # Step 2: 创建全局连接（索引0）
    global_node_idx = 0
    all_nodes = torch.arange(1, N + 1)  # 超节点不与自身相连

    # 创建双向连接（包含自环）
    src_global = torch.cat([
        all_nodes,  # 所有节点 -> 全局节点
        torch.full_like(all_nodes, global_node_idx)  # 全局节点 -> 所有节点
    ])
    dst_global = torch.cat([
        torch.full_like(all_nodes, global_node_idx),  # 全局节点 <- 所有节点
        all_nodes  # 全局节点 -> 所有节点
    ])

    # 合并所有边
    global_edge = torch.stack([src_global, dst_global])

    return torch.cat([new_edge_index, global_edge], dim=-1)


amino_acid_properties = {
    0: [1.28, 1.00, 6.11, 0.42, 0.23, 0.62, -0.50, 27.50, 8.10, 0.046, 1.181, 0.007187],
    1: [1.77, 2.43, 6.35, 0.17, 0.41, 0.29, -1.00, 44.60, 5.50, 0.128, 1.461, -0.03661],
    2: [1.60, 2.78, 2.95, 0.25, 0.20, -0.90, 3.00, 40.00, 13.00, 0.105, 1.587, -0.02382],
    3: [1.56, 3.78, 3.09, 0.42, 0.21, -0.74, 3.00, 62.00, 12.30, 0.151, 1.862, 0.006802],
    4: [2.94, 5.89, 5.67, 0.30, 0.38, 1.19, -2.50, 115.50, 5.20, 0.29, 2.228, 0.037552],
    5: [0.00, 0.00, 6.07, 0.13, 0.15, 0.48, 0.00, 0.00, 9.00, 0.00, 0.881, 0.179052],
    6: [2.99, 4.66, 7.69, 0.27, 0.30, -0.40, -0.50, 79.00, 10.40, 0.23, 2.025, -0.01069],
    7: [4.19, 4.00, 6.04, 0.30, 0.45, 1.38, -1.80, 93.50, 5.20, 0.186, 1.81, 0.021631],
    8: [1.89, 4.77, 9.99, 0.32, 0.27, -1.50, 3.00, 100.00, 11.30, 0.219, 2.258, 0.017708],
    9: [2.59, 4.00, 6.04, 0.39, 0.31, 1.06, -1.80, 93.50, 4.90, 0.186, 1.931, 0.051672],
    10: [2.35, 4.43, 5.71, 0.38, 0.32, 0.64, -1.30, 94.10, 5.70, 0.221, 2.034, 0.002683],
    11: [1.60, 2.95, 6.52, 0.21, 0.22, -0.78, 2.00, 58.70, 11.60, 0.134, 1.655, 0.005392],
    12: [2.67, 2.72, 6.80, 0.13, 0.34, 0.12, 0.00, 41.90, 8.00, 0.131, 1.468, 0.23953],
    13: [1.56, 3.95, 5.65, 0.36, 0.25, -0.85, 0.20, 80.70, 10.50, 0.18, 1.932, 0.049211],
    14: [2.34, 6.13, 10.74, 0.36, 0.25, -2.53, 3.00, 105.00, 10.50, 0.291, 2.56, 0.043587],
    15: [1.31, 1.60, 5.70, 0.20, 0.28, -0.18, 0.30, 29.30, 9.20, 0.062, 1.298, 0.004627],
    16: [3.03, 2.60, 5.60, 0.21, 0.36, -0.05, -0.40, 51.30, 8.60, 0.108, 1.525, 0.003352],
    17: [3.67, 3.00, 6.02, 0.27, 0.49, 1.08, -1.50, 71.50, 5.90, 0.14, 1.645, 0.057004],
    18: [3.21, 8.08, 5.94, 0.32, 0.42, 0.81, -3.40, 145.50, 5.40, 0.409, 2.663, 0.037977],
    19: [2.94, 6.47, 5.66, 0.25, 0.41, 0.26, -2.30, 117.30, 6.20, 0.298, 2.368, 0.023599]
}


class HDF5Dataset(Dataset):
    def __init__(self, file_path, surface_path, node_path, dataset_ids=None):
        self.file_path = file_path
        self.node_path = node_path
        self.hdf5_file = h5py.File(file_path, 'r')
        self.hdf5_file_surface = h5py.File(surface_path, 'r')
        self.all_keys = list(self.hdf5_file.keys())

        # train_ids是一个包含训练样本的ID的列表
        # 如果train_ids被提供，那么只包含ID在train_ids中的样本
        if dataset_ids is not None:
            self.keys = [key for key in self.all_keys if key in dataset_ids]
        else:
            self.keys = self.all_keys

    def __len__(self):
        # 返回数据集中的样本个数
        return len(self.keys)

    def __getitem__(self, index):
        # 根据索引获取hdf5文件中的数据
        group_key = self.keys[index]
        group = self.hdf5_file[group_key]
        group_surface = self.hdf5_file_surface[group_key]

        # 将数据从hdf5提取出来，并转换成tensor
        # node
        every_node_path = f"{self.node_path}/{group_key}.pt"
        seq_fea = torch.load(every_node_path)
        sasa = torch.from_numpy(np.array(group['sasa'])).float()
        ss = torch.from_numpy(np.array(group['second_structure'])).float()
        seq = torch.from_numpy(np.array(group['seq'])).float()
        plddt = torch.from_numpy(np.array(group['plddt'])).float()
        surface_aa = np.array(group_surface['connected_ca_indices'])
        node_physical = torch.tensor([amino_acid_properties[int(index)] for index in seq])
        # onehot
        num_amino_acids = 20
        seq = seq.long()
        one_hot = torch.zeros(len(seq), num_amino_acids)
        one_hot[torch.arange(len(seq)), seq] = 1
        # if at surface
        surface_aa_label = np.zeros(len(seq), dtype=int)
        # 将表面点标记为 1
        surface_aa_label[surface_aa] = 1
        surface_aa_label = torch.from_numpy(surface_aa_label).float()
        sasa = sasa.unsqueeze(-1)  # 转换为 [n, 1]
        plddt = plddt.unsqueeze(-1)  # 转换为 [n, 1]
        ss = ss.unsqueeze(-1)  # 转换为 [n, 1]
        surface_aa_label = surface_aa_label.unsqueeze(-1)  # 转换为 [n, 1]
        structure_node_feature = torch.cat([one_hot, node_physical, sasa, plddt, ss, surface_aa_label], dim=-1)  # 20+12+1+1+1+1=36
        surface_node_feature = torch.from_numpy(np.array(group_surface['surface_node_features'])).float()

        # edge
        structure_edge_feature = torch.from_numpy(np.array(group['edge_fea'])).float()
        surface_edge_feature = torch.from_numpy(np.array(group_surface['surface_edge_features'])).float()
        inter_edge_feature = torch.from_numpy(np.array(group_surface['inter_features'])).float()

        # index
        edge_idx_structure = torch.from_numpy(np.array(group['edge_index'])).long()
        edge_idx_surface = torch.from_numpy(np.array(group_surface['edge_index_surface'])).long()
        aa_local_label = torch.from_numpy(np.array(group_surface['aa_local_label'])).long()
        edge_idx_inter = torch.from_numpy(np.array(group_surface['edge_index_inter'])).long()
        edge_idx_structure_with_global = augment_graph(aa_local_label, edge_idx_structure)

        # label 1_sol, 0_insol
        solubility = torch.from_numpy(np.array(group['solubility'])).float()

        # 将数据和标签包装为dict返回
        sample = {
            'group_key': group_key,
            'seq_feature': seq_fea,
            'sasa': sasa,
            'plddt': plddt,
            'structure_node_feature': structure_node_feature,
            'surface_node_feature': surface_node_feature,
            'structure_edge_feature': structure_edge_feature,
            'surface_edge_feature': surface_edge_feature,
            'inter_edge_feature': inter_edge_feature,
            'edge_idx_structure_with_global': edge_idx_structure_with_global,
            'edge_idx_structure': edge_idx_structure,
            'edge_idx_surface': edge_idx_surface,
            'aa_local_label': aa_local_label,
            'edge_idx_inter': edge_idx_inter,
            'solubility': solubility,
        }

        return sample

    def close(self):
        # 确保在数据集不再使用时关闭hdf5文件
        self.hdf5_file.close()


def get_names(train_txt, valid_txt, test_txt):
    train_name = []
    valid_name = []
    test_name = []

    with open(train_txt, 'r') as f:
        for line in f.readlines():
            name = line.strip()[:-2]
            train_name.append(name)

    with open(valid_txt, 'r') as f:
        for line in f.readlines():
            name = line.strip()[:-2]
            valid_name.append(name)

    with open(test_txt, 'r') as f:
        for line in f.readlines():
            name = line.strip()[:-2]
            test_name.append(name)
    return train_name, valid_name, test_name


def get_name(txt):
    name_list = []

    with open(txt, 'r') as f:
        for line in f.readlines():
            name = line.strip()[:-2]
            name_list.append(name)

    return name_list


def get_name_(txt):
    name_list = []

    with open(txt, 'r') as f:
        for line in f.readlines():
            name = '_'.join(line.split('_')[:2])
            name_list.append(name)

    return name_list


def get_name__(txt):
    name_list = []

    with open(txt, 'r') as f:
        for line in f.readlines():
            name = line.strip()
            name_list.append(name)

    return name_list


des = "开始训练"
parser = argparse.ArgumentParser(description=des, formatter_class=RawTextHelpFormatter)
parser.add_argument('-d_seq', default=2560, type=int,
                    help="dimension of sequence embedding")
parser.add_argument('-d_structure_n', default=36, type=int,
                    help="dimension of node features in structure graph")
parser.add_argument('-d_surface_n', default=7, type=int,
                    help="dimension of node features in surface graph")
parser.add_argument('-d_structure_e', default=504, type=int,
                    help="dimension of edge features in structure graph")
parser.add_argument('-d_emb', default=256, type=int,
                    help="dimension of embedding")
parser.add_argument('-n_heads', default=4, type=int,
                    help="number of attention heads")
parser.add_argument('-drop_out', default=0.5, type=float,
                    help="dropout rate")
parser.add_argument('-num_structure_layer', default=2, type=int,
                    help="number of layers in structure graph network")
parser.add_argument('-num_seq_layer', default=1, type=int,
                    help="number of layers in sequence network")
parser.add_argument('-num_fusion_layer', default=2, type=int,
                    help="number of layers in fusion network")
parser.add_argument('-queue_size', default=256, type=int,
                    help="queue size for contrastive learning")
parser.add_argument('-global_cls', default=1, type=float,
                    help="weight for global contrastive loss")
parser.add_argument('-local_cls', default=1, type=float,
                    help="weight for local contrastive loss")
parser.add_argument('-lr', default=1e-4, type=float,
                    help="learning rate")

# Training and hardware configuration
parser.add_argument('-gpu_num', default=1, type=int,
                    help="number of GPUs to use")
parser.add_argument('-patience', default=50, type=int,
                    help="patience for early stopping")
parser.add_argument('-ck_point', default='finetune', type=str,
                    help="which checkpoint to use")


parser.set_defaults(dryrun=False)
inputarg = parser.parse_args()

# 使用
test3_edge_path = '../test/whole_edge_feature.hdf5'
test3_surface_path = '../test/all_structure_feature_test.hdf5'
node_path = '../test/esm_feature/'
test3_txt = '../test/test.txt'

test3_name = get_name(test3_txt)

test3_dataset = HDF5Dataset(test3_edge_path, test3_surface_path, node_path, dataset_ids=test3_name)
test3_dataloader = DataLoader(test3_dataset, batch_size=1, shuffle=False)


earlystop = EarlyStopping(monitor='val_auroc',
                          patience=inputarg.patience,
                          mode='max',
                          verbose=True)

lr_monitor = LearningRateMonitor()

checkpoint_callback = ModelCheckpoint(
    dirpath='./checkpoints/',  # 检查点保存的路径
    save_top_k=2,              # 只保留最优的一个检查点（根据监控的指标）
    verbose=True,              # 在检查点被保存时输出一条消息
    monitor='val_auroc',       # 需要监视的指标
    mode='max',                # "min"表示越小越好，"max"表示越大越好
    filename='{epoch}-{val_auroc:.3f}'
)

logger = CSVLogger("./", name='log')
trainer = pl.Trainer(devices=inputarg.gpu_num,
                     precision=32,
                     accelerator='gpu',
                     max_epochs=1000,
                     logger=logger,
                     # limit_train_batches=100,
                     log_every_n_steps=100,
                     check_val_every_n_epoch=1,
                     callbacks=[earlystop, checkpoint_callback],
                     strategy="ddp_find_unused_parameters_true")

model = QJGNN(
    d_seq=inputarg.d_seq,
    d_structure_n=inputarg.d_structure_n,
    d_surface_n=inputarg.d_surface_n,
    d_structure_e=inputarg.d_structure_e,
    d_emb=inputarg.d_emb,
    n_heads=inputarg.n_heads,
    drop_out=inputarg.drop_out,
    num_structure_layer=inputarg.num_structure_layer,
    num_seq_layer=inputarg.num_seq_layer,
    num_fusion_layer=inputarg.num_fusion_layer,
    queue_size=inputarg.queue_size,
    global_cls=inputarg.global_cls,
    local_cls=inputarg.local_cls,
    lr=inputarg.lr,
)


checkpoint_list = [
    f'../checkpoints/{inputarg.ck_point}.ckpt',
]


for checkpoint_path in checkpoint_list:

    print(f"\nTesting checkpoint: {checkpoint_path}")

    try:
        # 加载检查点
        checkpoint = torch.load(checkpoint_path)

        # 恢复模型状态
        model.load_state_dict(checkpoint['state_dict'])

        # 进行测试
        print(f"Test results for {checkpoint_path}: ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓")
        test_results = trainer.test(
            model,
            dataloaders=[test3_dataloader]
        )

    except Exception as e:
        print(f"Error testing checkpoint {checkpoint_path}: {str(e)}")
        continue
