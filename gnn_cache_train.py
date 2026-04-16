import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv
import numpy as np
import pickle
import argparse

# -------------------- 模型定义 --------------------
class CacheGNN(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim=64, num_content_categories=12):
        super().__init__()
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # 输出层：为每个内容类别输出一个Q值（或缓存概率）
        self.fc = nn.Linear(hidden_dim, num_content_categories)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        q_values = self.fc(x)          # shape: [num_nodes, num_content_categories]
        return q_values

# -------------------- 数据加载 --------------------
def load_training_data(data_path):
    """
    加载预处理好的训练数据。
    返回列表，每个元素为 (node_features, edge_index, content_category_labels, reward)
    - node_features: [N, F] 每行一个节点特征
    - edge_index: [2, E] 邻接矩阵
    - content_category_labels: [N] 每个节点实际缓存的内容类别（或动作）
    - reward: [N] 每个节点获得的奖励（如命中次数）
    """
    with open(data_path, 'rb') as f:
        samples = pickle.load(f)
    return samples

# -------------------- 训练循环 --------------------
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='训练数据文件路径')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CacheGNN(node_feat_dim=10).to(device)   # 需要根据实际特征维度调整
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()   # 回归任务：预测Q值

    dataset = load_training_data(args.data)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(args.epochs):
        total_loss = 0
        for data in loader:
            data = data.to(device)
            pred_q = model(data.x, data.edge_index)   # [sum_N, num_classes]
            # 假设我们有监督信号：每个节点应缓存的内容类别对应的目标Q值
            # 这里简化：用实际动作对应的目标Q值（如1表示成功缓存命中，0未命中）
            target = data.y   # [sum_N, num_classes]
            loss = loss_fn(pred_q, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch}, Loss: {total_loss/len(loader):.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'gnn_cache_model.pth')
    print('模型已保存至 gnn_cache_model.pth')

if __name__ == '__main__':
    train()