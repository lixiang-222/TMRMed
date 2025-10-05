# 导入必要的PyTorch库
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2Layer(nn.Module):
    def __init__(self, in_features, out_features, heads=1):
        super(GATv2Layer, self).__init__()
        self.heads = heads
        self.out_features = out_features
        
        # 线性变换矩阵
        self.W = nn.Linear(in_features, out_features * heads, bias=False)
        # 注意力机制参数
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(0.2)
        
    def forward(self, x, adj):
        # x: (num_nodes, in_features)
        # adj: (num_nodes, num_nodes)
        
        # 线性变换
        h = self.W(x)  # (num_nodes, out_features * heads)
        h = h.view(-1, self.heads, self.out_features)  # (num_nodes, heads, out_features)
        
        # 计算注意力系数
        h_i = h.unsqueeze(1)  # (num_nodes, 1, heads, out_features)
        h_j = h.unsqueeze(0)  # (1, num_nodes, heads, out_features)
        h_cat = torch.cat([h_i.repeat(1, h.size(0), 1, 1), 
                          h_j.repeat(h.size(0), 1, 1, 1)], dim=-1)  # (num_nodes, num_nodes, heads, 2*out_features)
        
        e = self.leakyrelu(torch.matmul(h_cat, self.a).squeeze(-1))  # (num_nodes, num_nodes, heads)
        
        # 应用邻接矩阵掩码
        e = e.masked_fill(adj.unsqueeze(-1) == 0, float('-inf'))
        
        # 计算注意力权重
        attention = F.softmax(e, dim=1)  # (num_nodes, num_nodes, heads)
        
        # 多头注意力聚合
        h_prime = torch.einsum('ijh,jhf->ihf', attention, h)  # (num_nodes, heads, out_features)
        
        # 合并多头
        if self.heads > 1:
            h_prime = h_prime.mean(dim=1)  # (num_nodes, out_features)
        else:
            h_prime = h_prime.squeeze(1)  # (num_nodes, out_features)
            
        return h_prime

class Ours(nn.Module):
    def __init__(self, embedding_dim, output_size, col_len, dropout_rate=0.5, dataset='Ours', device=torch.device('cpu')):
        super(Ours, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim

        # 为每列的离散值创建对应的嵌入层
        self.embedding_layers_x = nn.ModuleList([
            nn.Embedding(num_embeddings=col, embedding_dim=embedding_dim)
            for col in col_len
        ])

        if dataset == 'Ours':
            self.embedding_layers_y1 = nn.ModuleList([
                nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)
                for _ in range(4)  # y1有4个二分类特征
            ])

            self.embedding_layers_y2 = nn.ModuleList([
                nn.Embedding(num_embeddings=3, embedding_dim=embedding_dim)
            ])  # y2有1个三分类特征

            # y3药物嵌入层 (323种药物)
            self.embedding_y3 = nn.Embedding(num_embeddings=323, embedding_dim=embedding_dim)
        elif dataset == 'tcm_lung':

            self.embedding_layers_y1 = nn.ModuleList([
                nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)
                for _ in range(50)  # y1有50个二分类特征
            ])

            self.embedding_layers_y2 = nn.ModuleList([
                nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)
                for _ in range(61)
            ])  # y2有61个二分类特征

            # y3药物嵌入层 (379种药物)
            self.embedding_y3 = nn.Embedding(num_embeddings=379, embedding_dim=embedding_dim)
        else:
            raise ValueError('Invalid dataset.')

        self.dropout = nn.Dropout(dropout_rate)

        # MLP用于建模残差
        self.residual_x = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.residual_y1 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.residual_y2 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.residual_y3 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # GRU层用于序列建模
        self.gru_x = nn.GRU(embedding_dim, embedding_dim, batch_first=False)
        self.gru_y1 = nn.GRU(embedding_dim, embedding_dim, batch_first=False)
        self.gru_y2 = nn.GRU(embedding_dim, embedding_dim, batch_first=False)
        self.gru_y3 = nn.GRU(embedding_dim, embedding_dim, batch_first=False)
        
        # GATv2 layers
        self.x_y1_gcn = GATv2Layer(embedding_dim, embedding_dim)
        self.y1_y2_gcn = GATv2Layer(embedding_dim, embedding_dim)
        self.y2_y3_gcn = GATv2Layer(embedding_dim, embedding_dim)
        
        # shared encoder & specific encoder
        self.encoder0 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.encoder1 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.encoder3 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # 最终预测层        
        self.fc = nn.Sequential(
            nn.Linear(4*embedding_dim, 2*embedding_dim),
            nn.ReLU(),
            nn.Linear(2*embedding_dim, embedding_dim)
        )
        self.fc1 = nn.Linear(3 * embedding_dim, output_size[0])
        self.fc2 = nn.Linear(4 * embedding_dim, output_size[1])
        self.fc3 = nn.Linear(5 * embedding_dim, output_size[2])
        # self.fc4 = nn.Linear(embedding_dim, output_size[2])

        # decoder
        self.decoder1 = nn.Sequential(
            nn.Linear(3*embedding_dim, 2*embedding_dim),
            nn.ReLU(),
            nn.Linear(2*embedding_dim, embedding_dim)
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(4*embedding_dim, 2*embedding_dim),
            nn.ReLU(),
            nn.Linear(2*embedding_dim, embedding_dim)
        )
        self.decoder3 = nn.Sequential(
            nn.Linear(5*embedding_dim, 2*embedding_dim),
            nn.ReLU(),
            nn.Linear(2*embedding_dim, embedding_dim)
        )
        # self.decoder4 = nn.Sequential(
        #     nn.Linear(5*embedding_dim, 2*embedding_dim),
        #     nn.ReLU(),
        #     nn.Linear(2*embedding_dim, embedding_dim)
        # )

    def forward(self, patient):
        # 初始化序列存储
        x_list, y1_list, y2_list, y3_list = [], [], [], []
        
        # -----#1.特征编码------
        for visit_idx, visit_data in enumerate(patient):
            x, y1, y2, y3 = visit_data
            
            # 转换为张量
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            y1 = torch.tensor(y1, dtype=torch.long, device=self.device)
            y2 = torch.tensor(y2, dtype=torch.long, device=self.device)
            y3_binary = (y3 != 0).astype(int)
            y3 = torch.tensor(y3_binary, dtype=torch.long, device=self.device)

            # 按照您要求的方式处理x、y1、y2的表征
            embedded_x = [self.embedding_layers_x[i](x[i]) for i in range(x.size(0))]
            embedded_x = torch.stack(embedded_x, dim=0)  # (num_features, embedding_dim)
            embedded_x = self.dropout(embedded_x)
            embedded_x = embedded_x.sum(dim=0)  # (embedding_dim,)
            x_list.append(embedded_x)

            # 只处理历史y特征（排除最后一次）
            if visit_idx < len(patient) - 1:
                # 处理y1特征
                embedded_y1 = [self.embedding_layers_y1[i](y1[i]) for i in range(y1.size(0))]
                embedded_y1 = torch.stack(embedded_y1, dim=0)  # (4, embedding_dim)
                embedded_y1 = self.dropout(embedded_y1)
                embedded_y1 = embedded_y1.sum(dim=0)  # (embedding_dim,)
                y1_list.append(embedded_y1)

                # 处理y2特征
                embedded_y2 = [self.embedding_layers_y2[i](y2[i]) for i in range(y2.size(0))]
                embedded_y2 = torch.stack(embedded_y2, dim=0)  # (1, embedding_dim)
                embedded_y2 = self.dropout(embedded_y2)
                embedded_y2 = embedded_y2.sum(dim=0)  # (embedding_dim,)
                y2_list.append(embedded_y2)

                # 处理y3药物特征
                active_drugs = y3.nonzero().squeeze()  # 获取有效药物索引
                if active_drugs.dim() == 0:  # 只有一个药物
                    drug_embeddings = self.embedding_y3(active_drugs.unsqueeze(0))
                else:  # 多个药物
                    drug_embeddings = self.embedding_y3(active_drugs)
                embedded_y3 = drug_embeddings.sum(dim=0)  # (embedding_dim,)
                y3_list.append(self.dropout(embedded_y3))

        # -----#2.图网络建模------
        # 构建全连接图网络
        def build_graph(x_nodes, y1_nodes, y2_nodes, y3_nodes):
            """
            构建x与y之间的全连接图，以及y1、y2、y3之间的连接
            Args:
                x_nodes: x特征节点 (n_x, embedding_dim)
                y1_nodes: y1特征节点 (n_y1, embedding_dim)
                y2_nodes: y2特征节点 (n_y2, embedding_dim)
                y3_nodes: y3特征节点 (n_y3, embedding_dim)
            Returns:
                adj: 邻接矩阵 (n_x + n_y1 + n_y2 + n_y3, ...)
                node_features: 拼接后的节点特征
            """
            n_x = x_nodes.size(0)
            n_y1 = y1_nodes.size(0)
            n_y2 = y2_nodes.size(0)
            n_y3 = y3_nodes.size(0)
            total_nodes = n_x + n_y1 + n_y2 + n_y3
            
            # 初始化邻接矩阵
            adj = torch.zeros(total_nodes, total_nodes, device=self.device)
            
            # x与y1、y2、y3全连接
            adj[:n_x, n_x:n_x+n_y1] = 1  # x -> y1
            adj[:n_x, n_x+n_y1:n_x+n_y1+n_y2] = 1  # x -> y2
            adj[:n_x, n_x+n_y1+n_y2:] = 1  # x -> y3
            
            # y1与y2、y3连接
            adj[n_x:n_x+n_y1, n_x+n_y1:n_x+n_y1+n_y2] = 1  # y1 -> y2
            adj[n_x:n_x+n_y1, n_x+n_y1+n_y2:] = 1  # y1 -> y3
            
            # y2与y3连接
            adj[n_x+n_y1:n_x+n_y1+n_y2, n_x+n_y1+n_y2:] = 1  # y2 -> y3
            
            # 添加反向连接
            adj = adj + adj.t()
            adj = adj.clamp(0, 1)  # 确保值在0-1之间
            
            # 拼接所有节点特征
            node_features = torch.cat([x_nodes, y1_nodes, y2_nodes, y3_nodes], dim=0)
            
            return adj, node_features

        # 重构xlist和ylist：为每个就诊构图（排除最后一次）
        new_x_list, new_y1_list, new_y2_list, new_y3_list = [], [], [], []
        for visit_idx in range(len(patient)-1):  # 排除最后一次就诊
            curr_x = x_list[visit_idx].unsqueeze(0)
            curr_y1 = y1_list[visit_idx].unsqueeze(0) if visit_idx < len(y1_list) else torch.zeros_like(curr_x)
            curr_y2 = y2_list[visit_idx].unsqueeze(0) if visit_idx < len(y2_list) else torch.zeros_like(curr_x)
            curr_y3 = y3_list[visit_idx].unsqueeze(0) if visit_idx < len(y3_list) else torch.zeros_like(curr_x)
            
            # 构建图并传播消息
            adj, all_nodes = build_graph(curr_x, curr_y1, curr_y2, curr_y3)
            gcn_output = self.x_y1_gcn(all_nodes, adj)

            # 更新特征表示，并添加残差连接
            
            # x 节点的原始输入是 curr_x.squeeze(0)，GNN 输出是 gcn_output[0]
            new_x_list.append(curr_x.squeeze(0) + gcn_output[0])
            
            if visit_idx < len(y1_list):
                # y1 节点的原始输入是 curr_y1.squeeze(0)，GNN 输出是 gcn_output[1]
                new_y1_list.append(curr_y1.squeeze(0) + gcn_output[1])
            if visit_idx < len(y2_list):
                # y2 节点的原始输入是 curr_y2.squeeze(0)，GNN 输出是 gcn_output[2]
                new_y2_list.append(curr_y2.squeeze(0) + gcn_output[2])
            if visit_idx < len(y3_list):
                # y3 节点的原始输入是 curr_y3.squeeze(0)，GNN 输出是 gcn_output[3]
                new_y3_list.append(curr_y3.squeeze(0) + gcn_output[3])

        # 保留最后一次就诊的x特征(不保留y特征)
        new_x_list.append(x_list[-1])
        
        x_list = new_x_list
        y1_list = new_y1_list
        y2_list = new_y2_list
        y3_list = new_y3_list

        # -----#3.序列建模------
        # GRU序列处理函数
        def process_sequence(sequence, gru_layer):
            if len(sequence) == 0:
                return torch.zeros(self.embedding_dim, device=self.device)
            sequence_tensor = torch.stack(sequence).unsqueeze(1)  # (seq_len, 1, embedding_dim)
            _, hidden = gru_layer(sequence_tensor)
            return hidden.squeeze()  # (embedding_dim,)
        
        # 获取各特征表示(使用GCN处理后的特征)
        repr_x = process_sequence(x_list, self.gru_x)
        repr_y1_history = process_sequence(y1_list, self.gru_y1)
        repr_y2_history = process_sequence(y2_list, self.gru_y2)
        repr_y3_history = process_sequence(y3_list, self.gru_y3)

        # -----#4.正交建模------
        # shared encoder & specific encoder
        repr_x0 = self.encoder0(repr_x)
        repr_x1 = self.encoder1(repr_x)
        repr_x2 = self.encoder2(repr_x)
        repr_x3 = self.encoder3(repr_x)

        def orthogonality_loss(z1, z2):
            """
            基于Frobenius范数的正交性损失
            输入:
                z1, z2: 形状为(batch_size, feature_dim)的两个表征
            返回:
                loss: 标量损失值
            """
            z1 = z1.unsqueeze(dim=0)
            z2 = z2.unsqueeze(dim=0)

            # 归一化
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
            
            # 计算相关系数矩阵的Frobenius范数
            correlation = torch.mm(z1.T, z2)
            loss = torch.norm(correlation, p='fro') / z1.size(1)
            
            return loss
        
        orth_loss = (orthogonality_loss(repr_x1, repr_x2) + 
                    orthogonality_loss(repr_x1, repr_x3) + 
                    orthogonality_loss(repr_x2, repr_x3)) / 3
        

        # -----#5.预测------
        # 取最后一次的
        _, y1, y2, _ = patient[-1]
        # 转换为张量
        y1 = torch.tensor(y1, dtype=torch.long, device=self.device)
        y2 = torch.tensor(y2, dtype=torch.long, device=self.device)

        #  处理y1特征
        embedded_y1 = [self.embedding_layers_y1[i](y1[i]) for i in range(y1.size(0))]
        embedded_y1 = torch.stack(embedded_y1, dim=0)  # (4, embedding_dim)
        embedded_y1 = self.dropout(embedded_y1)
        embedded_y1 = embedded_y1.sum(dim=0)  # (embedding_dim,)
        y1_list.append(embedded_y1)
        repr_y1 = process_sequence(y1_list, self.gru_y1)

        # 处理y2特征
        embedded_y2 = [self.embedding_layers_y2[i](y2[i]) for i in range(y2.size(0))]
        embedded_y2 = torch.stack(embedded_y2, dim=0)  # (1, embedding_dim)
        embedded_y2 = self.dropout(embedded_y2)
        embedded_y2 = embedded_y2.sum(dim=0)  # (embedding_dim,)
        y2_list.append(embedded_y2)
        repr_y2 = process_sequence(y2_list, self.gru_y2)

        # 预测最后一次y1
        output1 = self.fc1(torch.cat([repr_x0, repr_x1, repr_y1_history])) # 3 * emb 
        # 预测最后一次y2
        output2 = self.fc2(torch.cat([repr_x0, repr_x2, repr_y1, repr_y2_history])) # 4 * emb 
        # 预测最后一次y3
        output3 = self.fc3(torch.cat([repr_x0, repr_x3, repr_y1, repr_y2, repr_y3_history])) # 5 * emb


        # ----- #6.残差对比学习 -----
        if len(patient) > 1:
            y1_last = self.decoder1(torch.cat([repr_x0, repr_x1, repr_y1_history]))
            y2_last = self.decoder2(torch.cat([repr_x0, repr_x2, repr_y1, repr_y2_history]))
            y3_last = self.decoder3(torch.cat([repr_x0, repr_x3, repr_y1, repr_y2, repr_y3_history]))
            
            y1_list.append(y1_last)
            y2_list.append(y2_last)
            y3_list.append(y3_last)

            min_len = len(patient) - 1
            
            # 计算每种类型的残差，shape: (min_len, embedding_dim)
            x_residuals = torch.stack([self.residual_x(x_list[i+1] - x_list[i]) for i in range(min_len)], dim=0)
            y1_residuals  = torch.stack([self.residual_y1(y1_list[i+1] - y1_list[i]) for i in range(min_len)], dim=0)
            y2_residuals = torch.stack([self.residual_y2(y2_list[i+1] - y2_list[i]) for i in range(min_len)], dim=0)
            y3_residuals = torch.stack([self.residual_y3(y3_list[i+1] - y3_list[i]) for i in range(min_len)], dim=0)

            loss_x_y1 = (1 - F.cosine_similarity(x_residuals, y1_residuals, dim=-1)).mean()
            loss_x_y2 = (1 - F.cosine_similarity(x_residuals, y2_residuals, dim=-1)).mean()
            loss_x_y3 = (1 - F.cosine_similarity(x_residuals, y3_residuals, dim=-1)).mean()
            
            loss_y1_y2 = (1 - F.cosine_similarity(y1_residuals, y2_residuals, dim=-1)).mean()
            loss_y1_y3 = (1 - F.cosine_similarity(y1_residuals, y3_residuals, dim=-1)).mean()
            
            loss_y2_y3 = (1 - F.cosine_similarity(y2_residuals, y3_residuals, dim=-1)).mean()

            contrastive_loss = loss_x_y1 + loss_x_y2 + loss_x_y3 + loss_y1_y2 + loss_y1_y3 + loss_y2_y3
        else:
            contrastive_loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
        # 最终预测
        return (
            output1,
            output2, 
            output3,
            orth_loss,
            contrastive_loss
        )