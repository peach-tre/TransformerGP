import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpTraitPredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_size=12, num_heads=4, num_layers=2, dropout_rate=0.5):
        super(ExpTraitPredictionModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.bn = nn.BatchNorm1d(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.bn(x)
        x = x.transpose(1, 2)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class SnpTraitPredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_size=16, num_heads=2, num_layers=1):
        super(SnpTraitPredictionModel, self).__init__()

        # 初始降维 - 使用线性层降维
        self.dim_reduction = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        # 特征选择 - 使用大卷积核
        self.feature_selection = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden_size, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
        )
        # 特征提取 - 使用小卷积核
        self.feature_extraction = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),
        )
        # Transformer模块
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size * 2,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.3,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(hidden_size * 2, output_dim)
        )

    def forward(self, x):
        # 初始降维
        x = self.dim_reduction(x)
        # 添加通道维度
        x = x.unsqueeze(1)
        # 特征选择
        x = self.feature_selection(x)
        identity = x  # 保存用于残差连接
        # 特征提取
        x = self.feature_extraction(x)
        # 残差连接（需要调整维度）
        identity = nn.functional.adaptive_avg_pool1d(identity, x.size(2))
        identity = nn.functional.pad(identity, (0, 0, 0, x.size(1) - identity.size(1)))
        x = x + identity
        # Transformer处理
        x = x.transpose(1, 2)
        x = self.transformer(x)
        # 全局池化
        x = x.mean(dim=1)
        # 分类
        x = self.classifier(x)

        return x


class CombinedTraitPredictionModel(nn.Module):
    def __init__(self, exp_input_dim, snp_input_dim, hidden_dim, output_dim,
                 num_heads=2, num_layers=1, dropout_rate=0.5):
        super(CombinedTraitPredictionModel, self).__init__()

        # EXP 分支
        self.exp_linear = nn.Linear(exp_input_dim, hidden_dim)
        self.exp_relu = nn.ReLU()

        # SNP 分支
        self.snp_linear = nn.Linear(snp_input_dim, hidden_dim)
        self.snp_relu = nn.ReLU()

        # 交互机制：门控机制
        self.gate_linear = nn.Linear(hidden_dim * 2, hidden_dim)

        # 卷积层 + 池化层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=2)
        self.bn = nn.BatchNorm1d(hidden_dim)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Dropout 和输出层
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, exp_x, snp_x):
        # EXP 分支
        exp_x = self.exp_linear(exp_x)  # [batch_size, hidden_size]
        exp_x = self.exp_relu(exp_x)

        # SNP 分支
        snp_x = self.snp_linear(snp_x)  # [batch_size, hidden_size]
        snp_x = self.snp_relu(snp_x)

        # 交互机制：门控机制
        combined_x = torch.cat((exp_x, snp_x), dim=1)  # [batch_size, hidden_size * 2]
        gate = torch.sigmoid(self.gate_linear(combined_x))  # [batch_size, hidden_size]
        combined_x = gate * exp_x + (1 - gate) * snp_x  # 加权融合 [batch_size, hidden_size]

        # 卷积层 + 池化层处理
        combined_x = combined_x.unsqueeze(1)  # [batch_size, 1, hidden_size]
        combined_x = self.conv1(combined_x)  # [batch_size, hidden_size, seq_len]
        combined_x = self.conv2(combined_x)  # [batch_size, hidden_size, seq_len]
        combined_x = self.pool(combined_x)  # [batch_size, hidden_size, seq_len // 2]
        combined_x = self.bn(combined_x)  # 批归一化

        # 调整维度顺序
        combined_x = combined_x.transpose(1, 2)  # [batch_size, seq_len // 2, hidden_size]

        # Transformer 编码器
        combined_x = self.transformer(combined_x)  # [batch_size, seq_len // 2, hidden_size]
        combined_x = combined_x.mean(dim=1)  # 全局平均池化 [batch_size, hidden_size]

        # Dropout 和输出层
        combined_x = self.dropout(combined_x)
        output = self.output_linear(combined_x)  # [batch_size, output_dim]

        return output