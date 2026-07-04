"""
models.py - 模型定义
包含三种模型：LSTM Seq2Seq、Transformer、CNN-Transformer（改进模型）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. LSTM Seq2Seq with Attention
# ============================================================

class EncoderLSTM(nn.Module):
    """LSTM 编码器（单向）"""
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout,
                            bidirectional=False)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        outputs, (hidden, cell) = self.lstm(x)
        # outputs: (batch, seq_len, hidden_dim)
        # hidden: (num_layers, batch, hidden_dim)
        return outputs, hidden, cell


class DecoderLSTM(nn.Module):
    """LSTM 解码器 with Luong Attention"""
    def __init__(self, output_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM: 输入 = [prev_output, context]
        self.lstm = nn.LSTM(hidden_dim + 1, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)

        # 注意力层
        # encoder_outputs: (batch, src_len, hidden_dim)
        # decoder_hidden: (batch, hidden_dim)
        # attn_input = [encoder_outputs | decoder_hidden] = hidden_dim * 2
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

        # 输出层: [lstm_output | context | decoder_input]
        # lstm_output: hidden_dim, context: hidden_dim, decoder_input: 1
        self.fc = nn.Linear(hidden_dim + hidden_dim + 1, output_dim)

    def forward(self, encoder_outputs, hidden, cell, target_seq=None,
                teacher_forcing_ratio=0.5, output_len=90):
        """
        encoder_outputs: (batch, src_len, hidden_dim)
        hidden: (num_layers, batch, hidden_dim)
        target_seq: (batch, tgt_len)
        """
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)

        # 初始输入: 0
        decoder_input = torch.zeros(batch_size, device=encoder_outputs.device)

        outputs = []

        use_teacher_forcing = (target_seq is not None and
                               torch.rand(1).item() < teacher_forcing_ratio)
        target_len = output_len if target_seq is None else target_seq.size(1)

        for t in range(target_len):
            # Attention: 使用最后一层hidden
            decoder_hidden_last = hidden[-1]  # (batch, hidden_dim)

            # 计算注意力分数
            hidden_expanded = decoder_hidden_last.unsqueeze(1).expand(-1, src_len, -1)
            attn_input = torch.cat([encoder_outputs, hidden_expanded], dim=-1)
            attn_weights = F.softmax(
                self.v(torch.tanh(self.attn(attn_input))).squeeze(-1), dim=-1)
            # attn_weights: (batch, src_len)

            # 上下文向量
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
            # context: (batch, 1, hidden_dim)

            # LSTM输入: [decoder_input; context]
            lstm_input = torch.cat(
                [context, decoder_input.view(batch_size, 1, 1)], dim=-1)
            lstm_output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
            # lstm_output: (batch, 1, hidden_dim)

            # 输出
            fc_input = torch.cat(
                [lstm_output, context,
                 decoder_input.view(batch_size, 1, 1)], dim=-1)
            output = self.fc(fc_input)
            # output: (batch, 1, output_dim)
            outputs.append(output[:, 0, :])  # (batch, output_dim)

            # 下一时刻的输入
            if use_teacher_forcing and t < target_seq.size(1):
                decoder_input = target_seq[:, t]
            else:
                decoder_input = output[:, 0, 0]

        outputs = torch.stack(outputs, dim=1)  # (batch, target_len, output_dim)
        return outputs


class LSTMModel(nn.Module):
    """LSTM Seq2Seq 完整模型"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = EncoderLSTM(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = DecoderLSTM(1, hidden_dim, num_layers, dropout)

    def forward(self, x, target_seq=None, teacher_forcing_ratio=0.5, output_len=90):
        encoder_outputs, hidden, cell = self.encoder(x)
        outputs = self.decoder(encoder_outputs, hidden, cell,
                               target_seq, teacher_forcing_ratio, output_len)
        return outputs.squeeze(-1)  # (batch, output_len)


# ============================================================
# 2. Transformer 模型
# ============================================================

class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer 编码器-解码器模型（轻量版）"""
    def __init__(self, input_dim, d_model=128, nhead=4, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=256, dropout=0.1,
                 output_len=90):
        super().__init__()
        self.d_model = d_model
        self.output_len = output_len

        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers)

        # Transformer 解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers)

        # 输出投影
        self.output_proj = nn.Linear(d_model, 1)

        # 解码器输入embedding（target的positional embedding）
        self.tgt_proj = nn.Linear(1, d_model)
        self.tgt_pos_encoder = PositionalEncoding(d_model, dropout=dropout)

    def forward(self, src, target_seq=None, teacher_forcing_ratio=0.5, output_len=None):
        if output_len is None:
            output_len = self.output_len

        batch_size = src.size(0)
        device = src.device

        # 编码
        src_emb = self.input_proj(src)
        src_emb = self.pos_encoder(src_emb)
        memory = self.transformer_encoder(src_emb)

        # 构建解码器输入序列
        use_tf = (target_seq is not None and
                  torch.rand(1).item() < teacher_forcing_ratio)

        if use_tf:
            # Teacher forcing: 直接用目标序列
            tgt_input = target_seq.unsqueeze(-1)  # (batch, out_len, 1)
            tgt_emb = self.tgt_proj(tgt_input)
            tgt_emb = self.tgt_pos_encoder(tgt_emb)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                output_len, device=device)
            out = self.transformer_decoder(tgt_emb, memory, tgt_mask)
            out = self.output_proj(out)  # (batch, output_len, 1)
            return out.squeeze(-1)
        else:
            # 自回归生成（避免inplace操作）
            outputs = []
            # 初始输入：[start] token (zeros)
            tgt_so_far = torch.zeros(batch_size, 1, 1, device=device)

            for t in range(output_len):
                tgt_emb = self.tgt_proj(tgt_so_far)
                tgt_emb = self.tgt_pos_encoder(tgt_emb)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    t + 1, device=device)
                out = self.transformer_decoder(tgt_emb, memory, tgt_mask)
                pred = self.output_proj(out)  # (batch, t+1, 1)

                # 取最后一步预测
                last_pred = pred[:, -1:, :]  # (batch, 1, 1)
                outputs.append(last_pred[:, 0, 0])  # (batch,)

                # 拼接（非inplace）
                tgt_so_far = torch.cat([tgt_so_far, last_pred], dim=1)

            out = torch.stack(outputs, dim=1)  # (batch, output_len)
            return out


# ============================================================
# 3. CNN-Transformer 改进模型
# ============================================================
# 创新点：使用多尺度1D卷积提取局部时序特征，
# 再通过Transformer捕获全局依赖，MLP解码减少误差累积

class MultiScaleConv(nn.Module):
    """多尺度1D卷积特征提取"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 确保三个卷积的输出通道数之和 = output_dim
        c1 = output_dim // 3
        c2 = output_dim // 3
        c3 = output_dim - c1 - c2  # 余数给第三个
        self.conv3 = nn.Conv1d(input_dim, c1, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(input_dim, c2, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(input_dim, c3, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        c3 = F.relu(self.conv3(x))
        c5 = F.relu(self.conv5(x))
        c7 = F.relu(self.conv7(x))
        out = torch.cat([c3, c5, c7], dim=1)
        out = self.bn(out)
        # (batch, output_dim, seq_len) -> (batch, seq_len, output_dim)
        return out.permute(0, 2, 1)


class CNNTransformerModel(nn.Module):
    """
    改进模型：CNN-Transformer
    - 多尺度CNN提取局部特征
    - Transformer编码器建模长程依赖
    - 直接MLP解码（非自回归，减少误差累积）
    """
    def __init__(self, input_dim, d_model=128, cnn_dim=64, nhead=4,
                 num_encoder_layers=3, dim_feedforward=256, dropout=0.1,
                 output_len=90):
        super().__init__()
        self.output_len = output_len

        # 多尺度CNN特征提取
        self.multi_conv = MultiScaleConv(input_dim, cnn_dim)

        # 投影到d_model维度
        self.conv_proj = nn.Linear(cnn_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers)

        # MLP解码器（直接多步预测）
        # 将编码器输出的全局表示映射到输出序列
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, output_len),
        )

        # 全局池化后的投影
        self.global_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )

    def forward(self, src, target_seq=None, teacher_forcing_ratio=0.5, output_len=None):
        if output_len is None:
            output_len = self.output_len

        # CNN特征提取
        cnn_features = self.multi_conv(src)  # (batch, seq_len, cnn_dim)

        # 投影到d_model
        x = self.conv_proj(cnn_features)  # (batch, seq_len, d_model)

        # 位置编码
        x = self.pos_encoder(x)

        # Transformer编码
        encoded = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        # 全局表示：取最后时间步 + 平均池化
        last_hidden = encoded[:, -1, :]  # (batch, d_model)
        avg_pooled = encoded.mean(dim=1)  # (batch, d_model)
        global_feat = self.global_proj(last_hidden + avg_pooled)

        # MLP解码得到完整输出序列
        if output_len == self.output_len:
            out = self.decoder(global_feat)  # (batch, output_len)
        else:
            # 动态输出长度（需要重建decoder）
            temp_decoder = nn.Sequential(
                nn.Linear(self.decoder[0].in_features, self.decoder[0].out_features),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.decoder[0].out_features, output_len),
            ).to(src.device)
            out = temp_decoder(global_feat)

        return out


# ============================================================
# 4. 简单MLP基线模型（用于对比）
# ============================================================

class MLPBaseline(nn.Module):
    """简单的MLP基线模型"""
    def __init__(self, input_dim, seq_len=90, output_len=90, hidden_dim=256):
        super().__init__()
        self.output_len = output_len
        self.flatten_dim = seq_len * input_dim
        self.net = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_len),
        )

    def forward(self, src, target_seq=None, teacher_forcing_ratio=0.5, output_len=None):
        if output_len is None:
            output_len = self.output_len
        batch_size = src.size(0)
        x = src.reshape(batch_size, -1)
        out = self.net(x)
        if output_len != self.output_len:
            # 截取或填充
            if output_len < self.output_len:
                out = out[:, :output_len]
            else:
                # 无法扩展，用最后值重复
                pad = out[:, -1:].repeat(1, output_len - self.output_len)
                out = torch.cat([out, pad], dim=1)
        return out


# ============================================================
# 5. 模型工厂函数
# ============================================================

def create_model(model_name, input_dim, output_len=90, **kwargs):
    """创建指定类型的模型"""
    if model_name == 'lstm':
        return LSTMModel(
            input_dim=input_dim,
            hidden_dim=kwargs.get('hidden_dim', 128),
            num_layers=kwargs.get('num_layers', 2),
            dropout=kwargs.get('dropout', 0.2))
    elif model_name == 'transformer':
        return TransformerModel(
            input_dim=input_dim,
            d_model=kwargs.get('d_model', 128),
            nhead=kwargs.get('nhead', 4),
            num_encoder_layers=kwargs.get('num_encoder_layers', 3),
            num_decoder_layers=kwargs.get('num_decoder_layers', 3),
            dim_feedforward=kwargs.get('dim_feedforward', 256),
            dropout=kwargs.get('dropout', 0.1),
            output_len=output_len)
    elif model_name == 'cnn_transformer':
        return CNNTransformerModel(
            input_dim=input_dim,
            d_model=kwargs.get('d_model', 128),
            cnn_dim=kwargs.get('cnn_dim', 64),
            nhead=kwargs.get('nhead', 4),
            num_encoder_layers=kwargs.get('num_encoder_layers', 3),
            dim_feedforward=kwargs.get('dim_feedforward', 256),
            dropout=kwargs.get('dropout', 0.1),
            output_len=output_len)
    elif model_name == 'mlp':
        return MLPBaseline(
            input_dim=input_dim,
            seq_len=kwargs.get('seq_len', 90),
            output_len=output_len,
            hidden_dim=kwargs.get('hidden_dim', 256))
    else:
        raise ValueError(f"Unknown model: {model_name}")
