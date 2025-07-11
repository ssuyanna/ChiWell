import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2, num_classes=8, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)

        # 分类头：只用最后一个时间步的输出做分类
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [batch_size, T, 6]
        lstm_out, _ = self.lstm(x)            # lstm_out: [B, T, hidden_dim]
        last_output = lstm_out[:, -1, :]      # 取最后时间步的输出 [B, hidden_dim]
        logits = self.classifier(last_output) # 输出 logits
        return logits
