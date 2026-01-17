import torch
import torch.nn as nn

class EMG3DCNNRegressor(nn.Module):
    """
    Input:  X (B, T, C, H, W)   C=12 (6 signal + 6 mask)
    Output: y (B, 2)

    Uses Conv3d over (T,H,W).
    """
    def __init__(self, in_ch: int = 12, base: int = 16, dropout: float = 0.15):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv3d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(cout),
                nn.SiLU(inplace=True),
            )

        self.net = nn.Sequential(
            block(in_ch, base),
            block(base, base),
            nn.MaxPool3d((2, 2, 2)),

            block(base, base * 2),
            block(base * 2, base * 2),
            nn.MaxPool3d((2, 2, 2)),

            block(base * 2, base * 4),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base * 4, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,T,C,H,W) -> (B,C,T,H,W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.net(x)
        return self.head(x)
