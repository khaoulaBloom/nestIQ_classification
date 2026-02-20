import torch
import torch.nn as nn


class MultiTaskPropertyModel(nn.Module):
    """Shared base + two heads:
    - Regression head: 2 outputs (sale_price, rent_price)
    - Classification head: 1 logit (high_demand)
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64) -> None:
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.regression_head = nn.Linear(hidden_dim, 2)
        self.classification_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        shared = self.base(x)
        reg_out = self.regression_head(shared)
        cls_logit = self.classification_head(shared)
        return reg_out, cls_logit
