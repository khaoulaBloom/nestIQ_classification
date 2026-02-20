import torch
from torch.utils.data import Dataset


class PropertyMultiTaskDataset(Dataset):
    """Synthetic dataset for multi-task learning:
    - Regression targets: sale_price, monthly_rent
    - Classification target: high_demand (0/1)
    """

    def __init__(self, num_samples: int = 4000, seed: int = 42) -> None:
        gen = torch.Generator().manual_seed(seed)

        area = torch.rand(num_samples, 1, generator=gen) * 480 + 20
        bedrooms = torch.randint(1, 7, (num_samples, 1), generator=gen).float()
        bathrooms = torch.randint(1, 5, (num_samples, 1), generator=gen).float()
        distance = torch.rand(num_samples, 1, generator=gen) * 30
        age = torch.rand(num_samples, 1, generator=gen) * 80

        self.X = torch.cat([area, bedrooms, bathrooms, distance, age], dim=1).float()

        sale_price = (
            area * 2600
            + bedrooms * 42000
            + bathrooms * 28000
            - distance * 6500
            - age * 1800
            + torch.randn(num_samples, 1, generator=gen) * 12000
            + 90000
        )
        sale_price = torch.clamp(sale_price, min=85000)

        monthly_rent = (
            area * 8.6
            + bedrooms * 220
            + bathrooms * 170
            - distance * 46
            - age * 8
            + torch.randn(num_samples, 1, generator=gen) * 120
            + 450
        )
        monthly_rent = torch.clamp(monthly_rent, min=500)

        demand_logit = (
            0.015 * area
            + 0.72 * bedrooms
            + 0.48 * bathrooms
            - 0.11 * distance
            - 0.045 * age
            + torch.randn(num_samples, 1, generator=gen) * 0.65
        )
        demand_prob = torch.sigmoid(demand_logit)
        demand_target = torch.bernoulli(demand_prob)

        self.y_reg = torch.cat([sale_price, monthly_rent], dim=1).float()
        self.y_cls = demand_target.float()

        # Normalize regression targets for stable MSE optimization.
        self.reg_mean = self.y_reg.mean(dim=0)
        self.reg_std = self.y_reg.std(dim=0).clamp_min(1e-6)
        self.y_reg_norm = (self.y_reg - self.reg_mean) / self.reg_std

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y_reg_norm[idx], self.y_cls[idx]
