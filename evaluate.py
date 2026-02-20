import torch
from torch.utils.data import DataLoader, random_split

from data import PropertyMultiTaskDataset
from model import MultiTaskPropertyModel


def evaluate_model(model_path: str = "saved_model.pt", num_samples: int = 1200, batch_size: int = 128) -> None:
    dataset = PropertyMultiTaskDataset(num_samples=num_samples, seed=123)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = random_split(dataset, [train_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    checkpoint = torch.load(model_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint format is invalid. Train the multitask model first.")

    model = MultiTaskPropertyModel(
        input_dim=checkpoint.get("input_dim", 5),
        hidden_dim=checkpoint.get("hidden_dim", 64),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    reg_mean = checkpoint.get("reg_mean", torch.zeros(2)).float()
    reg_std = checkpoint.get("reg_std", torch.ones(2)).float()

    mse_sum = 0.0
    mae_sum = 0.0
    cls_correct = 0
    cls_total = 0

    with torch.no_grad():
        for x_batch, y_reg_norm_batch, y_cls_batch in test_loader:
            pred_reg_norm, pred_logit = model(x_batch)

            y_reg = y_reg_norm_batch * reg_std.unsqueeze(0) + reg_mean.unsqueeze(0)
            pred_reg = pred_reg_norm * reg_std.unsqueeze(0) + reg_mean.unsqueeze(0)

            mse_sum += torch.mean((pred_reg - y_reg) ** 2).item()
            mae_sum += torch.mean(torch.abs(pred_reg - y_reg)).item()

            probs = torch.sigmoid(pred_logit)
            preds = (probs >= 0.5).float()
            cls_correct += (preds == y_cls_batch).sum().item()
            cls_total += y_cls_batch.size(0)

    avg_mse = mse_sum / len(test_loader)
    avg_mae = mae_sum / len(test_loader)
    cls_acc = cls_correct / cls_total if cls_total else 0.0

    print(f"Regression MSE: {avg_mse:.2f}")
    print(f"Regression MAE: {avg_mae:.2f}")
    print(f"Classification Accuracy: {cls_acc:.4f}")


if __name__ == "__main__":
    evaluate_model()
