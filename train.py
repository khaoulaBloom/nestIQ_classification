import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data import PropertyMultiTaskDataset
from model import MultiTaskPropertyModel


def train_model(
    num_samples: int = 5000,
    batch_size: int = 64,
    epochs: int = 80,
    lr: float = 1e-3,
    save_path: str = "saved_model.pt",
) -> None:
    dataset = PropertyMultiTaskDataset(num_samples=num_samples)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MultiTaskPropertyModel(input_dim=5, hidden_dim=64)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for x_batch, y_reg_batch, y_cls_batch in train_loader:
            pred_reg, pred_logit = model(x_batch)

            regression_loss = mse_loss(pred_reg, y_reg_batch)
            classification_loss = bce_loss(pred_logit, y_cls_batch)
            total_loss = regression_loss + classification_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x_batch, y_reg_batch, y_cls_batch in val_loader:
                pred_reg, pred_logit = model(x_batch)

                regression_loss = mse_loss(pred_reg, y_reg_batch)
                classification_loss = bce_loss(pred_logit, y_cls_batch)
                total_loss = regression_loss + classification_loss
                val_loss += total_loss.item()

                probs = torch.sigmoid(pred_logit)
                preds = (probs >= 0.5).float()
                correct += (preds == y_cls_batch).sum().item()
                total += y_cls_batch.size(0)

        val_acc = correct / total if total else 0.0
        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"train_loss={epoch_loss / len(train_loader):.4f} | "
            f"val_loss={val_loss / len(val_loader):.4f} | "
            f"val_cls_acc={val_acc:.4f}"
        )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": 5,
        "hidden_dim": 64,
        "reg_mean": dataset.reg_mean,
        "reg_std": dataset.reg_std,
    }
    torch.save(checkpoint, save_path)
    print(f"Saved model checkpoint to {save_path}")


if __name__ == "__main__":
    train_model()
