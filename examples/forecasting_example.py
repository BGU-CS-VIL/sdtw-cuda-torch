#!/usr/bin/env python3
"""
Simple time series forecasting with SoftDTW loss.

This example trains a simple MLP to forecast sine wave continuations using SoftDTW loss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from softdtw_cuda import SoftDTW


def generate_sine_waves(
    num_sequences: int = 50,
    sequence_length: int = 150,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate sine wave sequences for training."""
    sequences = []

    for _ in range(num_sequences):
        # Random frequency and phase
        freq = np.random.uniform(0.5, 3.0)
        phase = np.random.uniform(0, 4 * np.pi)

        # Generate sine wave
        t = np.linspace(0, 4 * np.pi, sequence_length)
        wave = np.sin(freq * t + phase).astype(np.float32)

        # Add small noise
        wave += 0.05 * np.random.randn(sequence_length)
        sequences.append(wave)

    sequences = np.array(sequences)

    # Split into input (first 100) and target (last 100)
    input_len = 100
    
    x = torch.from_numpy(sequences[:, :input_len]).unsqueeze(-1)  # (B, 100, 1)
    y = torch.from_numpy(sequences[:, input_len:sequence_length]).unsqueeze(-1)  # (B, 100, 1)

    return x, y


class MLPForecaster(nn.Module):
    """Simple MLP for time series forecasting."""

    def __init__(self, input_len: int = 100, output_len: int = 100, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, input_len, 1) -> y_pred: (B, output_len, 1)"""
        x_flat = x.squeeze(-1)
        y_flat = self.net(x_flat)
        return y_flat.unsqueeze(-1)


def main():
    """Train a simple forecasting model with SoftDTW loss."""
    print("Time Series Forecasting with SoftDTW Loss")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Generate data
    print("Generating data...")
    N_train = 200
    x_all, y_all = generate_sine_waves(num_sequences=300, sequence_length=200)
    x_train, y_train = x_all[:N_train], y_all[:N_train]
    x_val, y_val = x_all[N_train:N_train+50], y_all[N_train:N_train+50]
    x_test, y_test = x_all[N_train+50:], y_all[N_train+50:]
    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}\n")

    # Create model
    model = MLPForecaster(input_len=100, output_len=100, hidden_dim=128).to(device)   
    loss_fn = SoftDTW(gamma=1, normalize=False, fused=True)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Training
    print("Training...")
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience = 30 # Early stopping patience
    patience_counter = 0

    for epoch in range(300):
        # Train
        model.train()
        y_pred = model(x_train.to(device))
        loss = loss_fn(y_pred, y_train.to(device)).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())

        # Validate
        model.eval()
        with torch.no_grad():
            y_pred_val = model(x_val.to(device))
            val_loss = loss_fn(y_pred_val, y_val.to(device)).mean()
        val_losses.append(val_loss.item())

        scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch + 1:3d} | Train: {loss.item():.4f} | Val: {val_loss.item():.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}\n")
            break

    # Evaluate
    print("Evaluating...")
    model.eval()
    with torch.no_grad():
        y_pred_test = model(x_test.to(device))
        test_loss = loss_fn(y_pred_test, y_test.to(device)).mean()
    print(f"Test loss: {test_loss.item():.4f}\n")

    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("SoftDTW Loss")
    plt.title("Training Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    #plt.savefig("examples/training_curves.png", dpi=100)
    print("Training curves saved to training_curves.png")

    # Plot predictions
    print("Plotting predictions...")
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    for idx in range(4):
        ax = axes[idx]
        x = x_test[idx].numpy()
        y = y_test[idx].numpy()
        y_pred = y_pred_test[idx].cpu().numpy()

        t_input = np.arange(100)
        t_target = np.arange(100, 200)

        ax.plot(t_input, x, "k-", label="Input", linewidth=2, alpha=0.7)
        ax.plot(t_target, y, "g-", label="Ground Truth", linewidth=2)
        ax.plot(t_target, y_pred, "r--", label="Prediction", linewidth=2)

        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title(f"Sequence {idx + 1}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("examples/forecasting_results.png", dpi=100)
    print("Results saved to forecasting_results.png")
    plt.show()


if __name__ == "__main__":
    main()
