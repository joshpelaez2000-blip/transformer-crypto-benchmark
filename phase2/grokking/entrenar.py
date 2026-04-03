#!/usr/bin/env python3
"""Entrena transformer para suma modular y XOR via Grokking."""

import json
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/kota/investigacion/fase2/grokking')
from modelo import GrokTransformer

def load_dataset(path):
    with open(path) as f:
        data = json.load(f)
    train = data["train"]
    test = data["test"]
    train_a = torch.tensor([d["a"] for d in train])
    train_b = torch.tensor([d["b"] for d in train])
    train_t = torch.tensor([d["target"] for d in train])
    test_a = torch.tensor([d["a"] for d in test])
    test_b = torch.tensor([d["b"] for d in test])
    test_t = torch.tensor([d["target"] for d in test])
    return (train_a, train_b, train_t), (test_a, test_b, test_t)

def train_model(dataset_path, name, max_epochs=5000, target_acc=0.99):
    print(f"\n{'='*60}")
    print(f"TRAINING: {name}")
    print(f"{'='*60}")

    (train_a, train_b, train_t), (test_a, test_b, test_t) = load_dataset(dataset_path)

    model = GrokTransformer(p=97, dim=128, n_layers=2, n_heads=1, ff_dim=512)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    criterion = nn.CrossEntropyLoss()

    history = {"epoch": [], "train_acc": [], "test_acc": [], "train_loss": []}
    grok_epoch = None
    best_test_acc = 0

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        logits = model(train_a, train_b)
        loss = criterion(logits, train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                train_pred = model(train_a, train_b).argmax(dim=-1)
                train_acc = (train_pred == train_t).float().mean().item()
                test_pred = model(test_a, test_b).argmax(dim=-1)
                test_acc = (test_pred == test_t).float().mean().item()

            history["epoch"].append(epoch)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
            history["train_loss"].append(loss.item())

            if test_acc > best_test_acc:
                best_test_acc = test_acc

            # Detect grokking: first time test_acc jumps from <50% to >95%
            if grok_epoch is None and test_acc > 0.95:
                grok_epoch = epoch
                print(f"  *** GROKKING at epoch {epoch}! test_acc={test_acc:.4f} ***")

            if epoch % 100 == 0:
                print(f"  Epoch {epoch}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, loss={loss.item():.4f}")

            if test_acc >= target_acc:
                print(f"  Target {target_acc} reached at epoch {epoch}")
                break

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s. Best test_acc: {best_test_acc:.4f}")
    if grok_epoch:
        print(f"Grokking occurred at epoch {grok_epoch}")
    else:
        print(f"No grokking detected (test_acc never exceeded 0.95)")

    # Save model if good enough
    if best_test_acc > 0.95:
        model_path = f"/home/kota/investigacion/fase2/grokking/modelo_{name}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved: {model_path}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["epoch"], history["train_acc"], label="Train Acc", alpha=0.8)
    ax1.plot(history["epoch"], history["test_acc"], label="Test Acc", alpha=0.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"{name}: Accuracy vs Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    if grok_epoch:
        ax1.axvline(x=grok_epoch, color='red', linestyle='--', alpha=0.5, label=f'Grok @ {grok_epoch}')
        ax1.legend()

    ax2.plot(history["epoch"], history["train_loss"], label="Train Loss", alpha=0.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title(f"{name}: Loss vs Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plot_path = f"/home/kota/investigacion/fase2/grokking/grafico_{name}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved: {plot_path}")

    return {
        "name": name,
        "epochs_trained": history["epoch"][-1] if history["epoch"] else 0,
        "best_test_acc": best_test_acc,
        "grok_epoch": grok_epoch,
        "elapsed_seconds": round(elapsed, 1),
        "history": history
    }

if __name__ == "__main__":
    results = {}

    # 1. Suma modular
    r1 = train_model(
        "/home/kota/investigacion/fase2/grokking/dataset_mod_add.json",
        "mod_add", max_epochs=5000
    )
    results["mod_add"] = r1

    # 2. XOR modular
    r2 = train_model(
        "/home/kota/investigacion/fase2/grokking/dataset_mod_xor.json",
        "mod_xor", max_epochs=5000
    )
    results["mod_xor"] = r2

    # Save results (without full history for readability)
    summary = {}
    for k, v in results.items():
        summary[k] = {key: val for key, val in v.items() if key != "history"}

    with open("/home/kota/investigacion/fase2/grokking/resultados_entrenamiento.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResultados guardados en resultados_entrenamiento.json")
