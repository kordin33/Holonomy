"""
vit_b16_deepfake.py

Wersja "do PyCharm" na bazie notebooka `ViT.ipynb`.

Scenariusze:
- trening ViT-B/16 na A_standardized_224 (ImageFolder) + wybór najlepszego modelu po val
- ewaluacja na test_A oraz test_B
- opcjonalnie: "flip" na test_B (gdy etykiety w B są odwrócone względem A)

Uruchomienie (przykład):
    python vit_b16_deepfake.py --train --eval
albo tylko ewaluacja:
    python vit_b16_deepfake.py --eval --weights ./outputs/best_vit_b16.pth --flip-testB

Wymagane paczki:
    pip install torch torchvision scikit-learn tqdm pillow wandb

Uwaga:
- Skrypt zakłada, że dane A/B są już przygotowane (np. przez efficientnet_b0_deepfake.py --prepare).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    import wandb
except Exception:
    wandb = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def eval_loader_vit(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    positive_class: str = "real",
    flip: bool = False,
) -> Dict[str, float]:
    """
    Jeśli flip=True, to traktujemy predykcję jako "odwróconą":
        p_flip = 1 - p
        y_pred_flip = p_flip >= 0.5
    (dokładnie taki sens miał eval_loader_vit_flip w notebooku)
    """
    model.eval()

    if positive_class in class_names:
        pos_idx = class_names.index(positive_class)
    else:
        pos_idx = 1

    y_true = []
    y_prob = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        prob = logits.softmax(dim=1)[:, pos_idx].detach().cpu().numpy()
        y_prob.append(prob)
        y_true.append(y.numpy())

    y_prob = np.concatenate(y_prob)
    y_true = np.concatenate(y_true)

    y_bin = (y_true == pos_idx).astype(np.int32)

    if flip:
        y_prob = 1.0 - y_prob

    y_pred = (y_prob >= 0.5).astype(np.int32)

    out = {
        "acc": float(accuracy_score(y_bin, y_pred)),
        "f1": float(f1_score(y_bin, y_pred)),
    }
    try:
        out["auc"] = float(roc_auc_score(y_bin, y_prob))
    except Exception:
        out["auc"] = float("nan")
    return out


def train(
    model: nn.Module,
    dl_train: DataLoader,
    dl_val: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    out_path: Path,
    use_wandb: bool = False,
) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_losses = []
        tr_correct = 0
        tr_total = 0

        for x, y in tqdm(dl_train, desc=f"Epoch {epoch}/{epochs} [train]"):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tr_losses.append(loss.item())
            tr_correct += (logits.argmax(dim=1) == y).sum().item()
            tr_total += y.numel()

        train_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
        train_acc = float(tr_correct / max(1, tr_total))

        # val
        model.eval()
        va_losses = []
        va_correct = 0
        va_total = 0

        with torch.no_grad():
            for x, y in tqdm(dl_val, desc=f"Epoch {epoch}/{epochs} [val]"):
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                va_losses.append(loss.item())
                va_correct += (logits.argmax(dim=1) == y).sum().item()
                va_total += y.numel()

        val_loss = float(np.mean(va_losses)) if va_losses else float("nan")
        val_acc = float(va_correct / max(1, va_total))

        logs = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        print(json.dumps(logs, indent=2))

        if use_wandb and wandb is not None:
            wandb.log(logs)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_path)
            if use_wandb and wandb is not None:
                wandb.log({"best_val_acc": best_val_acc})

    return {"best_val_acc": float(best_val_acc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data", help="Katalog na dane (A_standardized_224 i B_standardized_224)")
    parser.add_argument("--out-dir", default="./outputs", help="Katalog na checkpointy")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-5)  # typowo mniejszy LR dla ViT
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--train", dest="do_train", action="store_true")
    parser.add_argument("--eval", dest="do_eval", action="store_true")
    parser.add_argument("--weights", default=None, help="Ścieżka do wag .pth (opcjonalnie). Jeśli brak -> użyję best_vit_b16.pth z out-dir")
    parser.add_argument("--flip-testB", action="store_true", help="Zastosuj flip na test_B (gdy label mapping jest odwrócony)")

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="msc-deepfake-detection")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    data_root = Path(args.data_root).expanduser().resolve()
    a_root = data_root / "A_standardized_224"
    b_root = data_root / "B_standardized_224"
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # W&B
    if args.use_wandb:
        if wandb is None:
            raise RuntimeError("Brak wandb. Zainstaluj: pip install wandb")
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "img_size": args.img_size,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "epochs": args.epochs,
                "seed": args.seed,
                "backbone": "vit_b_16",
                "data_root": str(data_root),
            },
        )

    # Transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.1, 0.1, 0.1, 0.05),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    eval_tf = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    ds_train = tv.datasets.ImageFolder(a_root / "train", transform=train_tf)
    ds_val = tv.datasets.ImageFolder(a_root / "val", transform=eval_tf)
    ds_testA = tv.datasets.ImageFolder(a_root / "test_A", transform=eval_tf)
    ds_testB = tv.datasets.ImageFolder(b_root / "test_B", transform=eval_tf)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=64, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    dl_testA = DataLoader(ds_testA, batch_size=64, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    dl_testB = DataLoader(ds_testB, batch_size=64, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    class_names = ds_train.classes
    print("class_names:", class_names)
    if args.use_wandb and wandb is not None:
        wandb.config.update({"class_names": class_names}, allow_val_change=True)

    # Model ViT-B/16
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, 2)
    model = model.to(device)

    best_path = out_dir / "best_vit_b16.pth"
    weights_path = Path(args.weights).expanduser().resolve() if args.weights else best_path

    if args.do_train:
        train(model, dl_train, dl_val, device, args.epochs, args.lr, best_path, use_wandb=args.use_wandb)

    if args.do_eval:
        if weights_path.exists():
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print("Loaded weights:", str(weights_path))
        else:
            print(f"Uwaga: nie znalazłem wag: {weights_path} -> ewaluuję aktualny model")

        mA = eval_loader_vit(model, dl_testA, device, class_names, positive_class="real", flip=False)
        mB = eval_loader_vit(model, dl_testB, device, class_names, positive_class="real", flip=args.flip_testB)

        print("ViT TEST_A:", mA)
        print("ViT TEST_B:", mB, "(flip)" if args.flip_testB else "")

        if args.use_wandb and wandb is not None:
            wandb.log({
                "vit_testA_acc": mA["acc"], "vit_testA_f1": mA["f1"], "vit_testA_auc": mA["auc"],
                "vit_testB_acc": mB["acc"], "vit_testB_f1": mB["f1"], "vit_testB_auc": mB["auc"],
                "vit_testB_flip": bool(args.flip_testB),
            })
            wandb.finish()


if __name__ == "__main__":
    main()
