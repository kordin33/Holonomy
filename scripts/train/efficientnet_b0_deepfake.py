"""
efficientnet_b0_deepfake.py

Wersja "do PyCharm" na bazie notebooka `msc_deepfake_v2.ipynb`.

Co robi:
1) (Opcjonalnie) pobiera i standaryzuje dane:
   - Dataset A: OpenRL/DeepFakeFace -> ImageFolder 224x224 w A_standardized_224/{train,val,test_A}/{real,fake}
   - Dataset B: prithivMLmods/Deepfake-vs-Real-v2 -> ImageFolder 224x224 w B_standardized_224/test_B/{real,fake}
2) Trenuje EfficientNet-B0 (2 klasy: fake/real) na A/train i wybiera najlepszy model po A/val.
3) Ewaluacja na A/test_A i B/test_B (accuracy, F1, ROC-AUC).
4) (Opcjonalnie) loguje metryki do Weights & Biases.

Uruchomienie (przykład):
    python efficientnet_b0_deepfake.py --prepare --train --eval

Wymagane paczki (minimum):
    pip install torch torchvision datasets huggingface_hub wandb scikit-learn pillow tqdm

Uwaga:
- Token HF jest opcjonalny (dla prywatnych/limitowanych zasobów). Ustaw:
    set HUGGINGFACE_TOKEN=...
- W&B jest opcjonalne. Ustaw:
    set WANDB_API_KEY=...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from PIL import Image

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

try:
    import wandb
except Exception:
    wandb = None


# -----------------------------
# Utils
# -----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def list_images(root: str | Path, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")) -> List[str]:
    root = Path(root)
    return [str(p) for p in root.rglob("*") if p.suffix.lower() in exts]


def split_70_15_15(paths: List[str]) -> Tuple[List[str], List[str], List[str]]:
    n = len(paths)
    tr = int(0.7 * n)
    va = int(0.15 * n)
    return paths[:tr], paths[tr:tr + va], paths[tr + va:]


def save_resized(
    files: Iterable[str],
    dst_dir: str | Path,
    img_size: int = 224,
    max_items: Optional[int] = None,
) -> int:
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in files:
        if max_items is not None and n >= max_items:
            break
        try:
            img = Image.open(p).convert("RGB").resize((img_size, img_size))
            out_path = dst_dir / f"{n:08d}.jpg"
            img.save(out_path, "JPEG", quality=95)
            n += 1
        except Exception:
            # uszkodzony obraz / błąd dekodowania -> pomijamy
            continue
    return n


def count_images_in_dir(dir_path: str | Path) -> int:
    dir_path = Path(dir_path)
    if not dir_path.exists():
        return 0
    return sum(1 for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"))


@dataclass
class PathsCfg:
    data_root: Path
    a_raw: Path
    a_std: Path
    b_std: Path
    meta_dir: Path
    out_dir: Path


def build_paths(data_root: str, out_dir: str) -> PathsCfg:
    dr = Path(data_root).expanduser().resolve()
    od = Path(out_dir).expanduser().resolve()
    return PathsCfg(
        data_root=dr,
        a_raw=dr / "DeepFakeFace",
        a_std=dr / "A_standardized_224",
        b_std=dr / "B_standardized_224",
        meta_dir=dr / "meta",
        out_dir=od,
    )


# -----------------------------
# Dataset preparation
# -----------------------------
def prepare_dataset_a(
    cfg: PathsCfg,
    img_size: int,
    seed: int,
    max_per_class: Optional[int],
    hf_repo: str = "OpenRL/DeepFakeFace",
    hf_token: Optional[str] = None,
) -> Dict[str, int]:
    """
    Pobiera dataset A (Hugging Face) i buduje ImageFolder w A_standardized_224.
    Oczekujemy struktury:
        DeepFakeFace/real/**.jpg
        DeepFakeFace/fake/**.jpg
    """
    cfg.a_raw.mkdir(parents=True, exist_ok=True)
    cfg.a_std.mkdir(parents=True, exist_ok=True)

    if snapshot_download is None:
        raise RuntimeError("Brak huggingface_hub. Zainstaluj: pip install huggingface_hub")

    # Pobranie snapshotu (jeśli nie ma)
    # Jeśli katalog nie jest pusty, uznajemy że już pobrane.
    if not any(cfg.a_raw.iterdir()):
        snapshot_download(
            repo_id=hf_repo,
            repo_type="dataset",
            local_dir=str(cfg.a_raw),
            token=hf_token,
        )

    real_dir = cfg.a_raw / "real"
    fake_dir = cfg.a_raw / "fake"
    if not real_dir.exists() or not fake_dir.exists():
        raise RuntimeError(
            f"Nie znalazłem folderów {real_dir} i {fake_dir}. "
            "Sprawdź strukturę pobranego datasetu A w data_root/DeepFakeFace."
        )

    r_all = list_images(real_dir)
    f_all = list_images(fake_dir)

    rng = random.Random(seed)
    rng.shuffle(r_all)
    rng.shuffle(f_all)

    n = min(len(r_all), len(f_all))
    if max_per_class is not None:
        n = min(n, max_per_class)

    r_all = r_all[:n]
    f_all = f_all[:n]

    r_tr, r_va, r_te = split_70_15_15(r_all)
    f_tr, f_va, f_te = split_70_15_15(f_all)

    # czyścimy poprzednią standaryzację (żeby nie mieszać wersji)
    if cfg.a_std.exists():
        shutil.rmtree(cfg.a_std)
    (cfg.a_std / "train").mkdir(parents=True, exist_ok=True)
    (cfg.a_std / "val").mkdir(parents=True, exist_ok=True)
    (cfg.a_std / "test_A").mkdir(parents=True, exist_ok=True)

    counts = {}
    counts["A_train_real"] = save_resized(r_tr, cfg.a_std / "train" / "real", img_size)
    counts["A_train_fake"] = save_resized(f_tr, cfg.a_std / "train" / "fake", img_size)
    counts["A_val_real"] = save_resized(r_va, cfg.a_std / "val" / "real", img_size)
    counts["A_val_fake"] = save_resized(f_va, cfg.a_std / "val" / "fake", img_size)
    counts["A_testA_real"] = save_resized(r_te, cfg.a_std / "test_A" / "real", img_size)
    counts["A_testA_fake"] = save_resized(f_te, cfg.a_std / "test_A" / "fake", img_size)

    return counts


def prepare_dataset_b(
    cfg: PathsCfg,
    img_size: int,
    seed: int,
    max_per_class: Optional[int],
    hf_repo: str = "prithivMLmods/Deepfake-vs-Real-v2",
) -> Dict[str, int]:
    """
    Buduje ImageFolder w B_standardized_224/test_B/{real,fake}.
    W notebooku mapowanie było: label==0 -> fake, label==1 -> real.
    """
    cfg.b_std.mkdir(parents=True, exist_ok=True)
    dst_root = cfg.b_std / "test_B"
    real_dst = dst_root / "real"
    fake_dst = dst_root / "fake"

    if load_dataset is None:
        raise RuntimeError("Brak datasets. Zainstaluj: pip install datasets")

    # czyść poprzednią wersję (żeby nie mieszać)
    if dst_root.exists():
        shutil.rmtree(dst_root)
    real_dst.mkdir(parents=True, exist_ok=True)
    fake_dst.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(hf_repo, split="train")

    # deterministycznie: najpierw przemieszaj, potem zapisuj z limitami
    ds = ds.shuffle(seed=seed)

    counts = {"B_testB_real": 0, "B_testB_fake": 0}
    limit = None if max_per_class is None else int(max_per_class)

    if "label" not in ds.column_names:
        raise RuntimeError(f"Dataset B nie ma kolumny 'label'. Kolumny: {ds.column_names}")
    img_col = "image" if "image" in ds.column_names else ds.column_names[0]

    for row in tqdm(ds, desc="Zapisywanie B -> ImageFolder"):
        lbl = int(row["label"])
        if lbl not in (0, 1):
            continue

        # notebook: lbl==0 -> fake, else -> real
        is_real = (lbl != 0)
        if is_real:
            if limit is not None and counts["B_testB_real"] >= limit:
                continue
            out_dir = real_dst
            out_idx = counts["B_testB_real"]
        else:
            if limit is not None and counts["B_testB_fake"] >= limit:
                continue
            out_dir = fake_dst
            out_idx = counts["B_testB_fake"]

        try:
            img = row[img_col]
            if hasattr(img, "convert"):
                pil = img.convert("RGB")
            else:
                pil = Image.fromarray(np.array(img)).convert("RGB")
            pil = pil.resize((img_size, img_size))
            pil.save(out_dir / f"{out_idx:08d}.jpg", "JPEG", quality=95)
            if is_real:
                counts["B_testB_real"] += 1
            else:
                counts["B_testB_fake"] += 1
        except Exception:
            continue

        if limit is not None and counts["B_testB_real"] >= limit and counts["B_testB_fake"] >= limit:
            break

    return counts


def write_meta(cfg: PathsCfg, meta: Dict) -> None:
    cfg.meta_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.meta_dir / "dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# -----------------------------
# Training & Evaluation
# -----------------------------
@torch.no_grad()
def eval_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    positive_class: str = "real",
) -> Dict[str, float]:
    model.eval()
    y_true = []
    y_prob = []

    if positive_class in class_names:
        pos_idx = class_names.index(positive_class)
    else:
        pos_idx = 1  # fallback

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        prob = logits.softmax(dim=1)[:, pos_idx].detach().cpu().numpy()
        y_prob.append(prob)
        y_true.append(y.numpy())

    y_prob = np.concatenate(y_prob)
    y_true = np.concatenate(y_true)

    y_bin = (y_true == pos_idx).astype(np.int32)
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
    parser.add_argument("--data-root", default="./data", help="Katalog na dane (A/B + meta)")
    parser.add_argument("--out-dir", default="./outputs", help="Katalog na checkpointy i logi")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--max-per-class-a", type=int, default=None)
    parser.add_argument("--max-per-class-b", type=int, default=None)

    parser.add_argument("--prepare", action="store_true", help="Pobierz/standaryzuj datasety A i B")
    parser.add_argument("--train", dest="do_train", action="store_true", help="Trenuj model")
    parser.add_argument("--eval", dest="do_eval", action="store_true", help="Ewaluuj model na test_A i test_B")

    parser.add_argument("--use-wandb", action="store_true", help="Loguj do W&B (wymaga WANDB_API_KEY)")
    parser.add_argument("--wandb-project", default="msc-deepfake-detection")
    parser.add_argument("--wandb-run-name", default=None)

    parser.add_argument("--hf-token", default=None, help="Token HF (opcjonalnie). Jeśli brak, spróbuję z ENV HUGGINGFACE_TOKEN")
    args = parser.parse_args()

    cfg = build_paths(args.data_root, args.out_dir)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

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
                "backbone": "efficientnet_b0",
                "data_root": str(cfg.data_root),
            },
        )

    # Prepare datasets
    meta_counts = {}
    if args.prepare:
        hf_token = args.hf_token or os.environ.get("HUGGINGFACE_TOKEN")
        a_counts = prepare_dataset_a(
            cfg=cfg,
            img_size=args.img_size,
            seed=args.seed,
            max_per_class=args.max_per_class_a,
            hf_token=hf_token,
        )
        b_counts = prepare_dataset_b(
            cfg=cfg,
            img_size=args.img_size,
            seed=args.seed,
            max_per_class=args.max_per_class_b,
        )
        meta_counts.update(a_counts)
        meta_counts.update(b_counts)

    # Datasets must exist for train/eval
    a_root = cfg.a_std
    b_root = cfg.b_std

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

    # DataLoaders
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

    # Model
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(device)

    best_path = cfg.out_dir / "best_efficientnet_b0.pth"

    if args.do_train:
        train_stats = train(
            model=model,
            dl_train=dl_train,
            dl_val=dl_val,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            out_path=best_path,
            use_wandb=args.use_wandb,
        )
        meta_counts.update({f"train_{k}": v for k, v in train_stats.items()})

    if args.do_eval:
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device))
        else:
            print(f"Uwaga: nie znalazłem checkpointu: {best_path} -> ewaluuję aktualny model")

        mA = eval_loader(model, dl_testA, device, class_names, positive_class="real")
        mB = eval_loader(model, dl_testB, device, class_names, positive_class="real")
        print("TEST_A:", mA)
        print("TEST_B:", mB)

        if args.use_wandb and wandb is not None:
            wandb.log({
                "testA_acc": mA["acc"], "testA_f1": mA["f1"], "testA_auc": mA["auc"],
                "testB_acc": mB["acc"], "testB_f1": mB["f1"], "testB_auc": mB["auc"],
            })

        meta_counts.update({
            "testA_acc": mA["acc"], "testA_f1": mA["f1"], "testA_auc": mA["auc"],
            "testB_acc": mB["acc"], "testB_f1": mB["f1"], "testB_auc": mB["auc"],
        })

    # Meta
    meta = {
        "created_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "data_root": str(cfg.data_root),
        "a_std": str(cfg.a_std),
        "b_std": str(cfg.b_std),
        "out_dir": str(cfg.out_dir),
        "img_size": args.img_size,
        "counts": meta_counts,
        "class_names": class_names,
    }
    write_meta(cfg, meta)

    if args.use_wandb and wandb is not None:
        try:
            art = wandb.Artifact("meta_files", type="meta")
            art.add_dir(str(cfg.meta_dir))
            wandb.log_artifact(art)
        except Exception:
            pass
        wandb.finish()


if __name__ == "__main__":
    main()
