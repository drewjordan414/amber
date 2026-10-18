"""Fine-tune a pretrained EfficientNet-B0 (ImageNet) into a growth-stage classifier on the photos amber.py collects.

Train:       python train.py                 (reads data/photos/<stage>/*.jpg, writes stage_model.pt)
DINOv2:      python train.py --model dinov2  (frozen DINOv2 ViT-S/14 + linear head; compare val against the default)
Smoke test:  python train.py --test          (split check + 1 epoch on generated images)
"""
import argparse, re, tempfile, zlib
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

HERE = Path(__file__).parent
VAL_EVERY = 5  # every 5th day goes to validation
SIZE = 224


def is_val(path):
    # Split by day, not by photo: hourly shots from the same day are near-duplicates,
    # so a random split leaks and val accuracy looks far better than it is.
    stem = Path(path).stem
    if re.fullmatch(r"\d{8}-\d{4}", stem):  # Amber's own photos
        return datetime.strptime(stem[:8], "%Y%m%d").toordinal() % VAL_EVERY == 0
    # prepare.py crops are <source>__<n>: keep every crop of one photo on the same side.
    # Phone names (20220810_065415963_iOS) are bursts seconds apart: group them by 10 minutes.
    src = stem.split("__")[0]
    m = re.match(r"\d{8}_\d{3}", src)
    return zlib.crc32((m.group() if m else src).encode()) % VAL_EVERY == 0


def make_model(arch, n_classes, pretrained=True):
    if arch == "dinov2":
        # Frozen backbone + linear head: only ~2k trainable params, so one grow's photos can't overfit the backbone.
        # ponytail: head only; unfreeze the last 2-4 blocks (lr ~1e-5) if val accuracy plateaus
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=pretrained)
        backbone.requires_grad_(False)
        return nn.Sequential(backbone, nn.Linear(backbone.embed_dim, n_classes))  # hub model returns the CLS embedding
    # ponytail: EfficientNet-B0 is small enough for a Pi; try convnext_tiny or DINOv2 if val accuracy plateaus
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)  # new head for our stages
    return model


norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats the backbone was trained on
TRAIN_TF = transforms.Compose([
    # ponytail: augmentation is doing the work of "more plants" until there are more grows
    transforms.RandomResizedCrop(SIZE, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.3, 0.3, 0.2, 0.02),  # small hue shift: pistil/leaf color is a real stage cue
    transforms.ToTensor(), norm])
EVAL_TF = transforms.Compose([transforms.Resize(SIZE + 32), transforms.CenterCrop(SIZE), transforms.ToTensor(), norm])


def train(data_dir, epochs, out, arch="efficientnet_b0", pretrained=True):
    train_ds, val_ds = datasets.ImageFolder(data_dir, TRAIN_TF), datasets.ImageFolder(data_dir, EVAL_TF)
    split = [is_val(p) for p, _ in train_ds.samples]
    tr = [i for i, v in enumerate(split) if not v]
    va = [i for i, v in enumerate(split) if v]
    classes = train_ds.classes
    counts = Counter(train_ds.targets[i] for i in tr)
    print(f"classes {classes}  train {len(tr)}  val {len(va)}  per-class {dict(sorted(counts.items()))}")
    if not tr or not va:
        raise SystemExit("need photos from more days so both splits are non-empty")

    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = make_model(arch, len(classes), pretrained).to(dev)
    # Weight rare stages up (seedling is a couple weeks; flower is two months).
    w = torch.tensor([1 / max(counts[c], 1) for c in range(len(classes))], device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=w / w.sum() * len(classes))
    lr = 1e-3 if arch == "dinov2" else 3e-4  # a lone linear head wants a higher lr than a full fine-tune
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tr_dl = DataLoader(Subset(train_ds, tr), batch_size=32, shuffle=True, num_workers=2)
    va_dl = DataLoader(Subset(val_ds, va), batch_size=64, num_workers=2)

    best = -1.0
    for ep in range(epochs):
        model.train()
        total = 0.0
        for x, y in tr_dl:
            x, y = x.to(dev), y.to(dev)
            loss = loss_fn(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(y)
        sched.step()

        model.eval()
        hit, seen = Counter(), Counter()
        with torch.no_grad():
            for x, y in va_dl:
                pred = model(x.to(dev)).argmax(1).cpu()
                for t, p in zip(y.tolist(), pred.tolist()):
                    seen[t] += 1; hit[t] += t == p
        # Mean per-class accuracy, so a model that always says "flower" doesn't look good.
        acc = sum(hit[c] / seen[c] for c in seen) / len(seen)
        per = "  ".join(f"{classes[c]} {hit[c]}/{seen[c]}" for c in sorted(seen))
        print(f"epoch {ep + 1:3d}  loss {total / len(tr):.3f}  val {acc:.1%}  [{per}]")
        if acc > best:
            best = acc
            torch.save({"arch": arch, "state": model.state_dict(), "classes": classes, "size": SIZE}, out)
    print(f"best val {best:.1%} -> {out}")


def test():
    assert is_val("20261015-0900.jpg") == is_val("20261015-2300.jpg")  # same day, same side
    days = [f"202610{d:02d}-1200.jpg" for d in range(1, 31)]
    assert sum(map(is_val, days)) == 6  # 1 in 5 days
    assert len({is_val(f"K-10-__{i}.jpg") for i in range(20)}) == 1  # crops of one photo stay together
    assert len({is_val(f"20220810_06{s:02d}15963_iOS__0.jpg") for s in range(50, 60)}) == 1  # one burst, one side
    # 1-epoch smoke run on generated images, two stages, 10 days each.
    from PIL import Image
    with tempfile.TemporaryDirectory() as d:
        for stage, color in (("veg", (40, 160, 40)), ("flower", (200, 180, 90))):
            (Path(d) / stage).mkdir()
            for day in range(1, 11):
                Image.new("RGB", (320, 240), color).save(Path(d) / stage / f"202610{day:02d}-1200.jpg")
        train(d, 1, Path(d) / "m.pt")
        ck = torch.load(Path(d) / "m.pt")
        assert ck["classes"] == ["flower", "veg"]
    m = make_model("dinov2", 3, pretrained=False)
    assert m(torch.zeros(1, 3, SIZE, SIZE)).shape == (1, 3)
    assert not any(q.requires_grad for q in m[0].parameters())  # backbone frozen
    print("ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=HERE / "data" / "photos")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--out", default=HERE / "stage_model.pt")
    p.add_argument("--model", choices=["efficientnet_b0", "dinov2"], default="efficientnet_b0")
    p.add_argument("--test", action="store_true")
    a = p.parse_args()
    test() if a.test else train(a.data, a.epochs, a.out, a.model)
