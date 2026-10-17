"""Fine-tune a pretrained EfficientNet-B0 (ImageNet) into a growth-stage classifier on the photos amber.py collects.

Train:       python train.py                 (reads data/photos/<stage>/*.jpg, writes stage_model.pt)
Smoke test:  python train.py --test          (split check + 1 epoch on generated images)
"""
import argparse, tempfile
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
    day = datetime.strptime(Path(path).stem[:8], "%Y%m%d").toordinal()
    return day % VAL_EVERY == 0


def make_model(n_classes, pretrained=True):
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


def train(data_dir, epochs, out, pretrained=True):
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
    model = make_model(len(classes), pretrained).to(dev)
    # Weight rare stages up (seedling is a couple weeks; flower is two months).
    w = torch.tensor([1 / max(counts[c], 1) for c in range(len(classes))], device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=w / w.sum() * len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
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
            torch.save({"arch": "efficientnet_b0", "state": model.state_dict(), "classes": classes, "size": SIZE}, out)
    print(f"best val {best:.1%} -> {out}")


def test():
    assert is_val("20261015-0900.jpg") == is_val("20261015-2300.jpg")  # same day, same side
    days = [f"202610{d:02d}-1200.jpg" for d in range(1, 31)]
    assert sum(map(is_val, days)) == 6  # 1 in 5 days
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
    print("ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=HERE / "data" / "photos")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--out", default=HERE / "stage_model.pt")
    p.add_argument("--test", action="store_true")
    a = p.parse_args()
    test() if a.test else train(a.data, a.epochs, a.out)
