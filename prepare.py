"""Merge data/public/ into one clean training folder per task: data/v1/<task>/<class>/<source>[__<box>].jpg

Detection sets become one crop per box, so every task trains with train.py as a plain classifier.
Roboflow's augmented copies (<name>_jpg.rf.<hash>.jpg) collapse to one image per source photo.

Build:       python prepare.py           (rebuilds data/v1/, prints counts and dropped labels)
Then:        python train.py --data data/v1/sex --model dinov2 --out sex_model.pt
Self-check:  python prepare.py --test
"""
import argparse, ast, re, shutil
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
PUB, OUT = HERE / "data" / "public", HERE / "data" / "v1"
PAD = 0.15   # context around each box, as a fraction of its size
MIN_PX = 16  # boxes smaller than this are too blurry to learn from
IMG = {".jpg", ".jpeg", ".png"}  # ponytail: 10-LMDC has ~3.3k HEIC; add pillow-heif if those are ever needed
MAX_PX = 512  # copies are downscaled; training runs at 224 and 10-LMDC originals are huge

SOURCES = {  # task -> dataset folders under data/public/
    "sex": ["sex/*", "plant/byron-wade*"],
    "trichome": ["trichome/*"],
    "health": ["health/*"],
    "type": ["plant/byron-wade*"],  # ponytail: one 336-photo set; hybrid has ~10, add the Thai cultivar set if it matters
}
# Source label (lowercased) -> v1 class. Anything unlisted is dropped and reported.
CLASSES = {
    "sex": {"female": "female", "male": "male", "hermaphrodite": "herm", "hermaphroditism": "herm"},
    "trichome": {"amber": "amber", "clear": "clear", "milky": "milky"},
    "health": {"disease-free leaves": "healthy", "normal cannabis leaves": "healthy", "nm": "healthy",
               "n": "deficiency_n", "p": "deficiency_p", "k": "deficiency_k", "yellow leaves wither": "yellowing",
               "powdery mildew": "powdery_mildew", "black rot": "black_rot", "rust fungi": "rust",
               "spider mite": "spider_mite", "aphid": "aphid", "mealybug": "mealybug", "caterpillar": "caterpillar",
               "nitrogen deficiency": "deficiency_n", "whitefly": "whitefly", "fusarium wilt": "fusarium_wilt"},
}


def label(task, name):
    # 10-LMDC folders look like "4.โรคขาดไนโตรเจน Nitrogen Deficiency 1097": drop the number, Thai, count, "infestation"
    n = re.sub(r"[^\x00-\x7f]", "", name)
    n = re.sub(r"^\s*\d+\.|\s*\d+\s*$| infestation", "", n.lower(), flags=re.I)
    n = " ".join(n.split())
    tags = set(n.split())  # Byron Wade folders are tag combos like "Indica bud female healthy"
    if task == "type":
        if "hybrid" in tags or {"indica", "sativa"} <= tags: return "hybrid"
        return "indica" if "indica" in tags else "sativa" if "sativa" in tags else None
    if n in CLASSES[task]: return CLASSES[task][n]
    hits = {CLASSES[task][t] for t in tags if t in CLASSES[task]}
    return hits.pop() if len(hits) == 1 else None  # "female male" is ambiguous, so drop it


def source(path):
    """Original photo name, shared by all of Roboflow's augmented copies of it."""
    stem = re.sub(r"_(jpe?g|png)\.rf\.[0-9a-f]+$", "", Path(path).stem, flags=re.I)
    return re.sub(r"_{2,}", "_", re.sub(r"[^\w.-]+", "-", stem))  # never "__": train.py splits crops on it


def crop_box(cx, cy, w, h, W, H):
    """YOLO-normalized box -> padded pixel rect, clamped to the image."""
    w, h = w * (1 + 2 * PAD), h * (1 + 2 * PAD)
    return (max(0, round((cx - w / 2) * W)), max(0, round((cy - h / 2) * H)),
            min(W, round((cx + w / 2) * W)), min(H, round((cy + h / 2) * H)))


def items(ds):
    """Yield (image path, label name, box or None) for a YOLO export or a class-folder dataset."""
    yaml = ds / "data.yaml"
    if yaml.exists():
        line = next(l for l in yaml.read_text().splitlines() if l.startswith("names:"))
        names = ast.literal_eval(line.split(":", 1)[1].strip())
        for img in sorted(ds.glob("*/images/*")):
            lab = img.parent.parent / "labels" / (img.stem + ".txt")
            for row in lab.read_text().splitlines() if lab.exists() else []:
                c, *xywh = row.split()
                if len(xywh) == 4:  # skip polygon rows
                    yield img, names[int(c)], tuple(map(float, xywh))
    else:
        for img in sorted(p for p in ds.rglob("*") if p.suffix.lower() in IMG):
            yield img, img.parent.name, None


def build(task):
    from PIL import Image
    out = OUT / task
    shutil.rmtree(out, ignore_errors=True)
    seen, counts, dropped = set(), Counter(), Counter()
    for pattern in SOURCES[task]:
        for ds in sorted(PUB.glob(pattern)):
            if not ds.is_dir() or ds.name.endswith(".part"): continue  # fetch.py still downloading
            rows = list(items(ds))
            keep = {}  # source photo -> the one copy we use (test/valid sort first, and Roboflow doesn't augment those)
            key = lambda img, name, box: source(img) if box else f"{label(task, name)}/{source(img)}"  # folder sets reuse names like "ND (12)" across classes
            for row in rows:
                keep.setdefault(key(*row), row[0])
            # ponytail: dedupe by filename; perceptual hash if overlapping sets turn out to rename files
            keep = {s: i for s, i in keep.items() if s not in seen}  # already taken from an earlier dataset
            seen |= keep.keys()
            n_box = Counter()
            for img, name, box in rows:
                src = source(img)
                if keep.get(key(img, name, box)) != img: continue
                cls = label(task, name)
                if cls is None:
                    dropped[name] += 1; continue
                (out / cls).mkdir(parents=True, exist_ok=True)
                if box is None:
                    with Image.open(img) as im:
                        im.thumbnail((MAX_PX, MAX_PX))
                        im.convert("RGB").save(out / cls / f"{src}.jpg", quality=92)
                else:
                    with Image.open(img) as im:
                        x0, y0, x1, y1 = crop_box(*box, *im.size)
                        if min(x1 - x0, y1 - y0) < MIN_PX:
                            dropped[f"{name} (<{MIN_PX}px)"] += 1; continue
                        im.convert("RGB").crop((x0, y0, x1, y1)).save(out / cls / f"{src}__{n_box[src]}.jpg", quality=95)
                    n_box[src] += 1
                counts[cls] += 1
    print(f"{task:9s} {dict(sorted(counts.items()))}")
    if dropped: print(f"{'':9s} dropped {dict(dropped.most_common(8))}")


def test():
    assert label("type", "Indica bud female healthy") == "indica"
    assert label("type", "Indica female sativa") == "hybrid"
    assert label("type", "bud female") is None
    assert label("sex", "Indica bud female") == "female"
    assert label("sex", "female male") is None
    assert label("sex", "Hermaphroditism") == "herm"
    assert label("health", "NM") == "healthy"
    assert label("health", "4.โรคขาดไนโตรเจน Nitrogen Deficiency 1097") == "deficiency_n"
    assert label("health", "1. Spider Mite Infestation  801") == "spider_mite"
    assert label("health", "10.Disease-Free Leaves 1071") == "healthy"
    assert label("health", "2.โรคราน้ำค้าง Powdery Mildew1205") == "powdery_mildew"
    assert source("20220810_065415963_iOS_jpg.rf.857f26f2cb52acccd025bc5ff099d9a6.jpg") == "20220810_065415963_iOS"
    assert source("K-10-_jpg.rf.036498fe.jpg") == source("K-10-_jpg.rf.55be47a2.jpg") == "K-10-"
    assert source("CI (12).jpg") == "CI-12-" and "__" not in source("a__b (1).png")
    assert crop_box(0.5, 0.5, 0.2, 0.2, 100, 100) == (37, 37, 63, 63)
    assert crop_box(0.05, 0.5, 0.2, 0.2, 100, 100)[0] == 0  # clamped at the edge
    print("ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=SOURCES)
    p.add_argument("--test", action="store_true")
    a = p.parse_args()
    if a.test: raise SystemExit(test())
    for task in SOURCES:
        if a.only in (None, task): build(task)
