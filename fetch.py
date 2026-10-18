"""Download public cannabis image datasets into data/public/<task>/<name>/ (survey: vault Projects/Amber/Amber Datasets).

Roboflow:    export ROBOFLOW_API_KEY=...     (free account, Settings -> API Keys)
Kaggle:      export KAGGLE_API_TOKEN=KGAT_...  (or legacy ~/.kaggle/kaggle.json; kaggle.com -> Settings -> API)
Run:         python fetch.py                 (skips anything already downloaded)
One task:    python fetch.py --only sex
Self-check:  python fetch.py --test
"""
import argparse, base64, json, os, shutil, time, urllib.request, zipfile
from pathlib import Path

OUT = Path(__file__).parent / "data" / "public"

# task -> Roboflow "workspace/project" (all CC BY 4.0) or "kaggle:owner/dataset"
DATASETS = {
    "sex": ["rajabathburiram/sex-cannabis", "rajabathburiram/female-cannabis", "cannabis-5jdhs/male-z6qkq",
            "cannabis-5jdhs/1-idtbp", "cannabis-5jdhs/2-gkfrb", "cannabis-5jdhs/3-ixz3m"],
    "trichome": ["trichome-identification/trichome-colour-detection"],
    "health": ["chokun1/cannabis-al8sm", "leaf-cannabis/cannabis-species-apfin", "pinyok/cannabis-sm6ax",
               "yellow-leaveswithered-jfia5/cannabis-diseases-lgeaj",
               "kaggle:engineeringubu/leaf-manifestation-diseases-of-cannabis"],  # 17.9 GB, CC BY-NC-SA: personal use only
    "plant": ["guy-shani/cannabis-nodes", "byron-wade-aatcj/cannabis-eph09"],
}


def get_json(url):
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def pick(info):
    """Newest version; class folders for classifiers, YOLO labels for detection/segmentation."""
    ver = max(int(v["id"].rsplit("/", 1)[1]) for v in info["versions"])
    return ver, "folder" if info["project"]["type"] == "classification" else "yolov8"


def roboflow_link(slug, key):
    ver, fmt = pick(get_json(f"https://api.roboflow.com/{slug}?api_key={key}"))
    for _ in range(60):  # Roboflow builds the export zip on first request
        d = get_json(f"https://api.roboflow.com/{slug}/{ver}/{fmt}?api_key={key}")
        if d.get("export", {}).get("link"):
            return d["export"]["link"]
        time.sleep(5)
    raise RuntimeError("export never became ready")


def kaggle_request(slug):
    if token := os.environ.get("KAGGLE_API_TOKEN"):  # new-style KGAT_ token
        auth = f"Bearer {token}"
    else:  # legacy kaggle.json
        c = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
        auth = "Basic " + base64.b64encode(f"{c['username']}:{c['key']}".encode()).decode()
    return urllib.request.Request(f"https://www.kaggle.com/api/v1/datasets/download/{slug}",
                                  headers={"Authorization": auth})


def dest(task, slug):
    return OUT / task / slug.split(":")[-1].replace("/", "__")


def fetch(task, slug):
    d = dest(task, slug)
    if d.exists():
        return print("have", d)
    src = kaggle_request(slug[7:]) if slug.startswith("kaggle:") else roboflow_link(slug, os.environ["ROBOFLOW_API_KEY"])
    zpath, part = d.parent / (d.name + ".zip"), d.parent / (d.name + ".part")
    d.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(src) as r, zpath.open("wb") as f:
        shutil.copyfileobj(r, f)  # streamed to disk: the Kaggle set is ~18 GB
    shutil.rmtree(part, ignore_errors=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(part)  # zipfile drops ".." and absolute paths, so nothing lands outside part/
    (part / "SOURCE.txt").write_text(f"{slug}\n")  # CC BY needs attribution; keep where it came from
    part.rename(d)  # only a complete extract counts as "have"
    zpath.unlink()
    print("got ", d)


def test():
    info = {"project": {"type": "object-detection"},
            "versions": [{"id": "ws/p/5"}, {"id": "ws/p/3"}, {"id": "ws/p/10"}]}
    assert pick(info) == (10, "yolov8")
    assert pick({**info, "project": {"type": "classification"}})[1] == "folder"
    assert dest("health", "kaggle:a/b") == OUT / "health" / "a__b"
    slugs = [s for v in DATASETS.values() for s in v]
    assert len(slugs) == len(set(slugs))
    print("ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=DATASETS)
    p.add_argument("--test", action="store_true")
    a = p.parse_args()
    if a.test:
        raise SystemExit(test())
    for task, slugs in DATASETS.items():
        if a.only in (None, task):
            for slug in slugs:
                try:
                    fetch(task, slug)
                except Exception as e:  # missing key/token or a dead project shouldn't stop the rest
                    print("skip", slug, f"({type(e).__name__}: {e})")
