"""
YOLO11 Training Pipeline - Baby Monitor custom model.

Usage:
  python training/train.py
  python training/train.py --validate-only
  python training/train.py --export
"""

import argparse, shutil, sys
from pathlib import Path
import yaml

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] pip install ultralytics"); sys.exit(1)


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def make_dataset_yaml(cfg):
    ds   = cfg["dataset"]
    base = Path(ds["path"]).resolve()
    out  = Path("training/dataset.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump({"path": str(base),
                   "train": ds.get("train_dir", "images/train"),
                   "val":   ds.get("val_dir",   "images/val"),
                   "test":  ds.get("test_dir",  "images/test"),
                   "names": ds["class_names"],
                   "nc":    len(ds["class_names"])}, f, allow_unicode=True)
    print(f"[INFO] dataset.yaml -> {out}")
    return str(out)

def train(cfg, ds_yaml):
    model = YOLO(cfg["model"].get("base", "yolo11n.pt"))
    tc    = cfg["training"]
    model.train(data=ds_yaml, epochs=tc.get("epochs",100), imgsz=tc.get("imgsz",640),
        batch=tc.get("batch",16), lr0=tc.get("lr0",0.01), lrf=tc.get("lrf",0.01),
        momentum=tc.get("momentum",0.937), weight_decay=tc.get("weight_decay",0.0005),
        warmup_epochs=tc.get("warmup_epochs",3), device=tc.get("device","auto"),
        workers=tc.get("workers",4), project=tc.get("project","training/runs"),
        name=tc.get("name","baby_monitor"), exist_ok=True, pretrained=True,
        optimizer=tc.get("optimizer","SGD"), patience=tc.get("patience",50),
        save_period=tc.get("save_period",10), plots=True, verbose=True)

def validate(cfg, ds_yaml):
    best = (Path(cfg["training"].get("project","training/runs"))
            / cfg["training"].get("name","baby_monitor") / "weights/best.pt")
    if not best.exists(): print(f"[WARN] {best} not found"); return
    met = YOLO(str(best)).val(data=ds_yaml)
    print(f"[INFO] mAP50={met.box.map50:.4f}  mAP50-95={met.box.map:.4f}")

def export_model(cfg):
    best = (Path(cfg["training"].get("project","training/runs"))
            / cfg["training"].get("name","baby_monitor") / "weights/best.pt")
    if not best.exists(): print("[WARN] best.pt not found"); return
    YOLO(str(best)).export(format=cfg.get("export",{}).get("format","onnx"))

def copy_best(cfg):
    best = (Path(cfg["training"].get("project","training/runs"))
            / cfg["training"].get("name","baby_monitor") / "weights/best.pt")
    if best.exists():
        shutil.copy2(best, "best_custom.pt")
        print("[INFO] best_custom.pt ready -> set model.weights in config.yaml")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",        default="training/train_config.yaml")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--skip-train",    action="store_true")
    ap.add_argument("--export",        action="store_true")
    args = ap.parse_args()
    cfg  = load_cfg(args.config)
    ds   = make_dataset_yaml(cfg)
    if args.validate_only:   validate(cfg, ds)
    elif args.export:        export_model(cfg)
    else:
        if not args.skip_train: train(cfg, ds)
        validate(cfg, ds); copy_best(cfg)
        if cfg.get("export", {}).get("enabled"): export_model(cfg)
    print("[DONE]")
