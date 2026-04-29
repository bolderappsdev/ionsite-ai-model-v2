"""Dataset preparation helper. See --help for usage."""
import argparse, shutil, random
from pathlib import Path
import cv2

def extract_frames(video, out_dir, every=30, max_frames=500):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened(): print(f"[ERROR] {video}"); return []
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    saved, fi, name = [], 0, Path(video).stem
    while len(saved) < max_frames:
        ret, frame = cap.read()
        if not ret: break
        if fi % every == 0:
            p = Path(out_dir)/f"{name}_{fi:06d}.jpg"
            cv2.imwrite(str(p), frame); saved.append(p)
        fi += 1
    cap.release(); print(f"[INFO] {len(saved)} frames -> {out_dir}"); return saved

def split_dataset(imgs_dir, out, tr=0.7, vl=0.2):
    imgs = list(Path(imgs_dir).glob("*.jpg")) + list(Path(imgs_dir).glob("*.png"))
    random.shuffle(imgs)
    n_tr, n_vl = int(len(imgs)*tr), int(len(imgs)*vl)
    for split, files in [("train",imgs[:n_tr]),("val",imgs[n_tr:n_tr+n_vl]),("test",imgs[n_tr+n_vl:])]:
        ip = Path(out)/"images"/split; lp = Path(out)/"labels"/split
        ip.mkdir(parents=True, exist_ok=True); lp.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, ip/f.name)
            lf = lp/(f.stem+".txt")
            if not lf.exists(): lf.touch()
        print(f"  {split}: {len(files)}")

def create_structure(out):
    for s in ["train","val","test"]:
        (Path(out)/"images"/s).mkdir(parents=True, exist_ok=True)
        (Path(out)/"labels"/s).mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Structure created: {out}")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video"); ap.add_argument("--output", default="training/data")
    ap.add_argument("--every-n", type=int, default=30)
    ap.add_argument("--max-frames", type=int, default=500)
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--create-structure", action="store_true")
    args = ap.parse_args()
    if args.create_structure: create_structure(args.output)
    elif args.video:
        raw = str(Path(args.output)/"raw_frames")
        extract_frames(args.video, raw, args.every_n, args.max_frames)
        if args.split: split_dataset(raw, args.output)
    else: create_structure(args.output)
