"""Train the knife + gun detector and install it into VigiLight.

1. Get a dataset in YOLO format (see README, "Train the knife and gun model").
   You need a data.yaml listing your classes (for example: gun, knife).
2. Train (a free Colab GPU takes roughly 30-60 minutes):

       python train_weapons.py --data path/to/data.yaml

3. The best weights are copied to weights/weapons.pt. VigiLight picks that file up
   automatically on the next start; every class in it is treated as a weapon.
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--base", default="yolo11s.pt", help="yolo11n.pt = fastest, yolo11s.pt = more accurate (default)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=960, help="keep equal to WEAPON_IMGSZ in .env")
    ap.add_argument("--batch", type=int, default=-1, help="-1 = automatic")
    ap.add_argument("--device", default="", help="0 for the first GPU, cpu to force CPU")
    args = ap.parse_args()

    kw = dict(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, patience=15,
              project="runs", name="weapons", exist_ok=True)
    if args.device:
        kw["device"] = args.device
    YOLO(args.base).train(**kw)

    best = Path("runs/weapons/weights/best.pt")
    if not best.exists():
        raise SystemExit("Training finished but best.pt was not found in runs/weapons/weights/")
    val_kw = {"data": args.data, "imgsz": args.imgsz}
    if args.device:
        val_kw["device"] = args.device
    m = YOLO(str(best)).val(**val_kw)
    print(f"\nValidation: precision {m.box.mp:.2f} | recall {m.box.mr:.2f} | mAP@50 {m.box.map50:.2f} | mAP@50-95 {m.box.map:.2f}")

    dest = Path(__file__).resolve().parent / "weights" / "weapons.pt"
    dest.parent.mkdir(exist_ok=True)
    shutil.copy(best, dest)
    print(f"Installed {dest}. Restart VigiLight to use it.")


if __name__ == "__main__":
    main()
