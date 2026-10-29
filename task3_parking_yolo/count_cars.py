"""Подсчёт автомобилей и оценка загруженности парковки по видео (YOLOv8 + ByteTrack).

Для каждого обработанного кадра считает:
  in_frame      – сколько машин видно в кадре
  in_roi        – сколько машин внутри зоны парковки (если зона задана)
  unique_total  – сколько разных машин появилось с начала видео (по ID трекера; в зоне, если она задана)
  occupancy     – загруженность парковки, если указана вместимость

Результаты сохраняются в out/<имя_видео>/:
  annotated.mp4  – видео с рамками и ID машин
  counts.csv     – покадровая статистика
  occupancy.png  – график загруженности
  summary.json   – итоговые цифры

Примеры:
  python count_cars.py --source parking.mp4
  python count_cars.py --source parking.mp4 --stride 2 --capacity 40
  python count_cars.py --source parking.mp4 --roi "0.05,0.35;0.95,0.35;0.95,0.98;0.05,0.98"
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# классы COCO, которые считаем транспортом
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def pick_device(requested: str = "auto") -> str:
    """NVIDIA GPU → Apple Silicon (mps) → CPU."""
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def parse_roi(text: str | None, w: int, h: int):
    """'x1,y1;x2,y2;...' в долях кадра (0..1) → массив пикселей для cv2."""
    if not text:
        return None
    pts = [tuple(map(float, p.split(","))) for p in text.split(";")]
    # точки на самой границе кадра (1.0) сдвигаем внутрь, иначе контур зоны не виден
    return np.array([[min(int(x * w), w - 2), min(int(y * h), h - 2)] for x, y in pts], dtype=np.int32)


def inside(roi, x: float, y: float) -> bool:
    return roi is None or cv2.pointPolygonTest(roi, (float(x), float(y)), False) >= 0


def draw(frame, roi, zone_mask, boxes, label: str):
    """Кадр для annotated.mp4: всё вне зоны затемнено, рамки только у машин в зоне."""
    vis = frame.copy()
    if roi is not None:
        vis = cv2.convertScaleAbs(frame, alpha=0.35)
        cv2.copyTo(frame, zone_mask, vis)  # внутри зоны — исходная яркость
        cv2.polylines(vis, [roi], True, (0, 220, 255), 3)
        x0, y0 = roi[:, 0].min(), roi[:, 1].min()
        ty = int(y0) - 10 if y0 > 40 else int(y0) + 28  # подпись над зоной, если сверху есть место
        cv2.putText(vis, "PARKING ZONE (counted)", (int(x0) + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
    for (x1, y1, x2, y2), tid, cls in boxes:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(vis, p1, p2, (60, 200, 60), 2)
        text = f"#{tid} {VEHICLE_CLASSES.get(cls, '')}" if tid is not None else VEHICLE_CLASSES.get(cls, "")
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (p1[0], p1[1] - th - 6), (p1[0] + tw + 6, p1[1]), (60, 200, 60), -1)
        cv2.putText(vis, text, (p1[0] + 3, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.rectangle(vis, (10, 10), (26 + tw, 52), (0, 0, 0), -1)
    cv2.putText(vis, label, (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return vis


def run(args) -> Path:
    from ultralytics import YOLO

    src = Path(args.source)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"Не удалось открыть видео: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    roi = parse_roi(args.roi, w, h)
    zone_mask = None
    if roi is not None:
        zone_mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(zone_mask, [roi], 255)

    out_dir = Path(args.out) / src.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / args.stride, (w, h))

    device = pick_device(args.device)
    model = YOLO(args.model)
    print(f"Видео: {src.name}, {w}x{h}, {fps:.4g} к/с, {n_frames} кадров. Устройство: {device}")

    rows, seen_ids = [], set()
    frame_idx, t0 = -1, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % args.stride:
            continue

        res = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=list(VEHICLE_CLASSES),
                          conf=args.conf, imgsz=args.imgsz, device=device, verbose=False)[0]
        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else np.empty((0, 4))
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
        classes = boxes.cls.int().cpu().tolist() if len(boxes) else []

        zone_boxes = []
        for box, tid, cls in zip(xyxy, ids, classes):
            x1, y1, x2, y2 = box
            if inside(roi, (x1 + x2) / 2, (y1 + y2) / 2):
                zone_boxes.append((box, tid, cls))
                if tid is not None:
                    seen_ids.add(tid)
        in_roi = len(zone_boxes)

        rows.append({
            "frame": frame_idx,
            "time_s": round(frame_idx / fps, 2),
            "in_frame": len(xyxy),
            "in_roi": in_roi,
            "unique_total": len(seen_ids),
        })

        # кадр с рамками + счётчик
        label = ("Cars in zone: " if roi is not None else "Cars: ") + str(in_roi)
        if args.capacity:
            label += f" / {args.capacity}  ({in_roi / args.capacity:.0%})"
        writer.write(draw(frame, roi, zone_mask, zone_boxes, label))

        if len(rows) % 50 == 0:
            speed = len(rows) / (time.time() - t0)
            print(f"  кадр {frame_idx}/{n_frames}  машин: {in_roi}  ({speed:.1f} к/с)")

    cap.release()
    writer.release()
    elapsed = time.time() - t0

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("В видео не обработано ни одного кадра.")
    # сглаживание: медиана за ~1 секунду убирает мерцание детекций
    window = max(1, int(round(fps / args.stride)))
    df["in_roi_smooth"] = df["in_roi"].rolling(window, center=True, min_periods=1).median()
    if args.capacity:
        df["occupancy"] = (df["in_roi_smooth"] / args.capacity).clip(upper=1)
    df.to_csv(out_dir / "counts.csv", index=False)

    summary = {
        "video": src.name,
        "device": device,
        "frames_processed": len(df),
        "processing_fps": round(len(df) / elapsed, 2),
        "cars_mean": round(float(df["in_roi_smooth"].mean()), 2),
        "cars_max": int(df["in_roi"].max()),
        "cars_min": int(df["in_roi"].min()),
        "unique_vehicles_tracked": len(seen_ids),
        "capacity": args.capacity,
        "occupancy_mean": round(float(df["occupancy"].mean()), 3) if args.capacity else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot(df, args.capacity, out_dir / "occupancy.png", src.name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Результаты: {out_dir.resolve()}")
    return out_dir


def plot(df: pd.DataFrame, capacity: int | None, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df["time_s"], df["in_roi"], color="#9aa0a6", lw=0.8, label="по кадрам")
    ax.plot(df["time_s"], df["in_roi_smooth"], color="#1f5fa8", lw=2, label="сглажено (≈1 с)")
    if capacity:
        ax.axhline(capacity, color="#d64545", ls="--", lw=1, label=f"вместимость {capacity}")
        sec = ax.secondary_yaxis("right", functions=(lambda v: v / capacity * 100, lambda p: p * capacity / 100))
        sec.set_ylabel("Загруженность, %")
    ax.set_xlabel("Время видео, с")
    ax.set_ylabel("Автомобилей в зоне")
    ax.set_title(f"Загруженность парковки — {title}")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="путь к видеофайлу")
    p.add_argument("--model", default="yolov8n.pt", help="веса YOLO (скачаются автоматически)")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--conf", type=float, default=0.3, help="порог уверенности детектора")
    p.add_argument("--imgsz", type=int, default=960, help="размер входа сети; больше — точнее для мелких машин")
    p.add_argument("--stride", type=int, default=1, help="обрабатывать каждый N-й кадр (ускорение)")
    p.add_argument("--roi", default=None, help="зона парковки: 'x1,y1;x2,y2;...' в долях кадра")
    p.add_argument("--capacity", type=int, default=None, help="число мест на парковке")
    p.add_argument("--out", default="out", help="папка для результатов")
    run(p.parse_args())


if __name__ == "__main__":
    main()
