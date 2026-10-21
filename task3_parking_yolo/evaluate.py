"""Оценка точности подсчёта: сравнение с ручным подсчётом на контрольных кадрах.

Шаг 1. Выгрузить контрольные кадры и заготовку таблицы:
    python evaluate.py extract --source parking.mp4 --n 25
  → out/<видео>/frames/frame_000123.jpg и out/<видео>/ground_truth.csv
    Откройте кадры, посчитайте машины в зоне вручную и впишите число в колонку true_count.

Шаг 2. Посчитать метрики:
    python evaluate.py score --source parking.mp4
  → MAE, MAPE, доля кадров с ошибкой ≤ 1 машины, график out/<видео>/evaluation.png
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def extract(args):
    src = Path(args.source)
    out_dir = Path(args.out) / src.stem
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:  # некоторые контейнеры не сообщают число кадров — считаем вручную
        while cap.grab():
            total += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if total <= 0:
        raise SystemExit(f"Не удалось прочитать кадры из видео: {src}")
    stride = args.stride
    # берём кадры, кратные stride, чтобы они совпадали с обработанными в count_cars.py
    candidates = np.arange(0, total, stride)
    picks = np.unique(candidates[np.linspace(0, len(candidates) - 1, args.n).astype(int)])
    for old in frames_dir.glob("frame_*.jpg"):  # кадры от прошлого запуска с другими --n/--stride
        old.unlink()
    for idx in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(frames_dir / f"frame_{idx:06d}.jpg"), frame)
    cap.release()
    gt = out_dir / "ground_truth.csv"
    if gt.exists() and list(pd.read_csv(gt)["frame"]) != list(picks):
        # набор кадров изменился: старую разметку не теряем, но и не оставляем рассинхронизированной
        backup = gt.with_name("ground_truth.bak.csv")
        gt.replace(backup)
        print(f"Набор кадров изменился, прежняя разметка сохранена в {backup}")
    if not gt.exists():
        pd.DataFrame({"frame": picks, "true_count": ""}).to_csv(gt, index=False)
    print(f"Кадры: {frames_dir}\nЗаполните true_count в {gt}")


def score(args):
    src = Path(args.source)
    out_dir = Path(args.out) / src.stem
    pred = pd.read_csv(out_dir / "counts.csv")
    gt_path = out_dir / "ground_truth.csv"
    gt = pd.read_csv(gt_path).dropna(subset=["true_count"])
    if gt.empty:
        raise SystemExit(f"Колонка true_count не заполнена. Посчитайте машины на кадрах и впишите в {gt_path}")
    df = gt.merge(pred[["frame", "in_roi"]], on="frame", how="inner")
    if df.empty:
        raise SystemExit("Нет совпадающих кадров. Используйте одинаковый --stride в count_cars.py и evaluate.py")
    if len(df) < len(gt):
        print(f"Внимание: не учтено размеченных кадров, которых нет в counts.csv: {len(gt) - len(df)}. "
              "Проверьте, что --stride в count_cars.py и evaluate.py одинаковый.")

    err = df["in_roi"] - df["true_count"]
    nonzero = df["true_count"] > 0
    metrics = {
        "frames": int(len(df)),
        "MAE": round(float(err.abs().mean()), 3),
        # MAPE считается только по кадрам, где машины есть; если таких нет — null
        "MAPE_%": round(float((err[nonzero].abs() / df.loc[nonzero, "true_count"]).mean() * 100), 2)
        if nonzero.any() else None,
        "bias": round(float(err.mean()), 3),  # < 0 — модель недосчитывает
        "within_1_%": round(float((err.abs() <= 1).mean() * 100), 1),
        "exact_%": round(float((err == 0).mean() * 100), 1),
    }
    (out_dir / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    lim = max(df["true_count"].max(), df["in_roi"].max()) + 1
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.scatter(df["true_count"], df["in_roi"], color="#1f5fa8", alpha=0.7)
    ax.set_xlabel("Ручной подсчёт")
    ax.set_ylabel("YOLOv8")
    ax.set_title(f"MAE = {metrics['MAE']}, ошибка ≤ 1 в {metrics['within_1_%']}% кадров")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    fig.tight_layout()
    fig.savefig(out_dir / "evaluation.png", dpi=150)
    print(f"График: {out_dir / 'evaluation.png'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("extract", extract), ("score", score)):
        s = sub.add_parser(name)
        s.add_argument("--source", required=True)
        s.add_argument("--out", default="out")
        s.set_defaults(func=fn)
        if name == "extract":
            s.add_argument("--n", type=int, default=25, help="сколько контрольных кадров")
            s.add_argument("--stride", type=int, default=1, help="тот же stride, что в count_cars.py")
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
