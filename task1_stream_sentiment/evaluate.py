"""Оценка качества разметки: сравниваем предсказание модели с оценкой автора.

Звёзды служат «слабой» разметкой: 1–2 ★ → negative, 3 ★ → neutral, 4–5 ★ → positive.
Сохраняет отчёт в data/evaluation.md и матрицу ошибок в data/confusion_matrix.png.

Запуск:  python evaluate.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)

import config
import storage

LABELS = ["negative", "neutral", "positive"]


def stars_to_label(score: int) -> str:
    return "negative" if score <= 2 else "neutral" if score == 3 else "positive"


def main():
    df = storage.load_all(config.DB_PATH)
    df = df.dropna(subset=["score"])
    if df.empty:
        raise SystemExit("В базе нет размеченных отзывов с оценкой.")
    y_true = df["score"].astype(int).map(stars_to_label)
    y_pred = df["label"]

    # бинарный вариант — только явный позитив/негатив, как в задании
    binary = y_true.isin(["negative", "positive"]) & y_pred.isin(["negative", "positive"])

    report = classification_report(y_true, y_pred, labels=LABELS, digits=3, zero_division=0)
    lines = [
        "# Оценка качества определения тональности",
        "",
        f"Отзывов: {len(df)}",
        f"Accuracy (3 класса): {accuracy_score(y_true, y_pred):.3f}",
        f"Macro-F1 (3 класса): {f1_score(y_true, y_pred, labels=LABELS, average='macro'):.3f}",
        f"Accuracy (только позитив/негатив, {binary.sum()} отзывов): "
        f"{accuracy_score(y_true[binary], y_pred[binary]):.3f}",
        "",
        "```",
        report,
        "```",
        "",
        "Распределение по классам (модель):",
        "",
        y_pred.value_counts().to_frame("отзывов").to_markdown(),
    ]
    out = config.DATA_DIR / "evaluation.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=LABELS, normalize="true", values_format=".0%",
        cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_xlabel("Модель")
    ax.set_ylabel("Оценка автора (звёзды)")
    ax.set_title("Матрица ошибок (по строкам)")
    fig.tight_layout()
    fig.savefig(config.DATA_DIR / "confusion_matrix.png", dpi=150)
    print(f"\nОтчёт: {out}\nМатрица: {config.DATA_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
