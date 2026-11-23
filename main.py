"""
Перед запуском установите зависимости и токен:

pip install huggingface_hub pillow scikit-learn
$env:HF_TOKEN="hf_ваш_токен"
python main.py
"""

import os
import csv
import sys
from pathlib import Path

from PIL import Image
from huggingface_hub import InferenceClient
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ---------- НАСТРОЙКИ ----------
DATASET_DIR = Path("dataset")
LABELS_CSV = DATASET_DIR / "labels_auto.csv"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

ANNOTATOR_MODEL = "openai/clip-vit-base-patch32"
MODEL_1 = "openai/clip-vit-base-patch32"
MODEL_2 = "google/siglip-base-patch16-224"

CANDIDATE_LABELS = ["military aircraft", "civilian aircraft"]
LABEL_MAP = {"military aircraft": "military", "civilian aircraft": "civilian"}

CONFIDENCE_THRESHOLD = 0.60

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("Ошибка: не задан HF_TOKEN. Установите переменную окружения HF_TOKEN.")
    sys.exit(1)

client = InferenceClient(provider="hf-inference", token=HF_TOKEN)


# ---------- ВСПОМОГАТЕЛЬНЫЕ ----------
def list_images():
    if not DATASET_DIR.exists():
        return []
    return sorted(
        p for p in DATASET_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    )


def zero_shot(image_path):
    """Запрос к Inference API. Возвращает (label, confidence)."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        result = client.zero_shot_image_classification(
            image=img,
            model=ANNOTATOR_MODEL if image_path.parent == DATASET_DIR else None,
            candidate_labels=CANDIDATE_LABELS,
        )
    best = max(result, key=lambda r: r["score"])
    return LABEL_MAP.get(best["label"], "uncertain"), float(best["score"])


def predict_with_model(model_name, image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        result = client.zero_shot_image_classification(
            image=img,
            model=model_name,
            candidate_labels=CANDIDATE_LABELS,
        )
    best = max(result, key=lambda r: r["score"])
    return LABEL_MAP.get(best["label"], "uncertain")


def print_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=["military", "civilian"])
    print()
    print(f"Матрица ошибок ({title}):")
    print("Истина \\ Предсказание | военный | гражданский")
    print(f"военный              | {cm[0][0]:>7} | {cm[0][1]:>10}")
    print(f"гражданский          | {cm[1][0]:>7} | {cm[1][1]:>10}")
    return cm


# ---------- ШАГ 1: РАЗМЕТКА ----------
def step_annotate():
    images = list_images()
    if not images:
        print("В dataset/ нет изображений.")
        sys.exit(1)

    print("=== Шаг 1: автоматическая разметка датасета (Inference API) ===")
    print(f"Найдено {len(images)} изображений. Модель-аннотатор: {ANNOTATOR_MODEL}")
    print("Автоматическая разметка...")

    rows = []
    for i, path in enumerate(images, 1):
        label, conf = zero_shot(path)
        if conf < CONFIDENCE_THRESHOLD:
            label = "uncertain"
        rows.append({"filename": path.name, "label": label, "confidence": f"{conf:.3f}"})
        if i % 8 == 0 or i == len(images):
            print(f"  Обработано {i}/{len(images)}")

    with LABELS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "confidence"])
        writer.writeheader()
        writer.writerows(rows)

    mil = sum(1 for r in rows if r["label"] == "military")
    civ = sum(1 for r in rows if r["label"] == "civilian")
    unc = sum(1 for r in rows if r["label"] == "uncertain")

    print()
    print(f"Разметка сохранена в {LABELS_CSV}")
    print(f"Итого: военных={mil}, гражданских={civ}, спорных={unc}")


# ---------- ШАГ 2: СРАВНЕНИЕ ----------
def step_compare():
    if not LABELS_CSV.exists():
        print("Файл labels_auto.csv не найден. Сначала выполните шаг 1.")
        sys.exit(1)

    entries = []
    with LABELS_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["label"] in ("military", "civilian"):
                entries.append((r["filename"], r["label"]))

    if not entries:
        print("В labels_auto.csv нет строк military/civilian.")
        sys.exit(1)

    mil = sum(1 for _, l in entries if l == "military")
    civ = sum(1 for _, l in entries if l == "civilian")

    print("=== Шаг 2: чтение разметки и сравнение моделей (Inference API) ===")
    print(f"Загружено {len(entries)} изображений (военных: {mil}, гражданских: {civ})")

    results = {}
    for model_name in (MODEL_1, MODEL_2):
        print()
        print(f"Запрос к модели: {model_name} (Inference API)")
        y_true, y_pred, errors = [], [], []
        for i, (fname, true_label) in enumerate(entries, 1):
            pred = predict_with_model(model_name, DATASET_DIR / fname)
            y_true.append(true_label)
            y_pred.append(pred)
            if pred != true_label:
                errors.append((fname, true_label, pred))
            if i % 8 == 0 or i == len(entries):
                print(f"  Обработано {i}/{len(entries)}")

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label="military", zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label="military", zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label="military", zero_division=0)

        print()
        print(f"===== {model_name} =====")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-score : {f1:.4f}")

        print_confusion(y_true, y_pred, model_name)

        if errors:
            print("Ошибочные файлы:")
            for fname, t, p in errors:
                print(f"  {fname}: истина={t}, предсказано={p}")
        else:
            print("Ошибок нет.")

        results[model_name] = {
            "acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "y_true": y_true, "y_pred": y_pred, "errors": errors,
        }

    print()
    print("=" * 50)
    print("Сводное сравнение моделей (Inference API)")
    print("=" * 50)
    header = f"{'Метрика':<12}{MODEL_1:<40}{MODEL_2:<40}"
    print(header)
    print("-" * len(header))
    for key, name in (("acc", "Accuracy"), ("prec", "Precision"), ("rec", "Recall"), ("f1", "F1")):
        r1 = results[MODEL_1][key]
        r2 = results[MODEL_2][key]
        print(f"{name:<12}{r1:<40.4f}{r2:<40.4f}")

    print()
    print("=" * 50)
    print("Полные матрицы ошибок")
    print("=" * 50)
    for model_name in (MODEL_1, MODEL_2):
        r = results[model_name]
        cm = confusion_matrix(r["y_true"], r["y_pred"], labels=["military", "civilian"])
        print()
        print(f"Матрица ошибок ({model_name}):")
        print("Истина \\ Предсказание | военный | гражданский")
        print(f"военный              | {cm[0][0]:>7} | {cm[0][1]:>10}")
        print(f"гражданский          | {cm[1][0]:>7} | {cm[1][1]:>10}")


# ---------- MAIN ----------
def main():
    if not LABELS_CSV.exists():
        step_annotate()
    else:
        step_compare()


if __name__ == "__main__":
    main()
