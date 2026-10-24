import csv
from pathlib import Path
from PIL import Image
import torch
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ================= НАСТРОЙКИ =================
DATA_DIR = Path("dataset")
LABELS_CSV = DATA_DIR / "labels_auto.csv"

# Кандидатные метки для zero-shot
CANDIDATE_LABELS = ["a military aircraft", "a civilian aircraft"]

# Модель-аннотатор (ей размечаем) и две модели для сравнения
ANNOTATOR_MODEL = "openai/clip-vit-base-patch32"
MODEL_1 = "openai/clip-vit-base-patch32"
MODEL_2 = "google/siglip-base-patch16-224"

IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
BATCH_SIZE = 8

# Порог уверенности: если модель не уверена — помечаем как "uncertain"
CONFIDENCE_THRESHOLD = 0.60
# ============================================


def collect_images(folder: Path):
    """Все изображения из папки (без рекурсии), отсортированные по имени."""
    files = []
    for ext in IMAGE_EXTS:
        files.extend(folder.glob(ext))
    return sorted(set(files), key=lambda p: p.name.lower())


def classify_batch(classifier, images, labels=CANDIDATE_LABELS):
    """Пакетная zero-shot классификация. Возвращает список (label, score)."""
    results_out = []
    total = len(images)
    for start in range(0, total, BATCH_SIZE):
        batch = images[start:start + BATCH_SIZE]
        results = classifier(batch, candidate_labels=labels)
        for res in results:
            top = res[0]
            results_out.append((top["label"], float(top["score"])))
        print(f"  Обработано {min(start + BATCH_SIZE, total)}/{total}")
    return results_out


def auto_label_dataset():
    """Автоматически размечает все фото из dataset/ через ANNOTATOR_MODEL.
       Сохраняет результат в labels_auto.csv: filename,label,confidence
    """
    files = collect_images(DATA_DIR)
    if not files:
        raise SystemExit(f"В папке {DATA_DIR} не найдено изображений.")

    print(f"Найдено {len(files)} изображений. Загружаем модель-аннотатор: {ANNOTATOR_MODEL}")
    classifier = pipeline(
        "zero-shot-image-classification",
        model=ANNOTATOR_MODEL,
        device=0 if torch.cuda.is_available() else -1,
    )

    images = [Image.open(f).convert("RGB") for f in files]
    print("Автоматическая разметка...")
    results = classify_batch(classifier, images)

    rows = []
    for f, (label, score) in zip(files, results):
        if score < CONFIDENCE_THRESHOLD:
            cls = "uncertain"
        elif label == CANDIDATE_LABELS[0]:
            cls = "military"
        else:
            cls = "civilian"
        rows.append((f.name, cls, f"{score:.4f}"))
        print(f"  {f.name:<60} -> {cls:<10} (уверенность {score:.3f})")

    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "label", "confidence"])
        writer.writerows(rows)

    print(f"\nРазметка сохранена в {LABELS_CSV}")
    n_mil = sum(1 for r in rows if r[1] == "military")
    n_civ = sum(1 for r in rows if r[1] == "civilian")
    n_unc = sum(1 for r in rows if r[1] == "uncertain")
    print(f"Итого: военных={n_mil}, гражданских={n_civ}, спорных={n_unc}")
    print("Проверьте файл labels_auto.csv и поправьте спорные/ошибочные метки,")
    print("затем запустите скрипт снова — он прочитает CSV и посчитает метрики.")


def load_labels_from_csv():
    """Читает labels_auto.csv, возвращает images, true_labels, names.
       Строки с 'uncertain' пропускаются.
    """
    if not LABELS_CSV.is_file():
        return None, None, None

    images, true_labels, names = [], [], []
    with open(LABELS_CSV, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            label = (row.get("label") or "").strip().lower()
            fname = (row.get("filename") or "").strip()
            if label not in ("military", "civilian"):
                continue
            path = DATA_DIR / fname
            if not path.is_file():
                print(f"  [пропуск] файл не найден: {fname}")
                continue
            images.append(Image.open(path).convert("RGB"))
            true_labels.append(1 if label == "military" else 0)
            names.append(fname)
    return images, true_labels, names


def predict_with_model(model_name, images):
    print(f"\nЗагрузка модели: {model_name}")
    classifier = pipeline(
        "zero-shot-image-classification",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1,
    )
    results = classify_batch(classifier, images)
    preds = [1 if lbl == CANDIDATE_LABELS[0] else 0 for lbl, _ in results]
    return preds


def compute_metrics(true_labels, pred_labels, model_name, names=None):
    acc = accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels, zero_division=0)
    rec = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)

    print(f"\n===== {model_name} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    cm = confusion_matrix(true_labels, pred_labels)
    print(f"Матрица ошибок:\n{cm}")
    print("  (строки: истина [военный, гражданский], столбцы: предсказание [военный, гражданский])")

    if names:
        errors = [(names[i], true_labels[i], pred_labels[i])
                  for i in range(len(true_labels)) if true_labels[i] != pred_labels[i]]
        if errors:
            print("Ошибочные файлы:")
            for n, t, p in errors:
                print(f"  {n}: истина={'военный' if t==1 else 'гражданский'}, "
                      f"предсказано={'военный' if p==1 else 'гражданский'}")
        else:
            print("Ошибок нет.")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


if __name__ == "__main__":
    if not LABELS_CSV.is_file():
        print("=== Шаг 1: автоматическая разметка датасета ===")
        auto_label_dataset()
        print("\nТеперь просмотрите labels_auto.csv. Если всё устраивает —")
        print("запустите скрипт ещё раз для сравнения моделей.")
        raise SystemExit(0)

    print("=== Шаг 2: чтение разметки и сравнение моделей ===")
    images, true_labels, names = load_labels_from_csv()
    if not images:
        raise SystemExit("В labels_auto.csv нет валидных строк (military/civilian).")

    n_mil = sum(true_labels)
    n_civ = len(true_labels) - n_mil
    print(f"Загружено {len(images)} изображений (военных: {n_mil}, гражданских: {n_civ})")

    preds_1 = predict_with_model(MODEL_1, images)
    metrics_1 = compute_metrics(true_labels, preds_1, MODEL_1, names)

    preds_2 = predict_with_model(MODEL_2, images)
    metrics_2 = compute_metrics(true_labels, preds_2, MODEL_2, names)

    print("\n" + "=" * 50)
    print("Сводное сравнение моделей")
    print("=" * 50)
    print(f"{'Метрика':<12} {MODEL_1:<35} {MODEL_2:<35}")
    print("-" * 82)
    for key in ["accuracy", "precision", "recall", "f1"]:
        print(f"{key.capitalize():<12} {metrics_1[key]:<35.4f} {metrics_2[key]:<35.4f}")