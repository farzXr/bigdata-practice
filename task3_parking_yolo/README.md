# Задание 3. Подсчёт автомобилей и оценка загруженности парковки по видео с камеры наблюдения с использованием YOLOv8 и трекинга ByteTrack

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/farzXr/bigdata-practice/blob/main/task3_parking_yolo/parking.ipynb)

| | |
| --- | --- |
| **Вход** | Видео парковки с камеры наблюдения или дрона |
| **Выход** | Число машин в кадре и в зоне парковки по времени, число уникальных машин за запись, загруженность в % |
| **Модель** | YOLOv8n, предобученная на COCO (car, motorcycle, bus, truck), без дообучения; трекинг ByteTrack |
| **Оценка** | Ручной подсчёт на 20–30 контрольных кадрах → MAE, MAPE, доля кадров с ошибкой ≤ 1 машины |

## Файлы

| Файл | Назначение |
| --- | --- |
| `count_cars.py` | Обработка видео: детекция, трекинг, подсчёт, график, итоговая сводка |
| `evaluate.py` | Выгрузка контрольных кадров и расчёт метрик по ручной разметке |
| `parking.ipynb` | Тот же процесс по шагам с картинками — для Colab или Jupyter |

## Видео

Подойдёт любая запись со статичной камеры, где видны машины: например, с
[Pexels](https://www.pexels.com/search/videos/parking%20lot/) по запросам «parking lot»
или «aerial parking», либо последовательности из датасета
[VisDrone](https://github.com/VisDrone/VisDrone-Dataset) (вид с дрона).
Достаточно 30–60 секунд. Положите файл в `videos/`. Видео в git не попадают
(см. `.gitignore`), потому что они большие.

## Запуск локально

```bash
cd task3_parking_yolo
pip install -r requirements.txt
python count_cars.py --source videos/parking.mp4
```

Веса `yolov8n.pt` (около 6 МБ) скачаются автоматически при первом запуске.
Устройство выбирается само: NVIDIA GPU → Apple Silicon (`mps`) → CPU.

Полезные параметры:

```bash
# только зона парковки (доли кадра, по часовой стрелке) и вместимость 40 мест
python count_cars.py --source videos/parking.mp4 --roi "0.05,0.35;0.95,0.35;0.95,0.98;0.05,0.98" --capacity 40

# быстрее на процессоре: каждый 2-й кадр и меньший вход сети
python count_cars.py --source videos/parking.mp4 --stride 2 --imgsz 640

# точнее для мелких машин: модель побольше и вход крупнее
python count_cars.py --source videos/parking.mp4 --model yolov8s.pt --imgsz 1280
```

Результаты появятся в `out/<имя_видео>/`: `annotated.mp4`, `counts.csv`,
`occupancy.png`, `summary.json`.

Ориентировочная скорость YOLOv8n на видео 720p: GPU NVIDIA — 50–150 кадров/с,
Apple M1–M3 — 20–40, обычный процессор — 5–15.

## Оценка точности

```bash
python evaluate.py extract --source videos/parking.mp4 --n 25   # кадры + заготовка ground_truth.csv
# посчитайте машины на кадрах из out/parking/frames/ и впишите в колонку true_count
python evaluate.py score --source videos/parking.mp4             # MAE, MAPE, график evaluation.png
```

Если запускали `count_cars.py` с `--stride 2`, передайте тот же `--stride 2` в `extract`,
чтобы контрольные кадры совпали с обработанными.
