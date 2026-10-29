# Задание 3. Подсчёт автомобилей и оценка загруженности парковки по видео с камеры наблюдения с использованием YOLOv8 и трекинга ByteTrack

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/farzXr/bigdata-practice/blob/main/task3_parking_yolo/parking.ipynb)

| | |
| --- | --- |
| **Вход** | Видео парковки с камеры наблюдения (демонстрационное видео уже в репозитории) |
| **Выход** | Число машин в кадре и в зоне парковки по времени, число уникальных машин за запись, загруженность в % |
| **Модель** | YOLOv8n, предобученная на COCO (car, motorcycle, bus, truck), без дообучения; трекинг ByteTrack |
| **Оценка** | Ручной подсчёт на 25 контрольных кадрах (готовая разметка в `ground_truth/`) → MAE, MAPE, доля кадров с ошибкой ≤ 1 машины |

## Файлы

| Файл | Назначение |
| --- | --- |
| `count_cars.py` | Обработка видео: детекция, трекинг, подсчёт, график, итоговая сводка |
| `evaluate.py` | Выгрузка контрольных кадров и расчёт метрик по ручной разметке |
| `parking.ipynb` | Тот же процесс по шагам с картинками — для Colab или Jupyter |
| `videos/parking.mp4` | Демонстрационное видео: 60 с, 1280×720, 15 к/с, ≈ 4 МБ |
| `ground_truth/parking.csv` | Ручной подсчёт машин в зоне на 25 контрольных кадрах этого видео |

## Быстрый запуск: всё одной кнопкой

Откройте `parking.ipynb` в Colab (кнопка выше) или в Jupyter и выберите **«Выполнить все»** (Run all).
Ничего класть и размечать не нужно: ноутбук сам поставит зависимости, возьмёт видео из `videos/`,
посчитает машины, подставит готовую разметку и выведет метрики и выводы.
Время: около минуты на GPU или Apple Silicon, несколько минут на процессоре.

## Видео

`videos/parking.mp4` — минутный фрагмент (с 4:17 до 5:17) записи
[BLK-HDPTZ12 Security Camera Parking Lot Surveillance Video](https://commons.wikimedia.org/wiki/File:BLK-HDPTZ12_Security_Camera_Parking_Lot_Surveillance_Video.webm)
с Wikimedia Commons (автор — Supercircuits, общественное достояние), уменьшенный до 1280×720 и 15 к/с.
Камера на этом отрезке неподвижна; за минуту одна машина выезжает с места, другая проезжает по проезду.

Зона парковки для этого видео — два ближних ряда: `--roi "0.07,0.6;0.96,0.6;0.96,1;0.07,1"`,
вместимость 12 мест (7 в первом ряду и 5 в нижнем). Машина считается в зоне, если в неё попадает центр её рамки.

Можно взять и своё видео со статичной камеры, например с [Pexels](https://www.pexels.com/search/videos/parking%20lot/)
по запросу «parking lot». Положите файл в `videos/`. Другие видео в git не попадают (см. `.gitignore`).

## Запуск локально

```bash
cd task3_parking_yolo
pip install -r requirements.txt
python count_cars.py --source videos/parking.mp4 --roi "0.07,0.6;0.96,0.6;0.96,1;0.07,1" --capacity 12
```

Веса `yolov8n.pt` (около 6 МБ) скачаются автоматически при первом запуске.
Устройство выбирается само: NVIDIA GPU → Apple Silicon (`mps`) → CPU.

Полезные параметры:

```bash
# своё видео: зона парковки (доли кадра, по часовой стрелке) и вместимость 40 мест
python count_cars.py --source videos/my_parking.mp4 --roi "0.05,0.35;0.95,0.35;0.95,0.98;0.05,0.98" --capacity 40

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

Для демонстрационного видео разметка готова — достаточно скопировать её и посчитать метрики:

```bash
python evaluate.py extract --source videos/parking.mp4 --n 25 --stride 2   # контрольные кадры для просмотра
cp ground_truth/parking.csv out/parking/ground_truth.csv
python evaluate.py score --source videos/parking.mp4
```

Для своего видео:

```bash
python evaluate.py extract --source videos/my_parking.mp4 --n 25   # кадры + заготовка ground_truth.csv
# посчитайте машины на кадрах из out/my_parking/frames/ и впишите в колонку true_count
python evaluate.py score --source videos/my_parking.mp4             # MAE, MAPE, график evaluation.png
```

Если запускали `count_cars.py` с `--stride 2`, передайте тот же `--stride 2` в `extract`,
чтобы контрольные кадры совпали с обработанными.
