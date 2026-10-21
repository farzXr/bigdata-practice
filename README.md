# Практические задания по дисциплине «Большие данные»

**Студент:** Фарзалиев Ренат Адилевич, магистрант СГТУ имени Гагарина Ю. А.

| № | Направление | Тема | Где запускать |
| --- | --- | --- | --- |
| 1 | Потоковая обработка | [Потоковый анализ тональности отзывов о мобильном приложении «Госуслуги» в Google Play с использованием Apache Kafka](task1_stream_sentiment/) | Локально (Docker + Python) |
| 2 | Машинное обучение | [Определение присутствия людей в помещении по показаниям датчиков CO₂, температуры, влажности и освещённости](task2_occupancy_ml/) | Локально или [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/farzXr/bigdata-practice/blob/main/task2_occupancy_ml/occupancy.ipynb) |
| 3 | Визуальная информация | [Подсчёт автомобилей и оценка загруженности парковки по видео с камеры наблюдения с использованием YOLOv8 и трекинга ByteTrack](task3_parking_yolo/) | Локально или [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/farzXr/bigdata-practice/blob/main/task3_parking_yolo/parking.ipynb) |

## Кратко о каждом задании

**1. Поток отзывов.** Отзывы из Google Play поступают в Kafka, консьюмер определяет тональность
нейросетью `rubert-tiny-sentiment-balanced`, а дашборд на Streamlit в реальном времени
показывает долю положительных и отрицательных отзывов. Есть живой режим (опрос источника)
и режим ускоренного воспроизведения для демонстрации.

**2. Занятость помещения.** Бинарная классификация на открытом датасете UCI Occupancy Detection:
сравнение логистической регрессии, Random Forest и градиентного бустинга, важность признаков
и эксперимент без датчика освещённости.

**3. Загруженность парковки.** Предобученная YOLOv8n находит машины на видео, ByteTrack
отслеживает каждую машину между кадрами. На выходе размеченное видео, график загруженности
и метрики точности по ручному подсчёту.

## Требования

- Python 3.10+
- Docker Desktop — только для задания 1
- GPU не обязателен: всё работает на процессоре, задание 3 — медленнее

У каждого задания свой `requirements.txt` и подробный `README.md` с инструкцией по запуску.

## Структура

```
bigdata-practice/
├── task1_stream_sentiment/   # producer.py, consumer.py, dashboard.py, docker-compose.yml
├── task2_occupancy_ml/       # occupancy.ipynb
└── task3_parking_yolo/       # count_cars.py, evaluate.py, parking.ipynb
```
