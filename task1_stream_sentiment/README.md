# Задание 1. Потоковый анализ тональности отзывов о приложении «Госуслуги» в Google Play с использованием Apache Kafka

Система почти в реальном времени (near real-time) собирает новые отзывы
о приложении из Google Play, определяет их тональность готовой нейросетью
и показывает на дашборде соотношение положительных и отрицательных отзывов.

## Архитектура

```
Google Play ──► producer.py ──► Kafka (топик reviews.raw) ──► consumer.py ──► SQLite ──► dashboard.py
               (сбор, опрос          (буфер, хранение           (rubert-tiny-                (Streamlit,
                раз в N секунд)       сообщений)                  sentiment)                   автообновление)
```

| Компонент | Файл | Что делает |
| --- | --- | --- |
| Брокер сообщений | `docker-compose.yml` | Kafka 3.8 в режиме KRaft, порт 19092 |
| Продюсер | `producer.py` | Забирает отзывы через `google-play-scraper` и публикует их в Kafka |
| Консьюмер | `consumer.py` | Читает топик пачками, размечает тональность, пишет в базу |
| Модель | `sentiment.py` | [`cointegrated/rubert-tiny-sentiment-balanced`](https://huggingface.co/cointegrated/rubert-tiny-sentiment-balanced), классы negative / neutral / positive |
| Хранилище | `storage.py` | SQLite, идемпотентная запись (`INSERT OR IGNORE` по `review_id`) |
| Дашборд | `dashboard.py` | Доли классов во времени, частые слова в негативе, последние отзывы |
| Оценка | `evaluate.py` | Сравнение ответов модели с оценкой автора в звёздах |

**Надёжность доставки.** Продюсер отправляет с `acks=all` и идемпотентностью.
Консьюмер фиксирует смещения только после записи в базу (at-least-once),
а повторно доставленные сообщения отсекает первичный ключ. Если остановить
консьюмер, отзывы будут копиться в Kafka и обработаются после перезапуска.

## Установка

Нужны Python 3.10+ и [Docker Desktop](https://www.docker.com/products/docker-desktop/).

```bash
cd task1_stream_sentiment
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d          # запускаем Kafka
```

При первом запуске консьюмер скачает модель (около 30 МБ) с HuggingFace.

## Запуск

Каждую команду запускайте в отдельном терминале с активированным окружением.

```bash
# Терминал 1 — консьюмер (анализ)
python consumer.py

# Терминал 2 — дашборд, откроется в браузере на http://localhost:8501
streamlit run dashboard.py

# Терминал 3 — продюсер в одном из режимов (см. ниже)
```

### Режим live — реальный поток

```bash
python producer.py live --interval 120 --backfill 50
```

Раз в 2 минуты продюсер запрашивает последние 100 отзывов и отправляет в Kafka
только те, которых ещё не видел (список хранится в `data/seen_ids.txt`).
При первом запуске отправляются 50 последних отзывов, чтобы дашборд не был пустым.

### Режим replay — ускоренное воспроизведение (для демонстрации)

Новые отзывы о «Госуслугах» появляются раз в несколько минут, поэтому
для показа удобно воспроизвести накопленную историю как поток:

```bash
python producer.py collect --count 5000   # один раз скачать историю в data/reviews_dump.csv
python producer.py replay --rate 10       # подавать в Kafka по 10 отзывов в секунду
```

Отзывы идут в хронологическом порядке, и на дашборде видно, как график строится
на глазах. Так тестируют любые потоковые системы.

### Оценка качества

```bash
python evaluate.py
```

Звёзды автора используются как слабая разметка: 1–2 ★ — негатив, 3 ★ — нейтрально,
4–5 ★ — позитив. Скрипт считает accuracy и F1 и сохраняет `data/evaluation.md`
и `data/confusion_matrix.png`.

## Настройки

Всё задаётся в `config.py` или переменными окружения: другое приложение
(`APP_ID=ru.sberbankmobile`), адрес Kafka (`KAFKA_BOOTSTRAP`), модель (`MODEL_NAME`).

Kafka слушает порт **19092**, а не стандартный 9092: тот часто занят другими проектами.
Если занят и 19092, задайте другой порт переменной `KAFKA_PORT` — одинаково для
`docker compose` и для скриптов:

```bash
export KAFKA_PORT=29092        # Windows (PowerShell): $env:KAFKA_PORT="29092"
docker compose up -d
python consumer.py
```

## Источник данных

Отзывы в Google Play публичные. `google-play-scraper` — неофициальная библиотека,
которая разбирает публичные страницы магазина. Данные используются только
в учебных целях, а продюсер делает паузы между запросами, чтобы не нагружать источник.
Если библиотека перестанет работать после изменений на стороне Google,
режим replay продолжит работать на уже сохранённом CSV.

## Остановка

`Ctrl+C` в терминалах, затем `docker compose down` (данные Kafka сохранятся в volume;
`docker compose down -v` удалит и их).
