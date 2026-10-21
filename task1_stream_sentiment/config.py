"""Общие настройки потоковой системы. Любой параметр можно переопределить
переменной окружения с тем же именем."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Kafka
# порт должен совпадать с docker-compose.yml; 9092 не используем — он часто занят
KAFKA_PORT = os.getenv("KAFKA_PORT", "19092")
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", f"localhost:{KAFKA_PORT}")
TOPIC_REVIEWS = os.getenv("TOPIC_REVIEWS", "reviews.raw")
CONSUMER_GROUP = os.getenv("CONSUMER_GROUP", "sentiment-analyzer")

# Источник: приложение в Google Play (id берётся из адреса страницы
# play.google.com/store/apps/details?id=<APP_ID>)
APP_ID = os.getenv("APP_ID", "ru.rostel")  # «Госуслуги»
APP_LANG = os.getenv("APP_LANG", "ru")
APP_COUNTRY = os.getenv("APP_COUNTRY", "ru")

# Модель тональности (HuggingFace). Классы: negative / neutral / positive
MODEL_NAME = os.getenv("MODEL_NAME", "cointegrated/rubert-tiny-sentiment-balanced")

# Хранилище результатов
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "reviews.db"))
REPLAY_CSV = Path(os.getenv("REPLAY_CSV", DATA_DIR / "reviews_dump.csv"))
SEEN_IDS_PATH = Path(os.getenv("SEEN_IDS_PATH", DATA_DIR / "seen_ids.txt"))
