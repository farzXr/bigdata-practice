"""Консьюмер: читает отзывы из Kafka, определяет тональность, сохраняет в SQLite.

Смещения (offsets) фиксируются в Kafka только ПОСЛЕ успешной записи в базу.
Если консьюмер упадёт посреди пачки, после перезапуска он перечитает её
заново (at-least-once), а дубли отсечёт первичный ключ в базе.

Запуск:  python consumer.py
         python consumer.py --from-beginning   # перечитать топик с начала
"""
import argparse
import json
import signal
import sys
from datetime import datetime, timezone

from confluent_kafka import Consumer, KafkaError

import config
import sentiment
import storage

_running = True


def _stop(*_):
    global _running
    _running = False
    print("\nОстановка...")


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)


def make_consumer(from_beginning: bool) -> Consumer:
    group = config.CONSUMER_GROUP
    if from_beginning:  # новая группа = чтение с самого начала топика
        group = f"{group}-{int(datetime.now().timestamp())}"
    return Consumer({
        "bootstrap.servers": config.KAFKA_BOOTSTRAP,
        "group.id": group,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })


def process_batch(messages, conn) -> int:
    """Разбирает пачку сообщений, размечает тональность и сохраняет."""
    records = []
    for m in messages:
        try:
            records.append(json.loads(m.value().decode("utf-8")))
        except (ValueError, AttributeError) as e:
            print(f"  [битое сообщение пропущено] {e}", file=sys.stderr)
    records = [r for r in records if r.get("content")]
    if not records:
        return 0

    preds = sentiment.classify([r["content"] for r in records])
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        {
            "review_id": r["review_id"],
            "published_at": r["published_at"],
            "processed_at": now,
            "score": r.get("score"),
            "content": r["content"],
            "label": label,
            "confidence": conf,
            "source": r.get("source", "unknown"),
        }
        for r, (label, conf) in zip(records, preds)
    ]
    return storage.save_batch(conn, rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-beginning", action="store_true")
    p.add_argument("--batch", type=int, default=32, help="размер пачки")
    args = p.parse_args()

    print("Загрузка модели...")
    sentiment.classify(["прогрев модели"])
    conn = storage.connect(config.DB_PATH)
    consumer = make_consumer(args.from_beginning)
    consumer.subscribe([config.TOPIC_REVIEWS])
    print(f"Слушаю топик '{config.TOPIC_REVIEWS}'. Ctrl+C — выход.")

    total = 0
    try:
        while _running:
            msgs = consumer.consume(num_messages=args.batch, timeout=1.0)
            good = []
            for m in msgs:
                if m.error():
                    if m.error().code() != KafkaError._PARTITION_EOF:
                        print(f"  [ошибка Kafka] {m.error()}", file=sys.stderr)
                    continue
                good.append(m)
            if not good:
                continue
            saved = process_batch(good, conn)
            consumer.commit(asynchronous=False)  # фиксируем смещения после записи
            total += saved
            print(f"  обработано {len(good)}, новых {saved}, всего {total}")
    finally:
        consumer.close()
        conn.close()


if __name__ == "__main__":
    main()
