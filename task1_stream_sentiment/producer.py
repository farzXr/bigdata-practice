"""Продюсер: собирает отзывы из Google Play и публикует их в Kafka.

Режимы:
  collect  – один раз скачать N последних отзывов в CSV (для режима replay)
  live     – опрашивать Google Play каждые N секунд и отправлять только новые отзывы
  replay   – воспроизвести сохранённый CSV как поток с заданной скоростью

Примеры:
  python producer.py collect --count 5000
  python producer.py live --interval 120 --backfill 50
  python producer.py replay --rate 10
"""
import argparse
import json
import signal
import sys
import time
from datetime import datetime, timezone

import pandas as pd
from confluent_kafka import Producer

import config

_running = True


def _stop(*_):
    global _running
    _running = False
    print("\nОстановка...")


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)


# ---------------------------------------------------------------- Google Play
def fetch_newest(count: int = 100, continuation_token=None):
    """Возвращает (список отзывов, токен продолжения) — самые новые первыми."""
    from google_play_scraper import Sort, reviews

    result, token = reviews(
        config.APP_ID,
        lang=config.APP_LANG,
        country=config.APP_COUNTRY,
        sort=Sort.NEWEST,
        count=count,
        continuation_token=continuation_token,
    )
    return result, token


def to_message(r: dict, source: str) -> dict:
    """Приводит отзыв Google Play к формату сообщения в топике."""
    published = r["at"]
    if isinstance(published, datetime):
        # библиотека отдаёт локальное время без зоны; astimezone считает его локальным
        published = published.astimezone(timezone.utc).isoformat()
    return {
        "review_id": str(r["reviewId"]),
        "published_at": str(published),
        "score": int(r["score"]) if r.get("score") is not None else None,
        "content": (r.get("content") or "").strip(),
        "source": source,
    }


# ---------------------------------------------------------------------- Kafka
def make_producer() -> Producer:
    return Producer({
        "bootstrap.servers": config.KAFKA_BOOTSTRAP,
        "acks": "all",               # подтверждение от брокера — сообщение не потеряется
        "enable.idempotence": True,  # без дублей при повторной отправке
        "linger.ms": 50,
    })


def _delivery_report(err, msg):
    if err is not None:
        print(f"  [ошибка доставки] {err}", file=sys.stderr)


def send(producer: Producer, message: dict) -> None:
    producer.produce(
        config.TOPIC_REVIEWS,
        key=message["review_id"].encode(),
        value=json.dumps(message, ensure_ascii=False).encode("utf-8"),
        on_delivery=_delivery_report,
    )
    producer.poll(0)


# ---------------------------------------------------------------------- modes
def cmd_collect(args):
    rows, token = [], None
    while len(rows) < args.count:
        batch, token = fetch_newest(min(200, args.count - len(rows)), token)
        if not batch:
            break
        rows.extend(to_message(r, "replay") for r in batch)
        print(f"  скачано {len(rows)}")
        time.sleep(1)  # бережём источник
    df = pd.DataFrame(rows).drop_duplicates("review_id")
    df = df[df["content"].str.len() > 0]
    df = df.head(args.count)  # с токеном продолжения библиотека может вернуть больше, чем просили
    df.to_csv(config.REPLAY_CSV, index=False)
    print(f"Сохранено {len(df)} отзывов в {config.REPLAY_CSV}")


def _load_seen() -> set[str]:
    if config.SEEN_IDS_PATH.exists():
        return set(config.SEEN_IDS_PATH.read_text().split())
    return set()


def _save_seen(seen: set[str]) -> None:
    config.SEEN_IDS_PATH.write_text("\n".join(sorted(seen)))


def cmd_live(args):
    producer = make_producer()
    seen = _load_seen()
    first_cycle = not seen
    print(f"Live: {config.APP_ID}, опрос каждые {args.interval} с. Ctrl+C — выход.")
    while _running:
        try:
            batch, _ = fetch_newest(args.fetch)
        except Exception as e:  # сеть, изменения вёрстки и т.п. — не падаем
            print(f"  [ошибка источника] {e}", file=sys.stderr)
            batch = []

        new = [r for r in batch if str(r["reviewId"]) not in seen and r.get("content")]
        if first_cycle:
            # при самом первом запуске отправляем только `backfill` последних
            new = new[: args.backfill]
            seen.update(str(r["reviewId"]) for r in batch)
            first_cycle = False

        for r in reversed(new):  # от старых к новым — сохраняем порядок времени
            send(producer, to_message(r, "live"))
            seen.add(str(r["reviewId"]))
        producer.flush(10)
        _save_seen(seen)
        stamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{stamp}] новых отзывов: {len(new)}")

        for _ in range(args.interval):  # sleep, который прерывается Ctrl+C
            if not _running:
                break
            time.sleep(1)
    producer.flush(10)


def cmd_replay(args):
    if not config.REPLAY_CSV.exists():
        sys.exit(f"Нет файла {config.REPLAY_CSV}. Сначала: python producer.py collect")
    df = pd.read_csv(config.REPLAY_CSV)
    df = df.sort_values("published_at")  # воспроизводим в хронологическом порядке
    if args.limit:
        df = df.head(args.limit)
    producer = make_producer()
    delay = 1.0 / args.rate
    print(f"Replay: {len(df)} отзывов, {args.rate} сообщ./с. Ctrl+C — выход.")
    for i, row in enumerate(df.itertuples(index=False), 1):
        if not _running:
            break
        msg = {
            "review_id": str(row.review_id),
            "published_at": str(row.published_at),
            "score": None if pd.isna(row.score) else int(row.score),
            "content": str(row.content),
            "source": "replay",
        }
        send(producer, msg)
        if i % 100 == 0:
            print(f"  отправлено {i}/{len(df)}")
        time.sleep(delay)
    producer.flush(10)
    print("Готово.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    c = sub.add_parser("collect", help="скачать отзывы в CSV")
    c.add_argument("--count", type=int, default=5000)
    c.set_defaults(func=cmd_collect)

    l = sub.add_parser("live", help="живой опрос Google Play")
    l.add_argument("--interval", type=int, default=120, help="период опроса, с")
    l.add_argument("--fetch", type=int, default=100, help="сколько последних отзывов смотреть за опрос")
    l.add_argument("--backfill", type=int, default=50, help="сколько отправить при первом запуске")
    l.set_defaults(func=cmd_live)

    r = sub.add_parser("replay", help="воспроизвести CSV как поток")
    r.add_argument("--rate", type=float, default=10, help="сообщений в секунду")
    r.add_argument("--limit", type=int, default=0, help="ограничить число сообщений (0 — все)")
    r.set_defaults(func=cmd_replay)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
