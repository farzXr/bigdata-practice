"""Дашборд: соотношение положительных и отрицательных отзывов в реальном времени.

Запуск:  streamlit run dashboard.py
Страница сама перечитывает базу каждые несколько секунд.
"""
from datetime import datetime, timedelta, timezone

import altair as alt
import pandas as pd
import streamlit as st

import config
import storage
import textstats

st.set_page_config(page_title="Тональность отзывов", layout="wide")

COLORS = {"positive": "#2e9d5b", "neutral": "#9aa0a6", "negative": "#d64545"}
NAMES = {"positive": "Положительные", "neutral": "Нейтральные", "negative": "Отрицательные"}
PERIODS = {"Час": "h", "День": "D", "Неделя": "W"}

st.title(f"Тональность отзывов о приложении `{config.APP_ID}`")
st.caption("Google Play → Kafka → rubert-tiny-sentiment → SQLite → этот дашборд")

with st.sidebar:
    st.header("Настройки")
    period = st.selectbox("Шаг графика", list(PERIODS), index=1)
    sources = st.multiselect("Источник", ["live", "replay"], default=["live", "replay"])
    hide_neutral = st.checkbox("Скрыть нейтральные", value=False)
    refresh = st.slider("Обновление, с", 2, 30, 5)


@st.fragment(run_every=refresh)
def live_view():
    df = storage.load_all(config.DB_PATH)
    if df.empty:
        st.info("Данных пока нет. Запустите консьюмер и продюсер (см. README).")
        return
    df = df[df["source"].isin(sources)]
    if df.empty:
        st.warning("Нет данных для выбранных источников.")
        return

    counts = df["label"].value_counts()
    total = len(df)
    pos, neg, neu = (counts.get(k, 0) for k in ("positive", "negative", "neutral"))
    recent = df[df["processed_at"] > datetime.now(timezone.utc) - timedelta(minutes=1)]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Обработано отзывов", f"{total:,}".replace(",", " "))
    c2.metric("Положительные", f"{pos / total:.0%}")
    c3.metric("Отрицательные", f"{neg / total:.0%}")
    c4.metric("Позитив : негатив", f"{pos / neg:.2f}" if neg else "—")
    c5.metric("Скорость, отзывов/мин", len(recent))

    # --- доля классов по времени публикации
    shown = df if not hide_neutral else df[df["label"] != "neutral"]
    ts = (
        shown.set_index("published_at")
        .groupby("label")
        .resample(PERIODS[period])
        .size()
        .rename("n")
        .reset_index()
    )
    ts["share"] = ts["n"] / ts.groupby("published_at")["n"].transform("sum")
    ts["Тональность"] = ts["label"].map(NAMES)
    labels = [l for l in COLORS if l in ts["label"].unique()]
    color = alt.Color(
        "Тональность:N",
        scale=alt.Scale(domain=[NAMES[l] for l in labels], range=[COLORS[l] for l in labels]),
    )
    chart = (
        alt.Chart(ts)
        .mark_line(point=True)
        .encode(
            x=alt.X("published_at:T", title="Время публикации"),
            y=alt.Y("share:Q", title="Доля отзывов", axis=alt.Axis(format="%")),
            color=color,
            tooltip=[
                alt.Tooltip("published_at:T", title="Период"),
                "Тональность:N",
                alt.Tooltip("share:Q", format=".0%", title="Доля"),
                alt.Tooltip("n:Q", title="Отзывов"),
            ],
        )
        .properties(height=340)
    )
    st.subheader("Динамика настроений")
    st.altair_chart(chart, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Частые слова в негативе")
        words = textstats.top_words(df.loc[df["label"] == "negative", "content"])
        if words:
            wdf = pd.DataFrame(words, columns=["Слово", "Отзывов"])
            st.altair_chart(
                alt.Chart(wdf)
                .mark_bar(color=COLORS["negative"])
                .encode(x="Отзывов:Q", y=alt.Y("Слово:N", sort="-x"))
                .properties(height=360),
                use_container_width=True,
            )
    with right:
        st.subheader("Последние отзывы")
        last = df.sort_values("processed_at", ascending=False).head(15)
        st.dataframe(
            last.assign(Тональность=last["label"].map(NAMES))[
                ["published_at", "score", "Тональность", "confidence", "content"]
            ].rename(columns={
                "published_at": "Опубликован", "score": "★",
                "confidence": "Уверенность", "content": "Текст",
            }),
            hide_index=True,
            use_container_width=True,
            height=360,
        )


live_view()
