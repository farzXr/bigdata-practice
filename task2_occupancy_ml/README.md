# Задание 2. Определение присутствия людей в помещении по показаниям датчиков CO₂, температуры, влажности и освещённости

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/farzXr/bigdata-practice/blob/main/task2_occupancy_ml/occupancy.ipynb)

| | |
| --- | --- |
| **Вход** | Показания датчиков раз в минуту: температура, влажность, освещённость, CO₂, абсолютная влажность и время |
| **Выход** | Есть ли в помещении люди (1/0), то есть бинарная классификация |
| **Данные** | [UCI Occupancy Detection](https://archive.ics.uci.edu/dataset/357/occupancy+detection), около 20,5 тыс. записей: обучение на `datatraining.txt`, проверка на двух независимых периодах `datatest.txt` и `datatest2.txt` |
| **Модели** | Логистическая регрессия, Random Forest, градиентный бустинг (HistGradientBoosting) |
| **Метрики** | Accuracy, Precision, Recall, F1, ROC-AUC, матрица ошибок |

## Что внутри ноутбука

1. Загрузка датасета: сначала с UCI, при недоступности — с GitHub автора.
2. Разведочный анализ: графики датчиков во времени, распределения по классам, корреляции.
3. Дополнительные признаки: час, выходной день, изменение CO₂ и освещённости за 5 минут.
4. Обучение трёх моделей и сравнительная таблица метрик на двух тестовых периодах.
5. ROC-кривые и матрицы ошибок лучшей модели.
6. Важность признаков (permutation importance).
7. Эксперимент без датчика освещённости, то есть только CO₂, температура и влажность.
8. Выводы: цифры подставляются автоматически из результатов.

## Запуск

**В Google Colab** нажмите кнопку выше, затем «Среда выполнения → Выполнить всё».

**Локально:**

```bash
cd task2_occupancy_ml
pip install -r requirements.txt
jupyter notebook occupancy.ipynb
```

Датасет скачается в `data/` при первом запуске. Если сети нет, скачайте архив
со [страницы датасета](https://archive.ics.uci.edu/dataset/357/occupancy+detection)
и распакуйте три `.txt` файла в `data/`. Весь ноутбук выполняется меньше чем за минуту
на обычном ноутбуке, видеокарта не нужна.

## Источник

Candanedo L. M., Feldheim V. Accurate occupancy detection of an office room from light,
temperature, humidity and CO₂ measurements using statistical learning models //
Energy and Buildings. 2016. Vol. 112. P. 28–39.
