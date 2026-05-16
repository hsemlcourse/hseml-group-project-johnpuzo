[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)

# ML Project — Предсказание качества рецепта по его характеристикам

**Студент:** Брехов Виталий Игоревич

**Группа:** БИВ237


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуск](#запуск)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Отчёт](#отчёт)


## Описание задачи

**Задача:** многоклассовая классификация качества рецепта.

**Датасет:** [Food.com Recipes and User Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?resource=download).

**Целевая переменная:** класс качества рецепта:

- `bad` — рецепты из нижней части распределения среднего рейтинга;
- `normal` — рецепты со средним качеством;
- `good` — рецепты из верхней части распределения среднего рейтинга.

Таргет строится на уровне рецепта по среднему пользовательскому рейтингу `mean_rating`. Чтобы разметка была устойчивее, в итоговую выборку включаются только рецепты, у которых есть не менее 10 пользовательских оценок. После этого рецепты делятся по квантилям среднего рейтинга: нижние 30% относятся к `bad`, средние 40% — к `normal`, верхние 30% — к `good`.

Фактические пороги разметки:

- `bad`: `mean_rating <= 4.625`;
- `normal`: `4.625 < mean_rating < 4.8571`;
- `good`: `mean_rating >= 4.8571`.

В отчёте эти границы округляются до 4.6 и 4.9.

**Целевая метрика:** `macro F1`.

`macro F1` выбрана как основная метрика, потому что задача является многоклассовой, и важно учитывать качество модели по каждому классу отдельно. Даже при близком к сбалансированному разбиении классы не полностью равны по размеру, поэтому обычная `accuracy` может скрывать ситуацию, когда модель хорошо предсказывает только самый частый класс. `macro F1` усредняет F1-score по классам без учёта их размера, поэтому лучше отражает качество для всех трёх классов: `bad`, `normal`, `good`.

Дополнительно считаются:

- `weighted F1` — учитывает размер классов;
- `balanced accuracy` — показывает среднюю полноту по классам.


## Структура репозитория

```text
.
├── data
│   ├── processed               # Очищенные и обработанные данные, таблицы результатов
│   └── raw                     # Исходные файлы RAW_recipes.csv и RAW_interactions.csv
├── models                      # Сохранённая модель и список признаков
├── notebooks
│   ├── 01_eda.ipynb            # EDA, очистка, визуализации, формирование таргета
│   ├── 02_baseline.ipynb       # Baseline-модели на базовых признаках
│   └── 03_experiments.ipynb    # Эксперименты, подбор модели, ablation study, интерпретация
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   └── report.md               # Финальный отчёт
├── src
│   ├── api.py                  # FastAPI-сервис для запросов к модели
│   ├── interpretation.py       # Интерпретация модели и ablation study
│   ├── modeling.py             # Обучение и оценка моделей
│   ├── preprocessing.py        # Предобработка данных и feature engineering
│   ├── streamlit_app.py        # Streamlit-интерфейс
│   └── train_pipeline.py       # Полный цикл обучения из Docker
├── tests
│   └── test.py                 # Тесты пайплайна
├── .dockerignore
├── .flake8
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── requirements.txt
├── requirements-docker.txt
└── README.md
```

Основная логика проекта вынесена в `src/`. Ноутбуки используются как рабочие сценарии: EDA, baseline и основные эксперименты.


## Запуск

### 1. Клонировать репозиторий

```bash
git clone https://github.com/hsemlcourse/hseml-group-project-johnpuzo.git
cd hseml-group-project-johnpuzo
git checkout cp2
```

### 2. Создать виртуальное окружение

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Установить зависимости

Для локальной разработки и запуска ноутбуков:

```bash
pip install -r requirements.txt
```

Для Docker используется облегчённый файл зависимостей:

```text
requirements-docker.txt
```

### 4. Запустить ноутбуки

```bash
jupyter notebook
```

Порядок запуска:

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_baseline.ipynb`
3. `notebooks/03_experiments.ipynb`

### 5. Запустить тесты и линтер

```bash
pytest
flake8 src tests
```

Или с 'make':

```bash
make test
make lint
make check
```

### 6. Запуск через Docker

В проекте предусмотрены два Docker-сценария.

#### Сценарий 1: полный цикл обучения

Контейнер читает данные из `data/raw`, выполняет предобработку, feature engineering, обучение моделей, подбор гиперпараметров и сохраняет артефакты в `data/processed` и `models`.

```bash
docker compose --profile train up --build trainer
```

После запуска должны появиться или обновиться файлы:

```text
data/processed/experiments_validation_results.csv
data/processed/best_models_by_family.csv
data/processed/summary_table.csv
data/processed/cleaning_table.csv
models/best_model.joblib
models/best_model_features.csv
```

#### Сценарий 2: деплой готовой модели

Контейнеры поднимают FastAPI и Streamlit. Модель не переобучается, а загружается из `models/best_model.joblib`.

```bash
docker compose --profile deploy up --build
```

После запуска доступны:

```text
FastAPI:   http://127.0.0.1:8000/docs
Streamlit: http://127.0.0.1:8501
```

Остановить контейнеры:

```bash
docker compose down
```

### 7. Локальный запуск FastAPI и Streamlit без Docker

FastAPI:

```bash
uvicorn src.api:app --reload
```

Streamlit:

```bash
streamlit run src/streamlit_app.py
```

Проверка API:

```bash
curl http://127.0.0.1:8000/health
```

Пример POST-запроса к модели:

```json
{
  "features": {
    "minutes": 45,
    "n_steps": 8,
    "n_ingredients": 10,
    "calories": 350,
    "total_fat_pdv": 20,
    "sugar_pdv": 10,
    "sodium_pdv": 15,
    "protein_pdv": 12,
    "saturated_fat_pdv": 8,
    "carbs_pdv": 14
  }
}
```


## Данные

- `data/raw/` — исходные файлы;
- `data/processed/` — предобработанные данные и таблицы результатов.

Исходные таблицы:

| Файл | Описание | Размер |
|---|---|---:|
| `RAW_recipes.csv` | Информация о рецептах | 231637 строк, 12 столбцов |
| `RAW_interactions.csv` | Пользовательские оценки и отзывы | 1132367 строк, 5 столбцов |

После очистки:

| Таблица | Размер |
|---|---:|
| `recipes` после очистки | 230542 строки |
| `interactions` после очистки | 1071520 строк |
| Итоговая recipe-level таблица | 19965 строк, 52 столбца |

Распределение классов в итоговой таблице:

| Класс | Количество | Доля |
|---|---:|---:|
| `bad` | 6044 | 0.3027 |
| `normal` | 7917 | 0.3965 |
| `good` | 6004 | 0.3007 |

В модель подаются два набора признаков:

| Набор признаков | Количество признаков | Назначение |
|---|---:|---|
| `base` | 10 | baseline-модели |
| `full` | 38 | основные эксперименты |

В `base` входят только базовые числовые признаки: время приготовления, число шагов, число ингредиентов и признаки пищевой ценности.

В `full` дополнительно входят признаки, полученные через feature engineering: длины списков, длины текстов, признаки даты публикации, отношения вроде `minutes_per_step`, `calories_per_ingredient`, бинарные флаги быстрых/долгих рецептов и логарифмированные признаки.

### Очистка и выбросы

В ходе очистки удаляются дубликаты, приводятся типы данных, фильтруются некорректные значения рейтинга и некорректные значения базовых числовых признаков.

Выбросы выявлялись через IQR. Массово удалять их не стали, потому что для рецептов экстремальные значения могут быть содержательно корректными: очень долгое приготовление, высокая калорийность, большое количество сахара или натрия. Чтобы снизить влияние тяжёлых хвостов, были добавлены `log1p`-признаки, а для линейных моделей использовался `RobustScaler`.

### Data leakage

Признаки `mean_rating`, `median_rating` и `rating_count` используются только для построения целевой переменной. Они не включаются ни в `X_base`, ни в `X_full`, потому что напрямую связаны с таргетом и привели бы к data leakage.

Разделение на `train`, `validation` и `test` выполняется после формирования recipe-level таблицы. Split стратифицирован по целевому классу, чтобы сохранить распределение `bad`, `normal`, `good` во всех частях выборки.

Все preprocessing-шаги внутри моделей выполняются через `sklearn Pipeline`, поэтому `SimpleImputer`, `RobustScaler` и PCA обучаются только на train-части в рамках конкретного эксперимента.


## Результаты

### Baseline

Baseline-модели обучались на наборе `base` из 10 признаков.

| Модель | Признаки | Macro F1 | Weighted F1 | Balanced Accuracy | Примечание |
|---|---|---:|---:|---:|---|
| `DummyClassifier` | base | 0.1893 | 0.2253 | 0.3333 | Наивный baseline |
| `LogisticRegression` | base | 0.3576 | 0.3539 | 0.3694 | Простая линейная модель |
| `KNN` | base | 0.3716 | 0.3811 | 0.3746 | Лучший baseline |

Baseline показывает, что даже простые модели находят некоторый сигнал в данных, но качество остаётся заметно ниже, чем у моделей на полном наборе признаков.

### Основные эксперименты

В основном блоке экспериментов сравнивались следующие семейства моделей:

- `LogisticRegression`;
- `KNN`;
- `RandomForestClassifier`;
- `ExtraTreesClassifier`;
- `HistGradientBoostingClassifier`;
- PCA-варианты для `LogisticRegression` и `KNN`.

Всего было проведено 83 эксперимента с разными моделями, наборами признаков и гиперпараметрами.

Лучшие результаты на validation:

| Модель | Признаки | Параметры | Macro F1 | Weighted F1 | Balanced Accuracy |
|---|---|---|---:|---:|---:|
| `RandomForest` | full | `max_depth=20`, `max_features=sqrt`, `min_samples_leaf=10` | 0.4217 | 0.4238 | 0.4217 |
| `RandomForest` | full | `max_depth=20`, `max_features=0.7`, `min_samples_leaf=10` | 0.4207 | 0.4232 | 0.4205 |
| `RandomForest` | full | `max_depth=10`, `max_features=sqrt`, `min_samples_leaf=10` | 0.4165 | 0.4151 | 0.4212 |
| `ExtraTrees` | full | `max_depth=20`, `max_features=sqrt`, `min_samples_leaf=3` | 0.4070 | 0.4100 | 0.4069 |
| `HistGradientBoosting` | full | `learning_rate=0.1`, `max_depth=8`, `min_samples_leaf=40` | 0.4061 | 0.4172 | 0.4137 |

Лучшей моделью стала `RandomForestClassifier` с параметрами:

```text
max_depth = 20
max_features = sqrt
min_samples_leaf = 10
```

Результаты лучшей модели на test:

| Модель | Macro F1 | Weighted F1 | Balanced Accuracy |
|---|---:|---:|---:|
| `RandomForestClassifier` | 0.4189 | 0.4199 | 0.4188 |

Качество на test почти не ниже, чем на validation, поэтому сильного переобучения по итоговым метрикам не наблюдается.

По классам на test:

| Класс | Precision | Recall | F1-score |
|---|---:|---:|---:|
| `bad` | 0.4248 | 0.3925 | 0.4080 |
| `normal` | 0.4257 | 0.4322 | 0.4289 |
| `good` | 0.4086 | 0.4317 | 0.4199 |

Модель предсказывает все три класса примерно на сопоставимом уровне. Это важно, потому что целевая метрика `macro F1` не должна улучшаться только за счёт одного класса.

### Вывод по экспериментам

`RandomForestClassifier` оказался лучшим семейством моделей в проведённых экспериментах. Модели `ExtraTrees` и `HistGradientBoosting` дали близкие, но немного более слабые результаты. `LogisticRegression` и `KNN` уступили ансамблевым моделям.

PCA не улучшил качество. В большинстве PCA-экспериментов качество было ниже, чем на полном наборе признаков. Поэтому финальная модель использует полный набор признаков без уменьшения размерности.

Финальная модель выбрана по максимальному `macro F1` на validation и дополнительно проверена на test.


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)

В отчёте подробно описаны:

1. введение и постановка задачи;
2. поиск и описание данных;
3. обработка и подготовка данных;
4. baseline-модель;
5. эксперименты;
6. финальная модель и интерпретируемость;
7. деплой;
8. заключение и выводы.
