# Log Regretion HW

Учебный ML-проект из курса KarpovCourse HardDA: предсказание оттока пользователей (churn) на данных сервиса объявлений.

## О проекте

Решение задачи бинарной классификации: предсказать, уйдёт ли пользователь (создатель объявления) из сервиса в течение месяца. Данные — объявления за октябрь 2022, таргет — активность на 1 ноября.

## Что внутри

`Log Regretion HW.ipynb` — полный пайплайн:

### 1. Выгрузка и подготовка данных
- Подключение к ClickHouse
- Выгрузка объявлений (`live_adverts`, `user_passports`)
- Расчёт таргета `churn` на основе дат активности

### 2. Feature engineering
- `opening_adverts_amount` — количество активных объявлений
- `price` — средняя цена
- `auto_age` — средний возраст автомобиля
- `advert_age` — средний возраст объявления
- `platform` — основная платформа (с приоритетом android > ios > desktop > mobile > unknown)
- `is_top_model` — есть ли топ-10 модель
- `user_type_cars_name` — тип продавца

### 3. Дополнительные признаки из транзакций
- Агрегация `user_transactions` по классам: `revenue`, `renewal`, `premium`
- Логарифмирование через `log1p`
- Мердж с основным датафреймом

### 4. Пайплайн предподготовки
- `SimpleImputer` для числовых (mean) и категориальных (most_frequent / constant)
- `OneHotEncoder` с `drop='first'`
- `StandardScaler`

### 5. Кастомные классы

**`ThresholdClassifier`** — обёртка над моделью, позволяющая настраивать порог классификации как гиперпараметр:
- если у модели нет `predict_proba` — оборачивает в `CalibratedClassifierCV`
- `predict` сравнивает вероятности с `threshold`

**`make_refit_with_recall_threshold`** — кастомный `refit` для `GridSearchCV`:
- выбирает модель с максимальным precision при условии recall > 0.8

### 6. Обучение моделей
- **LogisticRegression** (l1, l2) — итоговая модель
- **LinearSVC** — сравнение
- `GridSearchCV` с `StratifiedKFold(n_splits=3)`
- Метрики: precision, recall

### 7. Оценка качества
- Precision / Recall на тесте
- F1 (вручную и через sklearn)
- PR-кривая, PR-AUC (`auc` + `average_precision_score`)
- ROC-кривая, ROC-AUC (`auc` + `roc_auc_score`)
- Permutation importance (в работе)

## Стек

- Python 3.12
- pandas, numpy
- scikit-learn
- clickhouse-driver
- matplotlib

