# МЕТРИКИ КАЧЕСТВА
## Machine Learning Classification
## Knee MRI Dataset (KL Grading)


## 1. ВВЕДЕНИЕ / КІРІСПЕ

**Цель работы:** определить метрики качества классификации (Accuracy, Precision, Recall, F1-Score) на датасете Knee MRI с использованием машинного обучения.

**Исходные данные:**
- **Датасет:** `results/eda_figures/eda_features.csv`
- **Размер:** 1 733 объектов, 20 признаков
- **Задача:** предсказание степени остеоартрита коленного сустава (KL-grade)


## 2. ПОСТАНОВКА ЗАДАЧИ

### 2.1 Описание классификации

Задача — мультиклассовая классификация по степени Kellgren–Lawrence (KL-grade):

| Класс | Обозначение | Описание |
|-------|-------------|----------|
| 0 | KL-0 | Здоровый сустав (норма) |
| 1 | KL-1 | Сомнительный остеоартрит |
| 2 | KL-2 | Минимальный остеоартрит |
| 3 | KL-3 | Умеренный остеоартрит |
| 4 | KL-4 | Тяжёлый остеоартрит |

**Распределение классов:**
- KL-0: 400 объектов (23.08%)
- KL-1: 400 объектов (23.08%)
- KL-2: 400 объектов (23.08%)
- KL-3: 400 объектов (23.08%)
- KL-4: 133 объекта (7.67%)

Классы 0–3 сбалансированы; класс KL-4 представлен слабее — это отражает реальное клиническое распределение (тяжёлые случаи реже).


## 3. МЕТОДОЛОГИЯ

### 3.1 Подготовка данных

**Шаг 1: Загрузка данных**

```python
import pandas as pd

df = pd.read_csv('results/eda_figures/eda_features.csv').dropna()
print(f'Загружено: {len(df)} объектов')
```

Загружено: 1 733 объектов

**Шаг 2: Разделение на обучающую и тестовую выборки**

```python
from sklearn.model_selection import train_test_split

feature_cols = [c for c in df.columns if c not in ['kl_grade', 'class_name']]
X = df[feature_cols]
y = df['kl_grade']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Параметры:**
- `test_size=0.2` — 20% данных для теста
- `random_state=42` — фиксированная случайность (воспроизводимость)
- `stratify=y` — сохранение пропорций классов

**Результат:**
- Обучающая выборка: 1 386 объектов (80%)
- Тестовая выборка: 347 объектов (20%)

**Шаг 3: Нормализация признаков (StandardScaler)**

Признаки имеют разный масштаб:
- `mean_intensity`: 90–150
- `laplacian_var`: 600–1200
- `homogeneity`: 0.4–0.8

Без нормализации модели будут считать признаки с большим масштабом более важными.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

После нормализации все признаки: среднее = 0, стандартное отклонение = 1.


### 3.2 Обучение моделей

**Шаг 4: Тестирование 5 различных моделей машинного обучения**

| № | Модель | Как работает |
|---|--------|-------------|
| 1 | Logistic Regression | Линейная модель. Строит границы между классами с помощью гиперплоскостей. |
| 2 | Random Forest | 300 деревьев решений голосуют. Каждое дерево учится на разных данных. |
| 3 | Decision Tree | Дерево вопросов. На каждом узле: «joint_space_width > 4.5? Да/Нет» |
| 4 | SVM (Support Vector Machine) | Ищет оптимальную гиперплоскость для разделения классов (RBF-ядро). |
| 5 | K-Nearest Neighbors | Смотрит на 7 ближайших соседей и выбирает самый частый класс. |

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=3000, multi_class='multinomial'),
    'Random Forest': RandomForestClassifier(n_estimators=300),
    'Decision Tree': DecisionTreeClassifier(max_depth=12),
    'SVM': SVC(kernel='rbf'),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
```


### 3.3 Расчёт метрик качества

**Шаг 5: Вычисление Accuracy, Precision, Recall, F1-Score для каждой модели**

```python
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score)

for name, model in models.items():
    predictions = model.predict(X_test_scaled)

    accuracy  = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall    = recall_score(y_test, predictions, average='weighted')
    f1        = f1_score(y_test, predictions, average='weighted')

    print(f'{name}:')
    print(f'  Accuracy:  {accuracy:.4f}')
    print(f'  Precision: {precision:.4f}')
    print(f'  Recall:    {recall:.4f}')
    print(f'  F1-Score:  {f1:.4f}')
```

`average='weighted'`:
Метрики рассчитываются для каждого класса отдельно, затем усредняются с учётом количества объектов в классе.


## 4. РЕЗУЛЬТАТЫ

### 4.1 Сводная таблица

| Ранг | Модель | Accuracy | Precision | Recall | F1 |
|------|--------|--------:|----------:|-------:|---:|
| 1 | Logistic Regression | 75.22% | 75.61% | 75.22% | 75.19% |
| 2 | SVM | 74.06% | 74.64% | 74.06% | 73.96% |
| 3 | Random Forest | 71.47% | 73.69% | 71.47% | 70.36% |
| 4 | K-Nearest Neighbors | 63.11% | 63.33% | 63.11% | 61.60% |
| 5 | Decision Tree | 54.18% | 54.34% | 54.18% | 54.19% |


### 4.2 Визуализация

Графики создаются с помощью:
```python
import matplotlib.pyplot as plt

plt.bar(...)         # Столбчатая диаграмма F1 по моделям
ConfusionMatrixDisplay(...)  # Confusion Matrix лучшей модели
```

![Метрики классификации и Confusion Matrix](results/homework_template_style/classification_bar_and_confusion.png)

**Описание графиков:**
- **Левый (Bar chart):** F1-score (weighted) для каждой модели. Logistic Regression лидирует.
- **Правый (Confusion Matrix):** Матрица ошибок лучшей модели (Logistic Regression). На диагонали — правильные предсказания, вне диагонали — ошибки. Видно, что классы KL-0 и KL-3 распознаются лучше всего.


## 5. АНАЛИЗ РЕЗУЛЬТАТОВ

### 5.1 Logistic Regression — лучшая модель

- **Данные хорошо разделяются линейно.** После нормализации MRI-признаки (текстуры, joint space width, osteophyte score) достаточно линейно разделяют KL-классы. Сложные модели (Random Forest, SVM) не дают существенного преимущества.

- **Простота = надёжность.** Logistic Regression не переобучается, быстро работает, легко интерпретируется.

- **Дисбаланс KL-4.** Класс KL-4 содержит лишь 7.67% объектов, что снижает общую точность. Для улучшения можно применить SMOTE или взвешивание классов.

### 5.2 Ключевые наблюдения

1. **Logistic Regression** и **SVM** показывают очень близкие результаты (~75% vs ~74%) — оба хорошо работают на линейно разделимых данных.
2. **Random Forest** чуть хуже (71.5%), что говорит о том, что ансамбли деревьев неоптимальны для данного набора признаков.
3. **Decision Tree** показывает худший результат (54%) — одно дерево переобучается на 20 признаках.
4. **KNN** (63%) страдает от проклятия размерности при 20 признаках.


## 6. ОПРЕДЕЛЕНИЯ МЕТРИК

- **Accuracy** — доля правильно классифицированных образцов: $Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$
- **Precision** — доля истинно положительных среди всех предсказанных положительных: $Precision = \frac{TP}{TP + FP}$
- **Recall** — доля истинно положительных среди всех реально положительных: $Recall = \frac{TP}{TP + FN}$
- **F1-Score** — гармоническое среднее Precision и Recall: $F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$

Для мультиклассовой задачи (5 KL-грейдов) используется `weighted` усреднение — метрики рассчитываются для каждого класса, затем средневзвешенно объединяются.


## 7. СОХРАНЁННЫЕ ФАЙЛЫ

| Файл | Описание |
|------|----------|
| `results/homework_template_style/classification_class_distribution.csv` | Распределение классов |
| `results/homework_template_style/classification_summary_metrics.csv` | Сводная таблица метрик (Accuracy, Precision, Recall, F1) |
| `results/homework_template_style/classification_bar_and_confusion.png` | Визуализация: F1 bar chart + Confusion Matrix |
| `notebooks/knee_template_metrics_and_linear_regression.ipynb` | Jupyter Notebook с полным кодом |
