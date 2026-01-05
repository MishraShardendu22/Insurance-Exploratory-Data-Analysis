# Insurance Analysis

## Index

**1. inplace=True** applies the change directly to the existing DataFrame.

- inplace=True → modifies dfCleaned itself, returns None
- inplace=False (default) → returns a new DataFrame, original unchanged

**2. A Target Variable** is the output you want to predict.

- Used in supervised learning
- Also called: label, dependent variable.

**3. One-hot encoding** converts a categorical column into **multiple binary columns**.

Rule:

- One category → `1`
- All others → `0`

Example:

Before:

```text
region
north
south
west
```

After:

```text
region_north  region_south  region_west
1             0             0
0             1             0
0             0             1
```

Why:

- ML models can’t use text
- Prevents false ordering (north ≠ south ≠ west)

In pandas:

```python
pd.get_dummies(df, columns=["region"])
```

**4. Feature scaling** Specifically: **Standardization (Z-score normalization)**.

What it does:

- Mean → `0`
- Std dev → `1`

Formula:

```txt
x' = (x − mean) / std
```

Used for:

- Linear regression
- Logistic regression
- SVM
- KNN
- Neural networks

**5. Random State Usage**.

Some operations use **randomness**
(example: `train_test_split`, shuffling data, initializing weights).

Without control, results change every run.

---

### Example WITHOUT `random_state`

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
```

Run this **twice**:

- Train data → different rows
- Test data → different rows
- Model accuracy → different

Unstable.

---

### Example WITH `random_state`

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42
)
```

Run this **100 times**:

- Same rows in train
- Same rows in test
- Same output

Stable.

---

### Why this happens

Internally:

```text
random numbers → data split
```

`random_state=42`:

```text
fixed starting point → same random numbers → same split
```

No magic.
`42` is just a number.

---

### Real-world analogy

- Shuffling cards **blindfolded** → different order every time
- Shuffling with **written instructions** → same order every time

---

### When to use

- Learning
- Debugging
- Comparing models

### When not required

- Final production runs

## Exploratory Data Analysis (EDA) Steps

Step 1: Setup.
This is used to suppress warnings during the analysis.
This is necessary as the warnings can clutter the output.

```py
import warnings
warnings.filterwarnings('ignore')
```

Step 2: Get the information.
Get the basic information about the dataset.

```py
df.head()
# Head shows the first five rows of the dataset.

df.info()
# Info gives a summary of the dataset (we learn about the data types and non-null counts).

df.isnull().sum()
# Shows how many null values are present in each column.

df.columns
# Shows the names of all the columns in the dataset.
```

Step 3: Mark the distributions.
Go ahead and plot the distributions of all numerical columns.
We do this so we can understand how the data is distributed.

```py
numClm = ['age', 'bmi', 'children', 'charges']

for i in numClm:
    plt.figure(figsize=(6,4))
    sns.histplot(df[i],kde=True,bins=20)
```

Get an idea of the categorical columns.
Basically, we will plot countplots for all categorical columns.
A categorical column contains labels or categories, not numeric measurements.

```py
sns.countplot(x = df['smoker'])
sns.countplot(x = df['children'])
```

Step 4: Mark the outliers and correlation.
Here we create box plots so we can see the outliers in the numerical columns.

```py
for col in numClm:
    plt.figure(figsize= (6,4))
    sns.boxplot(x = df[col])
```

We plot a heatmap to see the correlation between the numerical columns.

```py
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True),annot=True)
```

## Data Cleaning and Pre-processing Steps

Step 1: Make a copy of the dataset.

```py
dfCleaned = df.copy()
dfCleaned.notnull().sum()
```

Step 2: Add one-hot encoding to categorical columns, to clean the data for ML models.
Basically, ML models don't understand text data, so we convert text columns to numerical columns using one-hot encoding.

```py
import pandas as pd

dfCleaned = df.copy()

# drop duplicates
dfCleaned.drop_duplicates(inplace=True)

# encode sex
dfCleaned["sex"] = dfCleaned["sex"].map({"female": 0, "male": 1})
dfCleaned.rename(columns={"sex": "isMale"}, inplace=True)

# encode smoker
dfCleaned["smoker"] = dfCleaned["smoker"].str.strip().str.lower()
dfCleaned["smoker"] = dfCleaned["smoker"].map({"yes": 1, "no": 0})
dfCleaned.rename(columns={"smoker": "isSmoker"}, inplace=True)

# one-hot encode region
dfCleaned = pd.get_dummies(dfCleaned, columns=["region"], drop_first=True)

# Convert bool to int (if any)
region_cols = ["region_northwest", "region_southeast", "region_southwest"]
dfCleaned[region_cols] = dfCleaned[region_cols].astype(int)

dfCleaned.head()
```

## Feature Engineering and Extraction

Step 1: Feature Scaling using Standardization (Z-score normalization).

- Basically, what we do here is scale the numerical columns so that they have a mean of 0 and a standard deviation of 1. (Basically, it is between -1 and 1)
- This is used to tell if a certain variable affects the target variable more than others.

```py
from sklearn.preprocessing import StandardScaler
cols = ['age','bmi','children']
scaler = StandardScaler()

dfCleaned[cols] = scaler.fit_transform(dfCleaned[cols])
dfCleaned.head()
```

## Finding Co-relationship between Features and Target Variable

```py
from scipy.stats import pearsonr
import pandas as pd

# feature list (exclude target)
selected_features = [
    'age', 'isMale', 'bmi', 'children', 'isSmoker',
    'region_northwest', 'region_southeast', 'region_southwest',
    'bmiCategory_Normal', 'bmiCategory_Overweight', 'bmiCategory_Obese'
]

results = []

for feature in selected_features:
    r, p = pearsonr(dfCleaned[feature], dfCleaned["charges"])
    results.append({
        "Feature": feature,
        "Pearson_r": r,
        "p_value": p
    })

correlation_df = pd.DataFrame(results).sort_values(by="Pearson_r", ascending=False)
pd.options.display.float_format = "{:.3e}".format

significant_df = correlation_df[correlation_df["p_value"] < 0.05]
significant_df["Feature"]
```
