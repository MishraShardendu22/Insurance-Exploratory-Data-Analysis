# Insurance Analysis

## Exploratory Data Analysis (EDA) Steps

Step-1: Setup
This is used to suppress warnings during the analysis.
This is necessary as the warnings can clutter the output.

```py
import warnings
warnings.filterwarnings('ignore')
```

Step-2: Get the information
Get the basic information about the dataset.

```py
df.head()
# Head shows the first 5 rows of the dataset.

df.info()
# Info gives a summary of the dataset (we learn about the data types and non-null counts)

df.isnull().sum()
# Shows how many null values are present in each column.

df.columns
# Shows the names of all the columns in the dataset.
```

Step-3: Mark the distributions
Go ahead and plot the distributions of all numerical columns.
We do this so we can understand how the data is distributed.

```py
numClm = ['age', 'bmi', 'children', 'charges']

for i in numClm:
    plt.figure(figsize=(6,4))
    sns.histplot(df[i],kde=True,bins=20)
```

Get ideas about the categorical columns.
Basically, we will plot countplots for all categorical columns.
A categorical column contains labels or categories, not numeric measurements.

```py
sns.countplot(x = df['smoker'])
sns.countplot(x = df['children'])
```

Step-4: Mark the outliers and correlation
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
