import pandas as pd
import numpy as np
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt

# DATA PREPROCESSING STEP 1=====================================================
# 1-1. Load data
dirty_data = pd.read_csv('tracks.csv')
dataset = dirty_data

# 1-2. Checking data
print("dataset information")
print(dataset.info())
print(dataset.describe())
print(dataset.shape)

# 1.3. Fill missing data
print("fill missing data")
dataset = dataset.fillna(dataset.median())
dataset = dataset.apply(lambda x: x.fillna(x.value_counts().index[0]))

# 1.4.1. Drop not use column (unique data)
print("drop unique data")
dataset = dataset.drop(columns=['id', 'name', 'artists', 'id_artists'])

# 1.4.2. Drop duplicated data
dataset.duplicated().sum()
dataset = dataset[~dataset.duplicated()]
print("drop duplicate")

# 1.4.3. Drop popularity zero data
# => A song with zero popularity is an outlier that does not reflect proper information.
dataset = dataset[dataset.popularity > 0]
print("drop popularity == 0")

# 1.4.4. Drop Speechiness song (only instrumental)
dataset = dataset[dataset.speechiness > 0]
print("drop only instrumental")

# 1.4.5. Drop Liveness song
# => If the concert is live, it is highly likely to be duplicated because existing music sources may exist.
dataset = dataset[dataset.liveness < 0.9]
print("drop Liveness")

# 1.4.6. Drop Duration_ms short data
# => Less than a minute of data is outlier data, which is likely to be advertised music.
dataset = dataset[dataset.duration_ms > 60000]  # 1min
print("drop Duration_ms")

# 1.5 Modity release date (YYYY -> YYY (first 3 chat))
# ref : https://stackoverflow.com/questions/36505847/substring-of-an-entire-column-in-pandas-dataframe/36506041
dataset['release_date'] = dataset['release_date'].str[0:3].astype(float)

print(dataset.shape)
print(dataset['release_date'].isna().sum())

# DATA VISUALIZATION STEP =====================================================
# 2. Correlation of numeric features =====================================================
# ref : https://hong-yp-ml-records.tistory.com/33 [HONG YP's Data Science BLOG]

# 2.1. Select numeric columns
numeric_columns = dataset.columns[dataset.dtypes != 'object']
string_columns = dataset.columns[dataset.dtypes == 'object']
print('There are' + str(len(numeric_columns)) + 'numeric columns & ' + str(len(string_columns)) + ' string columns')
data_numeric = pd.DataFrame(data=dataset, columns=numeric_columns, index=dataset.index)

# 2.1.2. Calculate correlation
corr = np.abs(data_numeric.corr())

# 2.2 Visualize correlation of numeric features by heatmap
colormap = plt.cm.PuBu
plt.figure(figsize=(10, 8))
plt.title("Correlation of Features", y=1.00, size=10)
sns.heatmap(corr, annot=True, cmap=colormap, linecolor="white", fmt='.3f', )

# 2.2. Compare correlation of features by selecting showed high relation
# =====================================================

# 2.2.1. Define range of correlation to consider
low_criteria = 0.3
high_criteria = 1


def is_between_criteria(x):
    if low_criteria <= x < high_criteria:
        return True
    else:
        return False


# 2.2.2. Features to consider importantly
important_features = ['popularity', 'release_date']

# 2.2.3. Select feature shows high correlation and print
high_corr_important_feature = []
for important_feature in important_features:
    series = np.abs(corr[important_feature]).sort_values(ascending=False)
    print(important_feature + " absolute correlation list")
    # 2.2.3.1 print high related
    print('{:.20s}'.format(important_feature)+ " absolute correlation is not between " + '{:2.3f}'.format(low_criteria) + " ~ " + '{:2.3f}'.format(high_criteria))
    for i, row in enumerate(series):
        if is_between_criteria(row):
            print('{:20.20s}'.format(series.index[i]) + " : " + '{:2.3f}'.format(row))
            if important_feature == 'popularity': high_corr_important_feature.append(series.index[i])
    print("          " + '{:.20s}'.format(important_feature) + " absolute correlation is not between " + '{:2.3f}'.format(low_criteria) + " ~ " + '{:2.3f}'.format(high_criteria))
    for i, row in enumerate(series):
        if not is_between_criteria(row):
            if row == 1: continue
            print("          " + '{:20.20s}'.format(series.index[i]) + " : " + '{:2.3f}'.format(row))

high_corr_important_feature.append('popularity')
print(high_corr_important_feature)

# 2.3. Visualize relation of important features
# ref: https://seaborn.pydata.org/tutorial/relational.html
# ref: https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/axes_box_aspect.html # square view
sample = dataset.sample(1000)
fig, axs = plt.subplots(ncols=len(high_corr_important_feature), nrows=len(high_corr_important_feature),
                        constrained_layout=True, figsize=(12, 12))

for x in range(len(high_corr_important_feature)):
    for y in range(len(high_corr_important_feature)):
        sns.scatterplot(x=high_corr_important_feature[x], y=high_corr_important_feature[y], hue="popularity",
                        data=sample, ax=axs[x][y], size=1)
        axs[x][y].legend_.remove()
        axs[x, y].set_box_aspect(1)

plt.show()
