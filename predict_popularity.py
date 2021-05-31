import joblib
import pandas as pd

df = pd.read_csv("dirty_tracks.csv")

# 1-2. Checking data
print("dataset information")
print(df.info())
print(df.describe())
print(df.shape)

# 1.3. Fill missing data
print("fill missing data")
dataset = df.fillna(df.median())
dataset = dataset.apply(lambda x: x.fillna(x.value_counts().index[0]))

name = dataset['name'] == 'Dreams'
artists = dataset['artists'] == '[\'Fleetwood Mac\']'

dataset = dataset[name & artists]

# 1.4.1. Drop not use column (unique data)
print("drop unique data")
dataset = dataset.drop(columns=['id', 'name', 'artists', 'id_artists'])
print("drop dummy data")
dataset = dataset.drop(columns=['mode', 'explicit'])

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
dataset = dataset[dataset.duration_ms > 60000] # 1min
print("drop Duration_ms")

# 1.5 Modify release date (YYYY first 4 chars))
# ref : https://stackoverflow.com/questions/36505847/substring-of-an-entire-column-in-pandas-dataframe/36506041
dataset['release_date'] = dataset['release_date'].str[0:4]

print(dataset.head())
print(dataset.shape)
print(dataset['release_date'].isna().sum())

popularity = dataset['popularity']
dataset = dataset.drop('popularity', axis=1)

print(popularity.shape)
print(dataset.shape)

clf2 = joblib.load('./2015_2020/XGBoost model/robust_ordinal_model1_regression_dirtydata_1.pkl')

print('The predicted popularity of the song is : ', clf2.predict(dataset))

