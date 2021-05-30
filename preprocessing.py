import pandas as pd
import random

# DATA PREPROCESSING STEP 1=====================================================
# 1.1. Load data
original_data = pd.read_csv('tracks.csv')

# 1.2. Checking data
print("original dataset information")
print(original_data.info())
print(original_data.describe())
print(original_data.shape)


# 1.3. Generate dirty data randomly
# ref: https://blockdmask.tistory.com/383
dirty_data = original_data
for i in range(0, int((len(dirty_data) - 1) / 100 * 2)):
    # random seed 고정
    col = random.randint(0, len(dirty_data.columns) - 1)
    row = random.randint(0, len(dirty_data) - 1)
    dirty_data.iat[row, col] = None

# 1.3.1. Checking specification of generated dirty data
dirty_data.to_csv('dirty_tracks.csv', sep=',', na_rep='NaN')
print("2% of dirty data")
print(dirty_data.info())


# 1.3.2. Fill missing data
dirty_data = dirty_data.fillna(dirty_data.median())

# 1.4.1. Drop not use column (unique data)
print("drop unique data")
dirty_data = dirty_data.drop(columns=['id', 'name', 'artists', 'id_artists'])

# 1.4.2. Drop duplicated data
dirty_data.duplicated().sum()
dirty_data = dirty_data[~dirty_data.duplicated()]
print("drop duplicate")
print(dirty_data.shape)

# 1.4.3. Drop popularity zero data
# => A song with zero popularity is an outlier that does not reflect proper information.
dirty_data = dirty_data[dirty_data.popularity > 0]
print("drop popularity == 0")
print(dirty_data.shape)

# 1.4.4. Drop Speechiness song (only instrumental)
dirty_data = dirty_data[dirty_data.speechiness > 0]
print("drop only instrumental")
print(dirty_data.shape)

# 1.4.5. Drop Liveness song
# => If the concert is live, it is highly likely to be duplicated because existing music sources may exist.
dirty_data = dirty_data[dirty_data.liveness < 0.9]
print("drop Liveness")
print(dirty_data.shape)

# 1.4.6. Drop Duration_ms short data
# => Less than a minute of data is outlier data, which is likely to be advertised music.
dirty_data = dirty_data[dirty_data.duration_ms > 60000] # 1min
print("drop Duration_ms")
print(dirty_data.shape)

# 1.5 Modity release date (YYYY -> YYY (first 3 chat))
# ref : https://stackoverflow.com/questions/36505847/substring-of-an-entire-column-in-pandas-dataframe/36506041
dirty_data['release_date'] = dirty_data['release_date'].str[0:3]

print(dirty_data.shape)
print(dirty_data['release_date'].isna().sum())


