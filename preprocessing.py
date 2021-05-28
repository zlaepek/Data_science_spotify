import pandas as pd
import random

from sklearn import preprocessing

# Read data
original_data = pd.read_csv('tracks.csv')

# 2% -> dirty data
print(len(original_data.columns))
print(len(original_data))

### https://blockdmask.tistory.com/383 ###
### random ###
for i in range(0, int((len(original_data) - 1) / 100 * 2)):
    # random seed 고정
    col = random.randint(0, len(original_data.columns) - 1)
    row = random.randint(0, len(original_data) - 1)
    original_data.iat[row, col] = None


original_data.to_csv('newtracks.csv', sep=',', na_rep='NaN')

print("2% of dirty data")
print(original_data.info())


# fill
original_data = original_data.fillna(original_data.median())

# drop -> unique data
print("drop unique data")
original_data = original_data.drop(columns=['id', 'name', 'artists', 'id_artists'])

# drop duplicated data
original_data.duplicated().sum()
original_data = original_data[~original_data.duplicated()]
print("drop duplicate")
print(original_data.shape)

# drop popularity is 0 data
original_data = original_data[original_data.popularity > 0]
print("drop popularity == 0")
print(original_data.shape)

# release date 변환
### https://stackoverflow.com/questions/36505847/substring-of-an-entire-column-in-pandas-dataframe/36506041 ###
### -> dataframe slice ###
original_data['release_date'] = original_data['release_date'].str[0:3]

pd.options.display.max_columns = None
pd.options.display.max_rows = None
print("slice year (interval 10 year)")
print(original_data.shape)

# drop -> Speechiness (only instrumental)
original_data = original_data[original_data.speechiness > 0]
print("drop only instrumental")
print(original_data.shape)

# drop -> Liveness (콘서트곡을 자르는 근거 (원곡이 있는데 두개 올리는거))
original_data = original_data[original_data.liveness < 0.9]
print("drop Liveness")
print(original_data.shape)

# drop -> Duration_ms  (시간 너무 짧은 자르기 (광고 음악이나 시간 너무 짧은 음원 자르기))
original_data = original_data[original_data.duration_ms > 60000] # 1min
print("drop Duration_ms")
print(original_data.shape)
