import math
import time
import pickle
import pandas as pd
import ETE_scaling
from pprint import pprint as pp
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from xgboost import XGBRegressor

# DATA PREPROCESSING STEP 1=====================================================
# 1-1. Load data
dirty_data = pd.read_csv('dirty_tracks.csv')
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

# for 2010~2020 models
years_filtering_1 = dataset['release_date'] >= '2010'

# for 2015~2020 models
# years_filtering_1 = dataset['release_date'] >= '2015'


years_filtering_2 = dataset['release_date'] != '2021'
dataset = dataset[years_filtering_1 & years_filtering_2]


print(dataset.head())
print(dataset.shape)
print(dataset['release_date'].isna().sum())


# DATA PREPROCESSING STEP 2=====================================================
# 2.1 Set numerical column and categorical column
numerical_feature_list = ['duration_ms', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
categorical_feature_list = ['release_date', 'time_signature', 'key']


# 2.2 Run scaling & encoding using "scale_encode_combination" function (return type : dictionary)
combination_dataset = ETE_scaling.scale_encode_combination(dataset, numerical_feature_list, categorical_feature_list)
pp(combination_dataset)


# MODEL TRAINING STEP=====================================================
for key, mydataset in combination_dataset.items():
    # 3.1 Split target and feature
    print(mydataset.describe())

    datasetName = key
    feature = mydataset.drop('popularity', axis=1)
    target = mydataset['popularity']

    # 3.2 Split train and test
    train_X, test_X, train_Y, test_Y = train_test_split(feature, target, shuffle=True)

    # 3.3 Set model
    model = LinearRegression()

    # 3.4 Set hyper-parameter for GridSearchCV
    # for Linear Regression
    param_grid = {'fit_intercept': [True, False],
                  'normalize': [True, False],
                  'copy_X': [True, False],
                  'n_jobs': [None, 2, 10, 50, 100]}
    # for Lasso, Ridge
    # param_grid = {'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3, 4, 5, 6]}

    # for ElasticNet
    # param_grid = {'l1_ratio': [0.0001, 0.001, 0.01, 0.1, 1.0],
    #               'alpha': [0.0001, 0.01, 1, 2, 3, 4]}

    # for XGBRegressor
    # param_grid = {'learning_rate': [0.3, 0.05, 0.07],
    #               'max_depth': [5, 6, 7, 9],
    #               'min_child_weight': [4],
    #               'subsample': [0.7],
    #               'colsample_bytree': [0.7],
    #               'n_estimators': [100, 300, 500]}


    # 3.5 Define an evaluation metric as a r^2 (Coefficient of determination) in the scoring parameter
    # 3.6 Set and RUN GridSearchCV
    start_time = time.time()
    model_test = GridSearchCV(model, param_grid, scoring='r2', cv=3, verbose=1)
    model_test.fit(train_X, train_Y)
    print("The time that this function finish :", time.time() - start_time)


    # 3.7 Show the results of evaluation
    best_model = model_test.best_estimator_
    y_pred = best_model.predict(test_X)

    print(key + ' model 1\'s best estimator: ', best_model)
    print(key + 'model 1\'s best parameters: ', model_test.best_params_)
    print(key + 'model 1\'s best R^2 score: ', r2_score(test_Y, y_pred))
    print(key + 'model 1\'s best RMSE score: ', math.sqrt(mean_squared_error(test_Y, y_pred)))

    # 파일 이름 구성 -> combination type + classification인지 regresison인지 + dirty data/originaldata + 몇번쨰 시도인지 + 그외 저장해야 하는 정보
    with open(key + '_model1_regression_dirtydata_1.pkl', 'wb') as f:
        pickle.dump(model_test, f)
