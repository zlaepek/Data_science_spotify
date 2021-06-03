import time
import pickle
import pandas as pd
from Scale_Encode_Combination import scale_encode_combination
from pprint import pprint as pp
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from xgboost import XGBRegressor

# DATA PREPROCESSING STEP=====================================================
# 1-1. Load data
print("1-1. Load data")
dirty_data = pd.read_csv('dirty_tracks.csv')
dataset = dirty_data


# 1-2. Checking data
print("1-2. Data checking")
print(dataset.shape)
print(dataset.describe())
print(dataset.dtypes)


# 1-3. Fill missing data
print("1-3. Fill missing data")
print(dataset.isna().sum())
dataset = dataset.fillna(dataset.median())
dataset = dataset.apply(lambda x: x.fillna(x.value_counts().index[0]))
print("=> after fill missing data")
print(dataset.isna().sum())


# 1-4-1. Drop not use column(unique data) & duplicated data 
print("1-4-1. Drop not use column(unique data) & duplicated data")
print("drop unique data")
dataset = dataset.drop(columns=['id', 'name', 'artists', 'id_artists'])
print("drop dummy data")
dataset = dataset.drop(columns=['mode', 'explicit'])
print("drop duplicate")
dataset.duplicated().sum()
dataset = dataset[~dataset.duplicated()]


# 1-4-2. Drop Outlier
print("1-4-2. Drop Outlier")

# Drop popularity zero data
# => A song with zero popularity is an outlier that does not reflect proper information.
print("drop popularity == 0")
dataset = dataset[dataset.popularity > 0]

# Drop Speechiness song (only instrumental)
print("drop only instrumental")
dataset = dataset[dataset.speechiness > 0]

# Drop Liveness song 
# => If the concert is live, it is highly likely to be duplicated because existing music sources may exist.
print("drop Liveness")
dataset = dataset[dataset.liveness < 0.9]

# Drop Duration_ms short data 
# => Less than a minute of data is outlier data, which is likely to be advertised music.
print("drop Duration_ms")
dataset = dataset[dataset.duration_ms > 60000] # 1min


# 1-5. Text preprocessing - release date (YYYY)
# ref : https://stackoverflow.com/questions/36505847/substring-of-an-entire-column-in-pandas-dataframe/36506041
print("1-5. Text preprocessing - release date (YYYY)")
dataset['release_date'] = dataset['release_date'].str[0:4]


# 1-6. Data filtering
print("1-6. Data filtering (for 2015~2020)")
years_filtering_1 = dataset['release_date'] >= '2015'
years_filtering_2 = dataset['release_date'] != '2021'
dataset = dataset[years_filtering_1 & years_filtering_2]


# 1-7. Drop "release_date", a column that is no longer useful.
print("1-7. Drop release_date")
dataset = dataset.drop(columns=['release_date'])


# 1-8. Data normalization/standardization(Scaling) & Encoding
print("1-8. Data normalization/standardization(Scaling) & Encoding")

# Set numerical column and categorical column
numerical_feature_list = ['duration_ms', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
categorical_feature_list = ['time_signature', 'key']

# Run scaling & encoding using "scale_encode_combination" function
# (return type : dictionary)
combination_dataset = scale_encode_combination.scale_encode_combination(dataset, numerical_feature_list, categorical_feature_list)
print("=> combination dataset! (10 type)")
pp(combination_dataset)


# MODEL TRAINING STEP=====================================================
# Train each of the 10 datasets.
total_result = {}

for key, mydataset in combination_dataset.items():

    # 2-1. Detach target and feature from the dataset.
    print("2-1. Detach target and feature from the dataset")
    feature = mydataset.drop('popularity', axis=1)
    target = mydataset['popularity']

    # 2-2. Separate the data as train data and test data.
    print("2-1. Separate the data as train data and test data.")
    train_X, test_X, train_Y, test_Y = train_test_split(feature, target, shuffle=True)


    # 2-3. Set models
    # We didn't know which model would be the best, so we tried five models.
    print("2-3. Set models (five type)")
    models = [LinearRegression(), Lasso(), Ridge(), ElasticNet(), XGBRegressor()]
    models_name = ["LinearRegression", "Lasso", "Ridge", "ElasticNet", "XGBRegressor"]


    # 2-4. Set hyper-parameter for GridSearchCV
    print("2-4. Set hyper-parameters for GridSearchCV")

    # for Linear Regression
    Linear_param_grid = {'fit_intercept': [True, False],
                  'normalize': [True, False],
                  'copy_X': [True, False],
                  'n_jobs': [None, 2, 10, 50, 100]}

    # for Lasso, Ridge
    Lasso_Ridge_param_grid = {'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3, 4, 5, 6]}

    # for ElasticNet
    Elastic_param_grid = {'l1_ratio': [0.0001, 0.001, 0.01, 0.1, 1.0],
                       'alpha': [0.0001, 0.01, 1, 2, 3, 4]}

    # for XGBRegressor
    XGB_param_grid = {'learning_rate': [0.3, 0.05, 0.07],
                    'max_depth': [5, 6, 7, 9],
                    'min_child_weight': [4],
                    'subsample': [0.7],
                    'colsample_bytree': [0.7],
                    'n_estimators': [100, 300, 500]}

    prams = [Linear_param_grid, Lasso_Ridge_param_grid, Lasso_Ridge_param_grid, Elastic_param_grid, XGB_param_grid]



    # 2-5. Run GridsearchCV for each dataset, each model.
    print("2-5. Run GridsearchCV for each dataset, each model.")
    for modelname, model, param_grid in zip(models_name, models, prams):
        model_record = {}

        print("=> Train " + modelname + " using " + key)
        start_time = time.time()
        model_test = GridSearchCV(model, param_grid, scoring='r2', cv=3, verbose=1)
        model_test.fit(train_X, train_Y)
        print("The time that this function finish :", time.time() - start_time)

        # 2-6. Show the results of evaluation
        print("2-6. Show the results of evaluation")
        best_model = model_test.best_estimator_
        print(key + ' model 1\'s best estimator: ', best_model)
        print(key + 'model 1\'s best parameters: ', model_test.best_params_)
        print(key + 'model 1\'s best score  : ', best_model.score(test_X, test_Y))

        model_record['estimator'] = best_model
        model_record['parameters'] = model_test.best_params_
        model_record['score'] = best_model.score(test_X, test_Y)

        total_result[key + "_" + modelname] = model_record

        # Save best model weight
        save_name = "Regression_" + key + "_" + modelname 
        with open(save_name + '.pkl', 'wb') as f:
            pickle.dump(model_test, f)


pp(total_result)