import pandas as pd
import random
from ETE_scaling import scale_encode_combination
from pprint import pprint as pp
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score


import pickle, joblib
import time

# DATA PREPROCESSING STEP 1=====================================================
# 1-1. Load data
dirty_data = pd.read_csv('dirty_tracks.csv')
dataset = dirty_data

# 1-2. Checking data
print("dataset information")
#print(dataset.info())
#print(dataset.describe())
#print(dataset.shape)


# 1.3. Fill missing data
print("fill missing data")
#print(dataset.isnull().sum())
dataset = dataset.fillna(dataset.median())
dataset = dataset.apply(lambda x: x.fillna(x.value_counts().index[0]))
#print(dataset.isnull().sum())

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


print(dataset.shape)
print(dataset['release_date'].isna().sum())



# 1.5 Modity release date (YYYY -> YYY (first 3 chat))
# ref : https://stackoverflow.com/questions/36505847/substring-of-an-entire-column-in-pandas-dataframe/36506041
dataset['release_date'] = dataset['release_date'].str[0:3]
#class에 값이 1개 있는 190은 drop
dataset = dataset[dataset.release_date != '190']



feature = dataset.drop('release_date',axis=1)
target = dataset['release_date']


# DATA PREPROCESSING STEP 2=====================================================
# 2.1 Set numerical column and categorical column
numerical_feature_list = ['duration_ms', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
categorical_feature_list = ['time_signature', 'key']


# 2.2 Run scaling & encoding using "scale_encode_combination" function (return type : dictionary)
print("Scaling and Encoding feature")
combination_dataset = scale_encode_combination(feature, numerical_feature_list, categorical_feature_list)
#pp(combination_dataset)




# MODEL TRAINING STEP=====================================================
for key, mydataset in combination_dataset.items():

    print("Set traom and test")
    print(feature.shape)
    print(target.shape)


    # 3.2 Split train and test
    train_X, test_X, train_Y, test_Y = train_test_split(feature, target.ravel(), test_size = .3, shuffle=True, stratify=target)
    print(train_X)
    print(train_Y)
    print(train_X.shape)
    print(train_Y.shape)

    train_X, val_X, train_Y, val_Y= train_test_split(train_X, train_Y, test_size=0.3, random_state=12)

    # 3.3 Set model
    print("Set model")
    model = RandomForestClassifier()



    # for XGBRegressor
    param_grid = { 
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth' : [3, 5, 7],
    'criterion' :['gini', 'entropy']
    }


    # 3.5 Define an evaluation metric as root mean squared error in the scoring parameter
    # 3.6 Set and RUN GridSearchCV
    start_time = time.time()
    model_test = GridSearchCV(model, param_grid, cv=5, n_jobs=4, verbose=2)
    model_test.fit(train_X, train_Y)
    print("The time that this function finish :", time.time() - start_time)


    # 3.6 Show Result & Evaluation
    print('model 2 best estimator: ', model_test.best_estimator_)
    print('model 2 best parameters: ', model_test.best_params_)

        
    best_model = model_test.best_estimator_
    predict_test = best_model.predict(test_X)
    print('Best model 2 score : ', best_model.score(test_X, test_Y))
    print("Accuracy for Random Forest on CV data: ", accuracy_score(test_Y, predict_test))


    #파일 이름 구성 -> combination type + classification인지 regresison인지 + dirty data/originaldata + 몇번쨰 시도인지 + 그외 저장해야 하는 정보
    with open(key + '_rf' + '_model2_classification_dirtydata.pkl', 'wb') as f:
        pickle.dump(model_test, f)
