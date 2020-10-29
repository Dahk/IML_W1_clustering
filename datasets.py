import os
import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from collections import defaultdict


def load_kropt():
    # Read input dataset
    dataset = os.path.join('datasets', 'kropt.arff')
    data = loadarff(dataset)
    df_data = pd.DataFrame(data[0])

    df_describe = df_data.describe().reset_index()
    feat = list(df_data.columns)
    feat.remove('game')
    #feat.remove('Class')

    #print(feat)

    # Pre-Processing

    # Convert label features into numerical with Label Encoder
    le = LabelEncoder()
    label_encoder = np.zeros((df_data.shape[0], df_data.shape[1] - 1))

    for i in range(len(feat)):
        le.fit(df_data[feat[i]])
        label_encoder[:, i] = le.transform(df_data[feat[i]])

    y = le.fit_transform(df_data['game'])
    #df_data_pp_prev = pd.DataFrame(label_encoder, columns=feat)
    #df_data_pp_prev = df_data_pp_prev.describe().reset_index()

    scaler = MinMaxScaler()
    scaler.fit(label_encoder)
    data_trans_scaled = scaler.transform(label_encoder)

    df_data_pp = pd.DataFrame(data_trans_scaled, columns=feat)
    
    return df_data_pp, y


def load_satimage():
    path = os.path.join('datasets', 'satimage.arff')
    data_satimage, meta_satimage = loadarff(path)    
    df_satimage = pd.DataFrame(data_satimage)
    
    # Drop NaN values
    df_satimage.dropna(inplace=True)
    df_satimage.reset_index(drop=True,inplace=True)

    # Save cluster columns for accuracy
    cluster_satimage = pd.DataFrame(df_satimage["clase"])
    cluster_satimage = cluster_satimage.astype(int) - 1

    # Delete cluster columns
    del df_satimage["clase"]

    return df_satimage, cluster_satimage


def load_credita():
    path = os.path.join('datasets', 'credit-a.arff')
    raw_data = loadarff(path)
    df = pd.DataFrame(raw_data[0])

    y = df.pop('class')
    X = df

    y_label_encoder = LabelEncoder()
    y = y_label_encoder.fit_transform(y)

    # fill missing numerical values
    X.fillna(X.mean(), inplace=True)

    # fill missing categorical values
    categ_cols = X.select_dtypes(include=['category', object]).columns
    for col in categ_cols:
        X[col].replace(b'?', X[col].mode()[0], inplace=True)

    # standarize numerical features
    num_cols = X.select_dtypes(include=['number']).columns
    mm_scaler = MinMaxScaler()
    X[num_cols] = mm_scaler.fit_transform(X[num_cols])

    # use one transformer per feature to preserve its name in the generated features
    # since new feature names are based on the transformer's name
    transformers = [(col, OneHotEncoder(drop='first'), [col]) for col in categ_cols]
    col_transformer = ColumnTransformer(transformers, remainder='passthrough')
    X_arr = col_transformer.fit_transform(X)

    X = pd.DataFrame(X_arr, columns=col_transformer.get_feature_names())

    return X, y






def load_waveform():
    raw_data = loadarff('datasets/waveform.arff')
    df = pd.DataFrame(raw_data[0])

    y = df.pop('class')
    X = df

    # encode class
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    scaler = MinMaxScaler()
    X_arr = scaler.fit_transform(X)
    X = pd.DataFrame(X_arr, columns=X.columns)
    return X, y
    

def load_soybean():
    raw_data = loadarff('datasets/soybean.arff')
    df = pd.DataFrame(raw_data[0])
    
    y = df.pop('class')
    X = df

    # encode class
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    encoder = OneHotEncoder(drop='first')
    X_arr = encoder.fit_transform(X)
    X = pd.DataFrame(X_arr.todense(), columns=encoder.get_feature_names())

    return X, y


def load_sick():
    raw_data = loadarff('datasets/sick.arff')
    df = pd.DataFrame(raw_data[0])

    y = df.pop('class')
    X = df

    X.drop('TBG', axis=1, inplace=True)   # all NaN, useless

    implicit_cols = [col for col in X.columns if col.endswith('_measured')]
    X.drop(implicit_cols, axis=1, inplace=True)

    # Replace NaN values
    X.fillna(X.mean(), inplace=True)
    X['sex'].replace(b'?', X['sex'].mode()[0], inplace=True)

    # Standarize numerical features
    num_cols = X.select_dtypes(include=['number']).columns
    scaler = MinMaxScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    # Encode categorical features
    categ_cols = X.select_dtypes(include=['category', object]).columns
    categ_cols = categ_cols.drop('referral_source')

    # we use a dict where each feature has an entry with its encoder
    # for future or inverse transformations
    label_encoders = defaultdict(LabelEncoder)
    X[categ_cols] = X[categ_cols].apply(lambda x: label_encoders[x.name].fit_transform(x))

    ohe_encoder = OneHotEncoder()   # save for future or inverse transformations
    ohe_transformer = ColumnTransformer([('referral_source', ohe_encoder, ['referral_source'])], remainder='passthrough')
    X_arr = ohe_transformer.fit_transform(X)
    
    X = pd.DataFrame(X_arr, columns=ohe_transformer.get_feature_names())

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    candidates = []
    for eps in range(1, 11):
        for ms in range(4, 21):
            model = DBSCAN(eps=eps/10, min_samples=ms).fit(X)
            counts = np.unique(model.labels_, return_counts=True)[1]
            if len(counts) == 3:
                print(model, counts)
                candidates.append(model)