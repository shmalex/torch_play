# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# https://mmuratarat.github.io/2019-06-12/embeddings-with-numeric-variables-Keras
from __future__ import absolute_import, division, print_function, unicode_literals

from numpy.core import numeric
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
__all__ = []
import types
from tensorflow.python.keras.engine.base_layer import Layer
import tqdm
import time
import argparse
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing, metrics

import tensorflow as tf
from tensorflow.keras import layers, optimizers, callbacks
from tensorflow.keras.layers.experimental import preprocessing as ker_preprocessing

from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
tf.compat.v1.disable_eager_execution()

# DEBUG!
# tf.compat.v1.enable_eager_execution()

from . import cross_validation as cvm
from . import categorical_features as cfm
from . import metrics as mm
from . import utils as ut

ap = argparse.ArgumentParser()
ap.add_argument('-l1','--nlayer1',dest='nlayer1', type=int)
ap.add_argument('-l2','--nlayer2',dest='nlayer2', type=int)
ap.add_argument('-e','--epochs',dest='epochs', type=int)
ap.add_argument('-f','--num_folds',dest='num_folds', type=int)
args = ap.parse_args()
ut.print_arguments(args)
nlayer1 = args.nlayer1
nlayer2 = args.nlayer2
line = '-'.join(['']*50)
epochs = args.epochs
num_folds = args.num_folds
if not os.path.exists('./input/transformed_data_expanded.csv'):
    train = pd.read_csv('./input/train_expanded.csv')
    test = pd.read_csv('./input/test_expanded.csv')
    features = train.columns.difference(['id', 'target']).tolist()
    print('Features', features, len(features))

    cv = cvm.CrossValidation(train,
                            ['target'],
                            shuffle=False,
                            problem_type=cvm.PT_MULTI_LABLE_CLF,
                            num_folds=num_folds)
    train = cv.split()
    print(train.kfold.unique(), train.target.unique())
    print('Total kfolds', train.kfold.unique(),train.kfold.nunique())

    # select features to train on

    # for combination purpose
    test['target'] = -1
    test['kfold'] = -1
    join_data = pd.concat([train, test],axis=0).reset_index(drop=True)

    print('joined', join_data.shape, 'train', train.shape, 'test', test.shape)
    print(join_data.head())
    cf = cfm.CategoricalFeatures(join_data, features,'label', handle_na=True)
    transformed_data = cf.fit_transform()
    print('Transformed Data')
    print(transformed_data.head())
    transformed_data.to_csv('./input/transformed_data_expanded.csv', index=False)
else:
    transformed_data = pd.read_csv('./input/transformed_data_expanded.csv')
    features = transformed_data.columns.difference(['id', 'target','kfold']).tolist()

numerical_features = ('animal_','country_','heat_','rank_', 'inst_', 'rank', 'shape_')
cat_features = list(filter(lambda x: not x.startswith(numerical_features), features))
num_features = list(filter(lambda x:     x.startswith(numerical_features), features))

print('Categorical Features:', len(cat_features), ', '.join(cat_features))
print('Numerical Features:', len(num_features),  ', '.join(num_features))

train_df = transformed_data[transformed_data.target != -1].reset_index(drop=True)
test_df  = transformed_data[transformed_data.target == -1].reset_index(drop=True)

# train_df = cf.transform(train)
# test_df = cf.transform(test)

print(train_df.shape, test_df.shape)
print('train kfold', train_df.kfold.unique(), 'target', train_df.target.unique())
print('test kfold',   test_df.kfold.unique(), 'target',  test_df.target.unique())
log_name = f'exp_{int(time.time())}'

def auc(y_true, y_pred):
    def fallback_auc(y_true, y_pred):
        try:
            return metrics.roc_auc_score(y_true, y_pred)
        except:
            return 0.5
    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)
def get_model(df, cat_cols, numerical_cols, nlayer1, nlayer2):
    inputs = []
    outputs = []
    for c in cat_cols:
        num_unique = int(df[c].nunique())
        embed_dim = int(min(np.ceil(num_unique/2), 60))
        inp = layers.Input(shape=(1,), name=f'Categorical_{c}')
        out = layers.Embedding(num_unique + 1, embed_dim, name=f'Embedding_{c}')(inp)
        out = layers.SpatialDropout1D(0.3,name=f'SpatialDropout1D_{c}')(out)
        # apply dropout here
        out = layers.Reshape(target_shape=(embed_dim,), name=f"Reshape_{c}")(out)
        inputs.append(inp)
        outputs.append(out)

    # if len(numerical_cols) > 0:
    inp_num_data = layers.Input(shape=(len(numerical_cols),), name=f'Input_Numerical')
    # inp_normalizer = ker_preprocessing.Normalization(input_shape=[1,])
    # num_norm = inp_normalizer(inp_num_data)
    x2 = layers.BatchNormalization()(inp_num_data)
    x2 = layers.Dense(64, activation='relu', name=f'Dense1_Numerical')(x2)
    x2 = layers.Dropout(.3)(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dense(64, activation='relu', name=f'Dense2_Numerical')(x2)
    x2 = layers.Dropout(.3)(x2)


    x = layers.Concatenate()(outputs)
    x = layers.concatenate([x, x2])
    x = layers.BatchNormalization()(x)
    x = layers.Dense(nlayer1, activation='relu')(x)
    x = layers.Dropout(.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(nlayer2, activation='relu')(x)
    x = layers.Dropout(.3)(x)
    x = layers.BatchNormalization()(x)

    y = layers.Dense(1, activation='sigmoid')(x)
    # y = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=[inputs, inp_num_data], outputs=y )

    return model

test_data = [test_df.loc[:,f].values for f in cat_features]
test_data_nump = test_df[num_features].values
test_data.append(test_data_nump)
# summing the num_flods predictions
test_predictions = np.zeros(len(test_df))
kfold_predictions = np.zeros(len(test_df))

for kfold in range(num_folds):
    # kfold train and test datasets
    kf_train_df = train_df[train_df.kfold.values != kfold]
    kf_test_df  = train_df[train_df.kfold.values == kfold]
    kf_train_df.reset_index(inplace=True)
    kf_test_df .reset_index(inplace=True)
    kf_y_train = kf_train_df.target.values
    kf_y_test  = kf_test_df .target.values
    # prepare data for the model
    kf_train_data = [kf_train_df.loc[:,f].values for f in cat_features]
    # print(kf_train_data)
    kf_train_data_nump = kf_train_df[num_features].values
    kf_train_data.append(kf_train_data_nump)
    # print(kf_train_data)
    kf_test_data  = [kf_test_df .loc[:,f].values for f in cat_features]
    kf_test_data_nump = kf_test_df[num_features].values
    kf_test_data.append(kf_test_data_nump)

    # val_auc === validation AUC
    es_cb = callbacks.EarlyStopping(monitor='val_auc', min_delta=0.001, patience=5, verbose=2, mode='max', baseline=None, restore_best_weights=True)
    rlr_cb = callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, min_lr=1e-6, mode='max', verbose=2)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=f"./logs4/{log_name}_{nlayer1}_{nlayer2}_{kfold}", update_freq=1,histogram_freq=1, )

    print(line,kfold,line)
    print('kfold train', kf_train_df.shape, kf_train_df.kfold.unique())
    print('kfold test', kf_test_df.shape, kf_test_df.kfold.unique())
    model = get_model(kf_train_df, cat_features, num_features, nlayer1, nlayer2)
    opt = optimizers.Adam(learning_rate=0.0005)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[auc])#, run_eagerly=True)
    # print(model.summary())
    print(len(kf_train_data))
    # for i, d in enumerate(kf_train_data):
    #     print(i, d)
    model.fit(kf_train_data,
                kf_y_train,
                batch_size=1024,
                callbacks=[tensorboard_cb, es_cb, rlr_cb],
                epochs=epochs*kfold+epochs,
                validation_data=(kf_test_data, kf_y_test),
                validation_steps=1,
                initial_epoch=epochs*kfold,
                )
                # validation_data=[kf_test_data, kf_y_test])
    print('Predicting... kfold test #', kfold)
    kfold_prediction = model.predict(kf_test_data).ravel()
    # kfold_predictions = kfold_prediction
    
    cm = mm.ClassificationMetrics()
    print(f'KFold:{kfold}\t AUC', cm('auc', kf_y_test, None, y_proba=kfold_prediction))
    print('Predicting... test data')
    test_predictions += model.predict(test_data).ravel()    

    # reset keras state
    # model.save(f'./model/model_{kfold}_{epochs}')
    K.clear_session()

print('Done Training')
test_predictions /= num_folds
print("Predicting Test Data...")
print(test_predictions)

sub = pd.DataFrame.from_dict({'id':test_df['id'].values,'target': test_predictions.ravel()})
print(sub.head())
sub.to_csv(f'./input/submition_{nlayer1}_{nlayer2}.csv', index=False)