import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
from . import dispatcher
import joblib

TRAINING_DATA = os.environ.get('TRAINING_DATA')
TEST_DATA = os.environ.get('TEST_DATA')
# on which fold we want to train on
FOLD = int(os.environ.get('FOLD'))
MODEL = os.environ.get('MODEL')

FOLD_MAPPING = {
    0: [1,2,3,4],
    1: [0,2,3,4],
    2: [0,1,3,4],
    3: [0,1,2,4],
    4: [0,1,2,3]
}

if __name__ == "__main__":
    print('Model', MODEL)
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)

    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold == FOLD]

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df.drop(['id','target','kfold'], axis=1)
    valid_df = valid_df.drop(['id','target','kfold'], axis=1)

    # same columns order
    valid_df = valid_df[train_df.columns]

    label_encoders = {}
    print(train_df.head())
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        # join data to have full information about possible labels
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist() + df_test[c].values.tolist())
        train_df.loc[:,c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:,c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders[c] = lbl

    print(train_df.columns)
    print(train_df.head())
    train_df['days'] = train_df['month'] * 30 + train_df['day']
    exit()
    # data is ready to train

    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:,1]
    print(preds)
    print(metrics.roc_auc_score(yvalid, preds))
    joblib.dump(label_encoders, f'models/{MODEL}_{FOLD}_label_encoder.pkl')
    joblib.dump(clf, f'models/{MODEL}_{FOLD}.pkl')
    joblib.dump(train_df.columns, f'models/{MODEL}_{FOLD}_columns.pkl')