import os
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
from . import dispatcher
import joblib

TRAINING_DATA = os.environ.get('TRAINING_DATA')
TEST_DATA = os.environ.get('TEST_DATA')
MODEL = os.environ.get('MODEL')
def predict():
    df = pd.read_csv(TEST_DATA)
    test_idx = df['id'].values
    predictions = None
    for FOLD in range(5):
        df = pd.read_csv(TEST_DATA)
        label_encoders = joblib.load(os.path.join('models',f'{MODEL}_{FOLD}_label_encoder.pkl'))
        columns = joblib.load(os.path.join('models',f'{MODEL}_{FOLD}_columns.pkl'))
        for c in columns:
            lbl = label_encoders[c]
            df.loc[:,c] = lbl.transform(df[c].values.tolist())

        clf = joblib.load(os.path.join('models',f'{MODEL}_{FOLD}.pkl'))
        df = df[columns]
        preds = clf.predict_proba(df)[:,1]
        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds
    predictions /= 5

    sub = pd.DataFrame(np.column_stack((test_idx.astype(int), predictions)), columns=['id', 'target'])
    sub['id'] = sub['id'].astype(int)
    return sub
if __name__ == "__main__":
    print('Model', MODEL)
    submission = predict()
    submission.to_csv(f'models/{MODEL}_sub.csv', index=False)