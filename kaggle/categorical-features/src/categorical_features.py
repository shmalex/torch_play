"""
Hot to handle categorical data



+ label encoding
+ one hot encoding
+ binarization
- Entity Embedings for categorical variable
"""
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
import tqdm
from typing import List
from enum import Enum, auto

class CategoricalFeaturesType(Enum):
	LABEL:auto()
	BINARY:auto()
	ONEHOT:auto()
class CategoricalFeatures:

	def __init__(self,
				 df: pd.DataFrame,
				 categorical_features: list,
				 encoding_type: str,
				 handle_na=False
				 ) -> None:
		"""
		df: data, should not contain NA values
		categorical_features: list of column names 
		encoding_type: label, binary oneHot
		"""
		self.df = df
		self.cat_feat = categorical_features
		self.enc_type = encoding_type
		self.label_encoders = dict()
		self.binary_encoder = dict()
		self.onehot = None
		self.handle_na = handle_na
		if self.handle_na:
			for c in self.cat_feat:
				self.df.loc[:, c] = self.df.loc[:,c].astype(str).fillna("-999")
		self.output_df = self.df.copy(deep=True)

	def fit(self, data):
		pass

	def _label_encodings(self):
		# for every column
		for c in tqdm.tqdm(self.cat_feat, desc='Labeling Data',disable=False):
			lbl = preprocessing.LabelEncoder()
			lbl.fit(self.df[c].values)
			self.output_df.loc[:,c] = lbl.transform(self.df[c].values)
			self.label_encoders[c] = lbl
		return self.output_df
	def _binarization(self):
		for c in tqdm.tqdm(self.cat_feat, desc='Binarization Data'):
			lbl = preprocessing.LabelBinarizer()
			lbl.fit(self.df[c].values)
			bindata = lbl.transform(self.df[c].values)
			self.output_df.drop(c, axis=1, inplace=True)
			for j in range(bindata.shape[1]):
				new_column = c + f'__bin__{j}'
				self.output_df[new_column] = bindata[:,j]
				self.binary_encoder[c] = lbl
		return self.output_df
	def _onehotencoding(self):
		self.onehot = preprocessing.OneHotEncoder()
		self.onehot.fit(self.df[self.cat_feat].fillna("-999").values)
		return self.onehot.transform(self.df[self.cat_feat].fillna("-999").values)
	def fit_transform(self):
		if self.enc_type == 'label':
			return self._label_encodings()
		elif self.enc_type == 'binary':
			return self._binarization()
		elif self.enc_type == 'onehot':
			return self._onehotencoding()
		else:
			raise Exception('Encoding type not understood')

	def transform(self, df:pd.DataFrame):
		if self.handle_na:
			for c in self.cat_feat:
				df.loc[:,c] = df[c].astype(str).fillna("-999")
		if self.enc_type == 'label':
			for c, lbl in tqdm.tqdm(self.label_encoders.items(), desc='Labling Data',disable=False):
				df.loc[:,c] = lbl.transform(df[c].values)
			return df
		elif self.enc_type == 'binary':
			output = df.copy(deep=True)
			for c, lbl in tqdm.tqdm(self.binary_encoder.items(), desc='Binarization Data'):
				output.drop(c, axis=1, inplace=True)
				vals = lbl.transform(df[c].values)
				for i in range(vals.shape[1]):
					output[c+f'__bin_{i}'] = vals[:,i]
			return output
		elif self.enc_type == 'onehot':
			return self.onehot.transform(df[self.cat_feat].values)
		else:
			raise Exception('Encoding type unknown')
if __name__ == "__main__":
	begin ='<'.join(['']*100)
	line = '-'.join(['']*100)
	 
	# if label
	if False:
		print(begin)
		df = pd.read_csv('./input/train_cat2.csv').head(1000)
		print(df.columns)

		cols = [c for c in df.columns if c not in ['id','target']]
		cat_feat = CategoricalFeatures(df, 
										cols,
										'label',
										handle_na=True)
		output_df = cat_feat.fit_transform()
		print(output_df.columns.tolist())
		print(output_df.head())
		print(output_df.info())
	# Binary 
	if False:
		print(begin)
		df = pd.read_csv('./input/train_cat2.csv').head(10)
		df_test = pd.read_csv('./input/test.csv').head(10)
		df_test['target'] = -1
		print('\nTrain:',df.columns.tolist())
		print('Test:',df_test.columns.tolist())
		fulldata = pd.concat((df, df_test))
		cols = [c for c in df.columns if c not in ['id','target']]
		cat_feat = CategoricalFeatures(fulldata, 
										cols,
										'binary', #'label',
										handle_na=True)
		full_data = cat_feat.fit_transform()
		train_df = full_data[full_data['id'].isin(df['id'].values)].reset_index(drop=True)
		test_df = full_data[full_data['id'].isin(df_test['id'].values)].reset_index(drop=True)
		print(line)
		print(train_df.shape)
		print(train_df.columns.tolist())
		print(train_df.head())
		print(train_df.info())
		print(line)
		train_df = cat_feat.transform(df_test)
		print(train_df.shape)
		print(train_df.columns.tolist())
		print(train_df.head())
		print(train_df.info())

	# one hot
	if True:
		print(begin)
		df = pd.read_csv('./input/cat/train.csv')
		df_test = pd.read_csv('./input/cat/test.csv')
		df_sub = pd.read_csv('./input/cat/sample_submission.csv')
		df_test['target'] = -1
		print('\nTrain:',df.columns.tolist())
		print('Test:',df_test.columns.tolist())
		train_len = len(df)
		fulldata = pd.concat((df, df_test))
		cols = [c for c in df.columns if c not in ['id','target']]
		cat_feat = CategoricalFeatures(fulldata, 
										cols,
										'onehot',
										handle_na=True)
		full_data = cat_feat.fit_transform()
		X = full_data[:train_len]
		X_test = full_data[train_len:]
		print(X.shape)
		print(X_test.shape)
		from sklearn import linear_model
		clf = linear_model.LogisticRegression()
		clf.fit(X, df['target'].values)
		pred = clf.predict_proba(X_test)[:,1]
		df_sub.loc[:,'target'] = pred
		df_sub.to_csv('./input/cat/output-submition.csv', index=False)