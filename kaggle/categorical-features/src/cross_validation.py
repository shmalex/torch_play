import pandas as pd
from pandas.core.algorithms import unique
from sklearn import model_selection

"""
Type of problems
+ binary classification
	Cat Tog
+ multi class classification
	Cat Dog Fish
- multi label classification
	Image has : car, bicile, tree
	data set:
	id|target
	 1|1,2,3
	 2|4,6,8,9
	 3|1,4,9
+ single column regression
	House Sale Price
+ multi column regression

+ holdout
	- time series type of data, when you have millions of data points.
"""

PT_BIN_CLF = "bin_clf"
PT_MULTI_CLASS_CLF = "multi_class_clf"
PT_MULTI_LABLE_CLF = "multi_label_clf"
PT_SINGLE_COLUMN_REGR = "single_column_regression"
PT_MULTI_COLUMN_REGR = "multi_column_regression"
PT_HOLDOUT = "holdout"


class CrossValidation:
	def __init__(self, df: pd.DataFrame,
              target_cols: list,
              shuffle: bool,
              problem_type=PT_BIN_CLF,
              num_folds: int = 5,
			  multilabel_delimiter: str = ',',
              random_state: int = 42) -> None:

		self.df = df
		self.target_cols = target_cols
		self.num_targets = len(target_cols)
		self.problem_type = problem_type
		self.num_folds = num_folds
		self.shuffle = shuffle
		self.multilabel_delimiter = multilabel_delimiter
		self.random_state = random_state
		if self.shuffle:
			self.df = self.df.sample(frac=1).reset_index(drop=True)

	def split(self) -> pd.DataFrame:
		if self.problem_type in [PT_BIN_CLF, PT_MULTI_CLASS_CLF]:
			# check if we have only 1 target columns
			if self.num_targets != 1:
				raise Exception('Invalida number of targets')
			unique_values = self.df[self.target_cols[0]].nunique()
			if unique_values == 1:
				raise Exception('Only one unique values found')
			# binary classification
			elif unique_values > 1:
				# check the balance of the data
				target = self.target_cols[0]
				kf = model_selection.StratifiedKFold(self.num_folds,
                                         shuffle=False)
				#  random_state=self.random_state)
				# train_idx contains indices that will be used for training, and could be random across all dataset
				# valid_idx contains indices that are unique across all folds
				for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.df, y=self.df[target])):
					self.df.loc[val_idx, 'kfold'] = int(fold)
		elif self.problem_type in (PT_SINGLE_COLUMN_REGR, PT_MULTI_COLUMN_REGR):
			if self.num_targets != 1 and self.problem_type == PT_SINGLE_COLUMN_REGR:
				raise Exception('Invalid number of targets')
			if self.num_targets < 2 and self.problem_type == PT_MULTI_COLUMN_REGR:
				raise Exception('Invalid number of targets')
			kf = model_selection.KFold(n_splits=self.num_folds)
			for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.df)):
				self.df.loc[val_idx, 'kfold'] = fold
		elif self.problem_type.startswith('holdout_'):
			#holdout_5, holdout_10..
			holdout_percentage = int(self.problem_type.split('_')[1])
			num_holdout_samples = int(len(self.df) * holdout_percentage / 100)

			# assign the values to data
			self.df.loc[:num_holdout_samples, 'kfold'] = 0
			self.df.loc[num_holdout_samples:, 'kfold'] = 1
		elif self.problem_type == PT_MULTI_LABLE_CLF:
			if self.num_targets != 1:
				raise Exception('Invalid number of targets for this problem type')
			# split the targets
			targets = self.df[self.target_cols[0]].apply(lambda x: len(str(x).split(self.multilabel_delimiter)))
			kf = model_selection.StratifiedKFold(n_splits=self.num_folds, shuffle=False)
			for fold, (train_idx, val_idx) in enumerate(kf.split(self.df, targets)):
				self.df.loc[val_idx, 'kfold'] = fold
		else:
			raise Exception('Problem not supported' + self.problem_type)

		self.df['kfold'] = self.df.kfold.astype(int)
		return self.df


if __name__ == "__main__":
	# df = pd.read_csv('./input/train.csv')
	# df = pd.read_csv('./input/house/train.csv')
	df = pd.read_csv('./input/train_met.csv')
	print(df.shape)
	# cv = CrossValidation(df=df ,target_cols=['target'])
	cv = CrossValidation(df=df,
						target_cols=['attribute_ids'],
						# problem_type="holdout_50",
						problem_type=PT_MULTI_LABLE_CLF,
						shuffle=True,
						multilabel_delimiter=' ')
	df_split = cv.split()
	print(df_split.head())
	print(df_split.kfold.value_counts())
	print(df_split.shape)
