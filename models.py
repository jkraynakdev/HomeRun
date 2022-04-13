import pandas as pd
from data import Data
import numpy as np
from numpy import mean
from numpy import std
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix 
from sklearn.metrics import r2_score


class Models:
	''' Handles calculations using company data
	Attributes: 
		data_object: Data type object that allows acessing of pitch features and event outcomes
		sampled_df: DataFrame constructed by randomly sampling 200 homeruns and 200 non-homeruns
		FEATURES: Constant to store pitch features that we have access to
		PITCH_TYPES: Constant to store pitch types
	'''	

	def __init__(self) -> None:
		# Initialize object to access data
		raw_df = pd.read_csv('combined.csv', encoding="utf-8-sig") 
		self.data_object = Data(raw_df)

		# Construct sampled dataframes from Data object
		self.data_object.set_random_samples()
		self.sampled_df = self.data_object.random_concat

		# Constant to store feature options
		self.FEATURES = ['pfx_x', 'pfx_z', 'velo', 'zone', 'spin', 'effective_velo', 'VAA']
		
		# Constant to store pitch types
		#self.PITCH_TYPES = ['SL', 'FF', 'FC', 'SI', 'CH', 'CU', 'FS', 'KC']
		self.PITCH_TYPES = ['SL', 'FF', 'SI', 'CH', 'CU']


	def create_feature_frame(self, df):
		'''Returns a DataFrame of with given features'''

		hor_break = self.data_object.get_horizontal_break(df)
		ver_break = self.data_object.get_vertical_break(df)
		velo = self.data_object.get_velo(df)
		zone = self.data_object.get_zone(df)
		spin = self.data_object.get_spin(df)
		effective_velo = self.data_object.get_effective_velo(df)
		VAA = self.data_object.get_VAA(df)

		return pd.DataFrame({'hor_break': hor_break, 'ver_break': ver_break,
							'velo': velo, 'zone': zone, 'spin': spin, 'effective_velo': effective_velo,
							'VAA': VAA})

	def clean_data(self, df1, df2) -> list:
		'''Pairwise elminates NaN values and returns a list of clean DataFrames'''

		# clean out NaNs
		is_NaN = df1.isnull()
		row_has_NaN = is_NaN.any(axis=1)
		rows_with_NaN = df1[row_has_NaN]
		
		# delete from both
		df1.drop(df1.index[list(rows_with_NaN.index)], inplace=True)
		df2.drop(df2.index[list(rows_with_NaN.index)], inplace=True)

		return [df1, df2]

	def create_outcome_frame(self, df):
		''' Creates a df storing pitch outcomes '''
		return pd.DataFrame({'Homerun': self.data_object.assign_event_cat(df)})

	def feature_elemination(self, pitch_type: str) -> list:
		''' Runs a RFE to select out features and returns the lables of selected features'''
		pitch_type_frame = self.data_object.split_by_pitch_type(self.sampled_df, pitch_type)
		cleaned_data = self.clean_data(self.create_feature_frame(pitch_type_frame), self.create_outcome_frame(pitch_type_frame))
		selected_features = []


		y, X = cleaned_data[1], cleaned_data[0]
		logreg = LogisticRegression(class_weight='balanced', max_iter=3000)

		rfe = RFE(logreg, n_features_to_select=3)
		rfe = rfe.fit(X, y.values.ravel())
		print(rfe.support_)
		print(rfe.ranking_)


		for i in range(len(rfe.ranking_)):
			if rfe.ranking_[i] == 1:
				selected_features.append(self.FEATURES[i])

		return selected_features

	def run_regression(self, pitch_type: str) -> None:
		'''Runs logistic regression models and prints out metrics'''
		pitch_type_frame = self.data_object.split_by_pitch_type(self.sampled_df, pitch_type)
		cleaned_data = self.clean_data(self.create_feature_frame(pitch_type_frame), self.create_outcome_frame(pitch_type_frame))
		#cleaned_data = self.clean_data(self.create_feature_frame(pitch_type_frame), self.create_outcome_frame(pitch_type_frame))

		y, X = cleaned_data[1], cleaned_data[0]
		X = X[X.columns[X.columns.isin(self.feature_elemination(pitch_type))]]
		
		X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.3, random_state=101)


		logmodel = LogisticRegression()
		logmodel = logmodel.fit(X_train, y_train)

		y_pred = logmodel.predict(X_test)

		# Print header
		print('========' + pitch_type + '========')
		# Print metrics
		print(classification_report(y_test, y_pred))
		print(confusion_matrix(y_test, y_pred))

		# Model evaluation
		# define evaluation procedure
		cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
		# evaluate model
		scores = cross_val_score(logmodel, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
		# summarize performance
		print('Mean ROC AUC: %.3f' % mean(scores))

		print('\n\n\n')


if __name__ == '__main__':
	model_object = Models()

	for pitch_type in model_object.PITCH_TYPES:
		model_object.run_regression(pitch_type)

