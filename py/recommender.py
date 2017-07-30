#!/usr/bin/env python
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
import log
from per_user_product_freq import user_product_freq

def modelfit(clf, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

	if useTrainCV:
		xgb_param = clf.get_xgb_params()
		xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)

		cvresult = xgb.cv(
			xgb_param,
			xgtrain,
			num_boost_round=clf.get_params()['n_estimators'],
			nfold=cv_folds,
			metrics='auc',
			early_stopping_rounds=early_stopping_rounds,
			show_progress=False
			)

		clf.set_params(n_estimators=cvresult.shape[0])

	#Fit the algorithm on the data
	clf.fit(dtrain[predictors], dtrain['Disbursed'],eval_metric='auc')

	#Predict training set:
	dtrain_predictions = clf.predict(dtrain[predictors])
	dtrain_predprob = clf.predict_proba(dtrain[predictors])[:,1]

	#Print model report:
	print "\nModel Report"
	print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
	print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)

	feat_imp = pd.Series(clf.booster().get_fscore()).sort_values(ascending=False)
	feat_imp.plot(kind='bar', title='Feature Importances')
	plt.ylabel('Feature Importance Score')


if __name__ == '__main__' :

	#---------------------------------------------
	log.LOG_PATH = './data/ML/'
	log.init('log.log')
	log.msg('**************************************')
	log.msg('log file initialized')
	log.msg('\t\t\t** Xgboost using CV , script : ML_training_model.py')
	log.msg('reading X and Y files')
	#-----------------------------------------------
	X = pd.read_csv('./data/ML/X__1000.csv',index_col='Unnamed: 0')
	Y = pd.read_csv('./data/ML/Y__1000.csv',index_col='Unnamed: 0')
	#-------------------------------------------------
	log.msg('X and Y dataframes are ready')
	#--------------------------------------------------------------------
	log.msg('Initializing and training the model')

	#Choose all predictors except target & IDcols
	predictors = [x for x in train.columns if x not in [target, IDcol]]

	xgb1 = XGBClassifier(
		learning_rate = 0.1,
		n_estimators = 1000,
		max_depth = 5,
		min_child_weight = 1,
		gamma = 0,
		subsample = 0.8,
		colsample_bytree = 0.8,
		objective = 'binary:logistic',
		nthread = 4,
		scale_pos_weight = 1,
		seed = 27
		)
	modelfit(xgb1, train, predictors)
