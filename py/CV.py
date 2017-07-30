#!/usr/bin/env python
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import model_selection, metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV
import matplotlib.pylab as plt
import log

def modelfit(clf, X, Y,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
	if useTrainCV:
		xgb_param = clf.get_xgb_params()
		log.msg('getting training matrix')
		xgtrain = xgb.DMatrix(X.values, label=Y.values)

		cvresult = xgb.cv(
			xgb_param,
			xgtrain,
			num_boost_round=clf.get_params()['n_estimators'],
			nfold=cv_folds,
			metrics='auc',
			early_stopping_rounds=early_stopping_rounds,
			#show_progress=False
			)

		clf.set_params(n_estimators=cvresult.shape[0])

	#Fit the algorithm on the data
	log.msg('fitting model ...')
	clf.fit(X, Y['0'].ravel(),eval_metric='auc')
	log.msg('done fitting')
	#Predict training set:
	log.msg('predicting #1 ...')
	dtrain_predictions = clf.predict(X)
	log.msg('predicting #2 ...')
	dtrain_predprob = clf.predict_proba(X)[:,1]
	log.msg('done.')
	#Print model report:
	print "\nModel Report"
	print "Accuracy : %.4g" % metrics.accuracy_score(Y.values, dtrain_predictions)
	print "AUC Score (Train): %f" % metrics.roc_auc_score(Y, dtrain_predprob)

	xgb.plot_importance(clf, max_num_features=30)
	plt.show()

	return clf
#-------------------------------------------------------------------
def save_model(clf) :
	try :
		f = os.listdir('./data/ML')
		n = max([  int(i.split('model')[1].split('.pkl')[0]) for i in f if 'model' in i ]) + 1
	except :
		n = 1
	log.msg('saving to model%d.pkl'%n)
	with open('./data/ML/model%d.pkl'%n, 'wb') as f:
		pickle.dump(clf, f)
#--------------------------------------------------------------------------------
if __name__ == '__main__' :

	#---------------------------------------------
	log.LOG_PATH = './data/ML/'
	log.init('log.log')
	log.msg('**************************************')
	log.msg('log file initialized')
	log.msg('\t\t\t** Xgboost using CV , script : xgboost.py')
	log.msg('reading X and Y files')
	#-----------------------------------------------
	X = pd.read_csv('./data/ML/X_CV_100.csv',index_col='Unnamed: 0')
	Y = pd.read_csv('./data/ML/Y_CV_100.csv',index_col='Unnamed: 0')
	X.columns = range(0,336)
	print X.shape
	#-------------------------------------------------
	log.msg('X and Y dataframes are ready')
	#--------------------------------------------------------------------
	log.msg('Initializing and training the model')

	#Choose all predictors except target & IDcols

	xgb1 = XGBClassifier(
		reg_alpha = 100,
		reg_lambda = 100,
		learning_rate = 0.1,
		n_estimators = 1000,
		max_depth = 6,
		min_child_weight = 1,
		gamma = 0,
		subsample = 0.65,
		colsample_bytree = 0.75,
		objective = 'binary:logistic',
		n_jobs = 4,
		scale_pos_weight = 1,
		random_state = 27
		)
	param_test1 = {
		'reg_alpha': [100, 120, 150],
		'reg_lambda':[100, 120, 150]
		}
	#gsearch1 = GridSearchCV(xgb1,
	#	param_grid = param_test1, scoring='roc_auc',n_jobs = 4, iid = False, cv = 2,verbose = 50)

	#gsearch1.fit(X, Y['0'].ravel())

	#print gsearch1.grid_scores_, '\n' ,gsearch1.best_params_,'\n' ,gsearch1.best_score_

	clf = modelfit(xgb1, X, Y)
	save_model(clf)
	log.msg('DONE\nExiting...')
	log.close()
