#!/usr/bin/env python

import pickle
import log
import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
#---------------------------------------------
log.LOG_PATH = './data/ML/'
log.init('log.log')
log.msg('**************************************')
log.msg('log file initialized')
log.msg('\t\t\t** Training model , script : ML_training_model.py')
log.msg('reading X and Y files')
#-----------------------------------------------
X = pd.read_csv('./data/ML/X__1000.csv',index_col='Unnamed: 0')
Y = pd.read_csv('./data/ML/Y__1000.csv',index_col='Unnamed: 0')
#-------------------------------------------------
log.msg('X and Y dataframes are ready')
#--------------------------------------------------------------------
log.msg('Initializing and training the model')
params = {
	"objective"           : "reg:logistic",
	"eval_metric"         : "logloss",
	"learning_rate"       : 0.1,
	"max_depth"           : 6,
	"min_child_weight"    : 10,
	"gamma"               : 0.70,
	"subsample"           : 0.76,
	"colsample_bytree"    : 0.95,
	"reg_alpha"           : 2e-05,
	"reg_lambda"          : 10,
	"random_state"        : 40
}
X.columns = range(0,336)
model = XGBClassifier()
log.msg("fitting model")
model.fit(X, Y['0'].ravel())
#--------------------------------------------
log.msg('model fitted successfully')
#--------------------------------------------
try :
	f = os.listdir('./data/ML')
	n = max([  int(i.split('model')[1].split('.pkl')[0]) for i in f if 'model' in i ]) + 1
except :
	n = 1
log.msg('saving to model%d.pkl'%n)
with open('./data/ML/model%d.pkl'%n, 'wb') as f:
	pickle.dump(model, f)
log.msg('DONE, exiting')
log.close()
#--------------------------------------------------------------------
