#! /usr/bin/env python
import pandas as pd
import numpy as np
import pickle
import log
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#----------------------------------------------
log.LOG_PATH = './data/ML/'
log.init('log.log')
log.msg('**************************************')
log.msg('log file initialized')
log.msg('\t\t\t** Training model , script : ML_model_score_.py')
#------------------------------------------------
with open('./data/ML/model4.pkl','rb') as f:
	model = pickle.load(f)
#------------------------------------------------
log.msg('reading X and Y files')
#-----------------------------------------------
X = pd.read_csv('./data/ML/X__100.csv',index_col='Unnamed: 0')
Y = pd.read_csv('./data/ML/Y__100.csv',index_col='Unnamed: 0')
#-------------------------------------------------
log.msg('X and Y dataframes are ready')
#--------------------------------------------------
log.msg('getting sample data..')
sample_ratio = .2
rand = 40
_, Xt, _, Yt = train_test_split(X, Y, test_size = sample_ratio, random_state = rand)
#del X, Y
#----------------------------------------------------------
log.msg('predicting..')
pred = model.predict(X)
log.msg(classification_report(Y, pred))
#-------------------------------------------------------------
log.msg('DONE, exiting..')
log.close()
