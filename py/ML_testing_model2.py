#!/usr/bin/env python

import pickle
import log
import pandas as pd
import numpy as np


log.LOG_PATH = './data/ML/'
log.init('log.log')
log.msg('*********************************************')
log.msg('log file initialized')
log.msg('\t\t\t ** predicting test data, script : ML_testing_model.py')
log.msg('reading data files....')
#------------------------------------------------------
modelfile = './data/ML/model1.pkl'
Xfile     = './data/ML/X_CV_100.csv'
Yfile     = './data/ML/X_CV_100.csv'
with open(modelfile, 'rb') as f :
	model = pickle.load(f)
X = pd.read_csv(Xfile,index_col='Unnamed: 0')
Y = pd.read_csv(Yfile,index_col='Unnamed: 0')
X.columns = range(0,336)
#---------------------------------------------------------
log.msg('data is ready')
#---------------------------------------------------------
log.msg('predicting data with size %d' % len(X) )
#---------------------------------------------
t = np.datetime64('now')
predictions = model.predict(X)
log.msg('it took {}'.format(str(np.datetime64('now')-t)))
#-------------------------------------------------------





#---------------------------------------------
log.msg('saving to a csv file')
#----------------------------------------------
outputfile = './data/predictions/predictions.csv'
with open(outputfile, 'wb') as f:
	f.write('order_id,products\n')
	for k,v in predictions.items() :
		f.write('{},{}\n'.format(k,v))
#--------------------------------------------------
log.msg('DONE, exiting')
log.close()
#---------------------------------------------
