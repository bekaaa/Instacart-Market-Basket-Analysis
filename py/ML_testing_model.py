#!/usr/bin/env python

import pickle
import log
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from per_user_product_freq import user_product_freq


log.LOG_PATH = './data/ML/'
log.init('log.log')
log.msg('*********************************************')
log.msg('log file initialized')
log.msg('\t\t\t ** predicting test data, script : ML_testing_model.py')
log.msg('reading data files....')
#------------------------------------------------------
filename = './data/ML/model0.pkl'
with open(filename, 'rb') as f:
	model     = pickle.load(f)
DA_orders     = pd.read_csv('./data/data/orders.csv')
DA_products   = pd.read_csv('./data/data/order_products__prior.csv')
products_freq = pd.read_csv('./data/data/products_freq.csv', index_col='Unnamed: 0')
#---------------------------------------------------------
log.msg('data is ready')
#---------------------------------------------------------
predictions = dict()
#---------------------------------
def predict_order(test_order):
	u          = int(test_order.user_id)
	u_orders   = DA_orders[ DA_orders.user_id == u ]
	u_products = DA_products[ DA_products.order_id.isin(u_orders.order_id.tolist()) ]

	C          = (int(test_order.order_dow) * 24) + int(test_order.order_hour_of_day)
	X          = user_product_freq( u_orders, u_products )
	X          = X.apply(lambda d: 5*d + products_freq.loc[d.name].reset_index(drop=True), axis=1)
	X          = X[[ ( C + i ) % 168 for i in range(0,168) ]]
	X.columns  = range(0,168)
	p = model.predict(X)
	pred = []
	Xind = X.index
	for ind,val in zip(range(len(p)),p):
		if val == 1 :
			pred.append(Xind[ind])
	global predictions
	predictions[int(test_order.order_id)] = ' '.join([str(i) for i in pred])
	return
#----------------------------------------
test_orders = DA_orders[DA_orders.eval_set == 'test']
max_ = 75000 # number of test samples.
size = 75000
#---------------------------------------------
log.msg('testing data with size %d' % size )
eta = .66 * size
log.msg('estimated time is %.2f seconds, or %.2f minutes, or %.2f hours' %\
	(eta, eta / 60, eta / 120) )
#---------------------------------------------
t = np.datetime64('now')
test_orders.iloc[range(size)].apply(predict_order, axis=1)
log.msg('it took {}'.format(str(np.datetime64('now')-t)))
#-----------------------------------------------
log.msg('done, saving to a csv file')
#----------------------------------------------
outputfile = './data/ML/predictions.csv'
with open(outputfile, 'wb') as f:
	f.write('order_id,products\n')
	for k,v in predictions.items() :
		f.write('{},{}\n'.format(k,v))
#--------------------------------------------------
log.msg('DONE, exiting')
log.close()
#---------------------------------------------
