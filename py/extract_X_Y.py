#!/usr/bin/env python
import pandas as pd
import numpy as np
import log
from per_user_product_freq import user_product_freq
log.LOG_PATH = './data/ML/'
log.init('log.log')
log.msg('**************************************')
log.msg('log file initialized')
log.msg('\t\t\t** Extraction X and Y , script : extract_X_Y.py')
#--------------------
# reading data files
products_freq = pd.read_csv('./data/data/products_freq.csv', index_col='Unnamed: 0')
DA_orders     = pd.read_csv('./data/data/orders.csv')
DA_products   = pd.read_csv('./data/data/order_products__prior.csv')
DA_train      = pd.read_csv('./data/data/order_products__train.csv')
#------------------------------------
# normalizing
#products_freq  = products_freq.apply(lambda d : ( d + d.mean() ) / d.max(), axis=0 )
products_freq.columns = range(168,336);
#--------------------------
# initialize X and Y dataframes
X = pd.DataFrame(columns=range(0,336))  # 168*2
Y = pd.DataFrame()
#------------------------------
def get_X_Y(order):
	# order is a Series object.
	u            = int(order.user_id)
	u_orders     = DA_orders[ DA_orders.user_id == u ]
	u_products   = DA_products[ DA_products.order_id.isin(u_orders.order_id.tolist()) ]

	X_u                 = user_product_freq( u_orders, u_products )
	X_u[range(168,336)] = products_freq

	C            = (int(order.order_dow) * 24) + int(order.order_hour_of_day)
	cols         = [ ( C + i ) % 168 for i in range(0,168) ]
	for i in cols[:]:
		cols.append(i+168)
	X_u          = X_u[cols]
	X_u.columns  = range(0,336)

	n,_          = X_u.shape # training examples x features
	Y_u          = pd.Series(index=X_u.index, data=np.zeros(n))
	y_true       = DA_train[DA_train.order_id == order.order_id].product_id
	for t in y_true :
		try :
			if Y_u.loc[t] == 0 :
				Y_u.loc[t] = 1
		except KeyError :
			pass
	global X, Y
	X = pd.concat([X, X_u])
	Y = pd.concat([Y, Y_u])
	return
#------------------------------------------
total_size = 131209
size = 1000
orders_train   = DA_orders[DA_orders.eval_set == 'train']
#orders_train   = orders_train.iloc[range(0,size)]
orders_train = orders_train.sample(n=size)
#----------------------------------------------
eta = size * 0.7
t = np.datetime64('now')
log.msg('getting X and Y for training orders from 0 to %d'%size)
log.msg('estimated time is %.2f seconds, or %.2f minutes or %.2f hours' % \
	( eta, eta/60, eta/60/60 ))
#---------------------------------------
_ = orders_train.apply(get_X_Y, axis=1)
#-----------------------------------
log.msg('it took {}'.format(str(np.datetime64('now')-t)))
log.msg('X and Y are ready now.')
log.msg('saving to csv files...')
#---------------------------------
# save to CSV files
X.to_csv('./data/ML/X_CV_%d.csv' % size)
Y.to_csv('./data/ML/Y_CV_%d.csv' % size)
#--------------------------------
log.msg('DONE, exiting')
log.close()
#------------------------------------
