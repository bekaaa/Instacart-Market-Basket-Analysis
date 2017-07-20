#! /usr/bin/env python
import pandas as pd
import numpy as np
import pickle
import random
import log
from pathos.multiprocessing import ProcessingPool as Pool
from per_user_product_freq_v2 import user_product_freq
'''
WHAT'S NEW on this version :
	* Getting nlargest() for the whole day instead for the given hour only.
	* Also multiply the score for given hour's column by 2, to give it higher priority.
----------------------------------------------
Callable functions:
* predict(list())
	Input : orders ids as a list.
	Output : Dictionary type, key : order id, value : predicted products ids as a list.
* spredict(size=0,rand=False, eval_set="test")
	Input : size of the order_ids list ( sampled from the test data),
		rand : wether to take random x elements or the first x elements in the test data,
		eval_set : either "test" or "train".
	Output : Return dictionary type, key : order id, value : predicted products ids as a list,
		and also write the predictions to a csv file, named as predictions-datetime(now).csv.
* tpredict()
	Input : None
	Output : compute the predictions for the full 75,000 sized test data and write it to a csv file,
		also it generates a log file, as it compute each 1000 raw alone.
		And return None.
'''
products_freq = pd.read_csv('./data/data/products_freq.csv', index_col='Unnamed: 0')
DA_orders     = pd.read_csv('./data/data/orders.csv')
DA_products   = pd.read_csv('./data/data/order_products__prior.csv')

#normalize it.
products_freq = products_freq.apply(lambda d: (d - np.median(d)) / d.max(), axis=1)

def predict(orders_list):
	predictions = dict()
	for order_id in orders_list :
		t = np.datetime64('now');
		# compute user's products freq.
		order         = DA_orders[DA_orders.order_id == order_id];
		user_id       = int(order.user_id);
		user_orders   = DA_orders[ DA_orders.user_id == user_id ];
		user_products = DA_products[ DA_products.order_id.isin(user_orders.order_id) ];
		user_freq     = user_product_freq(user_orders, user_products)
		user_freq     = user_freq.apply(lambda d: d + .5 * products_freq.loc[d.name], axis=1)
		#--------------------------------------------------
		# compute X where x is the predicted number of products in that order,
		# I'll assume that X is the median of number of products for that user.
		sum_of_products = [];
		def AP_get_num_prod(order):
			sum_of_products.append( len(user_products[user_products.order_id == order.order_id]) )
		user_orders.apply(AP_get_num_prod, axis=1)
		X = int(np.median(sum_of_products))
		#-----------------------------------------------------
		# Now get the X highest products.
		computed_hour  = order.order_dow * 24 + order.order_hour_of_day;
		full_day_hours = [ int(order.order_dow) * 24 + i for i in range(0,23) ]
		user_freq[computed_hour] *= 2;
		maxN = user_freq[full_day_hours].nlargest(X,full_day_hours).index
		# Update the predictions dictionary.
		predictions[int(order.order_id)] = list(maxN) ;
		#----------------------------------------------------
	return predictions

def predict_s(size = 0,rand = False, eval_set='test'):
	if size <= 0 or eval_set not in ('test', 'train') : return dict();
	max_ = 75000 if eval_set == 'test' else 1384617
	if size > max_ : return dict();
	#--------------------------------------------
	# set orders ids based on the input parameters.
	test_orders_ids = DA_orders[ DA_orders.eval_set == eval_set ].order_id;
	if rand :
		sample = random.sample(range(0,max_),size)
		test_orders_ids = test_orders_ids.iloc[sample]
	else :
		test_orders_ids = test_orders_ids.iloc[0:size]
	#-----------------------------------------
	#get predictions.
	predictions = predict(test_orders_ids)
	#-------------------------------------------------
	# write predictions to a csv file.
	outputfile = './data/predictions/predictions-%s-%d-%s.csv' % ( eval_set, size, rand )
	with open(outputfile,'wb') as f:
		f.write('order_id,products\n')
		for k,v in predictions.items() :
			products = ' '.join([str(i) for i in v])
			f.write('{0},{1}\n'.format(k,products))
	return predictions

def predict_t():
	# initialize log file.
	filename = 'log-predict_t-%s.log' % str(np.datetime64('now'))
	log.init(filename)
	#-------------------------------------
	# set the ouput csv filename and header.
	outputfile = './data/predictions/predictions-test-%s.csv' % str(np.datetime64('now'))
	with open(outputfile,'ab') as f:
		f.write('order_id,products\n')
	#--------------------------------------------------------------------
	test_orders_ids = DA_orders[ DA_orders.eval_set == 'test' ].order_id;
	max_ = 75000 # number of test samples.
	step = 1000
	for i in np.arange(0,max_,step):
		t = np.datetime64('now')
		log.info('Step {0}, with orders ids from {1} to {2} ...'.format((i/1000)+1,i,i+step))
		#-----------------------------------
		# take the sample from i to i+1000 them get the predictions for them.
		sample = test_orders_ids.iloc[i:i+step]
		predictions = predict(sample)
		#---------------------------------------
		# append the predictions to the csv file.
		with open(outputfile,'ab') as f:
			for k,v in predictions.items() :
				products = ' '.join([str(i) for i in v])
				f.write('{0},{1}\n'.format(k,products))
		#----------------------------------------------
		log.msg('finished in %s' % str(np.datetime64('now') - t));
		log.msg('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
	log.close()
	return;

def predict_tp():
	# initialize log file.
	filename = 'log-predict_tp-%s.log' % str(np.datetime64('now'))
	log.init(filename)
	log.msg('starting `predict-test-parallel`, hope it goes well.')
	#-------------------------------------
	# set the ouput csv filename and header.
	outputfile = './data/predictions/predictions-test-%s.csv' % str(np.datetime64('now'))
	with open(outputfile,'ab') as f:
		f.write('order_id,products\n')
	#--------------------------------------------------------------------
	test_orders_ids = DA_orders[ DA_orders.eval_set == 'test' ].order_id;
	max_ = 75000 # number of test samples.
	step = 1000
	for i in np.arange(0,max_,step):
		predictions = dict();
		t = np.datetime64('now')
		log.msg('Step %d, with orders ids from %d to %d ...' % (i/step+1, i, i+step-1 ))
		#-----------------------------------
		# take the sample from i to i+1000 them get the predictions for them.
		change = step / 2
		sample1 = test_orders_ids.iloc[ i            : i + change   ]
		sample2 = test_orders_ids.iloc[ i + change   : i + change*2 ]
		#sample3 = test_orders_ids.iloc[ i + change*2 : i + change*3 ]
		#sample4 = test_orders_ids.iloc[ i + change*3 : i + change*4 ]
		pool = Pool().map(predict, [sample1, sample2])
		pool_output = pool
		for i in pool_output :
			predictions.update(i);
		#---------------------------------------
		# append the predictions to the csv file.
		with open(outputfile,'ab') as f:
			for k,v in predictions.items() :
				products = ' '.join([str(i) for i in v])
				f.write('{0},{1}\n'.format(k,products))
		#----------------------------------------------
		log.msg('finished in %s' % str(np.datetime64('now') - t));
		log.msg('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
	log.close();
	return;
