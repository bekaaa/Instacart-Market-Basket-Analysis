#! /usr/bin/env python
import pandas as pd
import numpy as np
import pickle
import random
import logging
from per_user_product_freq_v2 import user_product_freq
'''
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
pklfile = './data/pickles/product_freq_df.pkl'
with open(pklfile, 'rb') as f :
	products_freq = pickle.load(f);
#normalize it.
product_freq = products_freq.apply(lambda d: (d - np.median(d)) / d.max(), axis=1)
DA_orders    = pd.read_csv('./data/orders.csv')
DA_products  = pd.read_csv('./data/order_products__prior.csv')

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
		computed_hour = (order.order_dow * 24) + order.order_hour_of_day;
		maxN = user_freq[computed_hour].nlargest(X,computed_hour).index ;
		# Update the predictions dictionary.
		predictions[int(order.order_id)] = list(maxN) ;
		#----------------------------------------------------
	return predictions

def spredict(size = 0,rand = False, eval_set='test'):
	if size <= 0 or eval_set not in ('test', 'train') : return dict();
	max_ = 75000 if eval_set == test else 1384617
	if size > max_ : return dict();
	#--------------------------------------------
	# set orders ids based on the input parameters.
	test_orders_ids = DA_orders[ DA_orders.eval_set == eval_set ].order_id;
	if rand :
		sample = random.sample(range(max_),size)
		test_orders_ids = test_orders_ids.iloc[sample]
	else :
		test_orders_ids = test_orders_ids.iloc[0:size]
	#-----------------------------------------
	#get predictions.
	predictions = predict(test_orders_ids)
	#-------------------------------------------------
	# write predictions to a csv file.
	outputfile = './data/output/predictions-%s-%d-%s.csv' % ( eval_set, size, rand )
	with open(outputfile,'wb') as f:
		f.write('order_id,products\n')
		for k,v in predictions.items() :
			products = ' '.join([str(i) for i in v])
			f.write('{0},{1}\n'.format(k,products))
	return predictions

def tpredict():
	# initialize log file.
	log = logging.getLogger();
	log.setLevel(logging.INFO)
	filename = './data/output/log/log-%s.log' % str(np.datetime64('now'))
	handler = logging.FileHandler(filename)
	handler.setLevel(logging.INFO)
	formatter = logging.Formatter(
		fmt='%(asctime)s %(levelname)s: %(message)s',
		datefmt='%m-%d %H:%M'
		)
	handler.setFormatter(formatter)
	log.addHandler(handler)
	#-------------------------------------
	# set the ouput csv filename and header.
	outputfile = './data/output/predictions-test-{}.csv'.format(str(np.datetime64('now')))
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
		log.info('finished in %s' % str(np.datetime64('now') - t);
		log.info('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
	log.removeHandler(handler)
	return;
