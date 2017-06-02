#! /usr/bin/env python
import pandas as pd
import numpy as np
import pickle
from per_user_product_freq_v2 import user_product_freq

def predict(orders_list):
	predicted = dict()
	pklfile = './data/pickle/product_freq_df.pkl'
	with open(pklfile, 'rb') as f :
		products_freq = pickle.load(f);
	#normalize it.
	product_freq = products_freq.apply(lambda d: d - np.median(d) / d.max(), axis=1)
	DA_orders    = pd.read_csv('./data/orders.csv')
	DA_orders    = DA_orders[ DA_orders.order_id.isin(orders_list) ]
	DA_products  = pd.read_csv('./data/order_products__prior.csv')
	DA_products  = DA_products[ DA_products.order_id.isin(order_list) ]

	for order_id in orders_list :
		order = DA_orders[DA_orders.order_id == order_id];
		# compute user's products freq.
		user_id = order.user_id;
		user_orders = DA_orders[ DA_orders.user_id == user_id ];
		user_products = DA_products[ DA_products.order_id.isin(user_orders.order_id) ];
		user_freq = user_product_freq(user_orders, user_products)
		user_freq = user_freq.apply(lambda d: d + .5 * products_freq.loc[prod], axis=1)
		#--------------------------------------------------
		# compute X where x is the predicted number of products in that order,
		# I'll assume that X is the median of number of products for that user.
		sum_of_products = [];
		def AP_get_num_prod(order):
			sum_of_products.append( len(user_products[user_products.order_id == order.order_id]) )
		user_orders.apply(AP_get_num_prod, axis=1)
		X = np.median(sum_of_products)
		#-----------------------------------------------------
		# Now get the X highest products.
		computed_hour = (order.order_dow * 24) + order.order_hour_of_day;
		maxN = user_orders[computed_hour].nlargest(X);
		# Update the predicted dictionary.
		predicted[order.order_id] = list(maxN.index)

	return predicted
