#!/usr/bin/env python
'''
This file generate a M x N matrix where :
	- this matrix is meant for one user lets call him x.
	- M : number of product that x ever ordered.
	- N : number of days per week times number of hours per day (24*7=168)
		- Item of M,N : the total number of product M ordered in the give hour N.
		- N is computed by ( for given order X, N = X.order_dow * 24 + X.order_hour_of_day ).

The Algorithm goes as following :
/* START */
user_id = input;
num_of_products = number of products for user_id;
products_freq = num_of_prodcuts x 168 Matrix initialized with Zeroes;
orders_S = subset of orderes made by user_id;
products_S = subset of products_prior for order_S;
For order in orders_S :
	hour = order.order_dow * 24 + order.order_hour_of_day;
	products = subset of products_S for order;
	for product in products:
		products_freq[ product.product_id ][ hour ] += 1;
# Now after we computed the frequency matrix, lets normalize it.

'''

import pandas as pd
import numpy as np

def user_product_freq(orders, products):
	if len(orders) == 0 or len(products) == 0 : return -1;

	#disable pandas warnings.
	pd.options.mode.chained_assignment = None # default = 'warn'

	orders.drop(['user_id','eval_set','order_number','days_since_prior_order'],axis=1,inplace=True);
	products.drop(['add_to_cart_order','reordered'],axis=1,inplace=True);
	products.drop_duplicates(inplace=True);
	# 1 to number of  products.
	# 0 to 167 scores columns by hour, computed by : day * 24 + hour
	#	where "day" in range 0 to 6 , and hour in range 0 to 23.
	num_of_products = len(products.drop_duplicates('product_id'))
	data_ = np.zeros(shape=[num_of_products,168])
	products_freq = pd.DataFrame(data = data_,
		index = products.drop_duplicates('product_id').product_id)

	def set_freq(product, computed_hour):
		# takes a product series and computed_hour to update the matrix.
		products_freq[int(computed_hour)][product.product_id] += 1;
	def get_products(order):
		# takes order series to get the products then apply set_freq() for each product.
		products_sub = products[ products.order_id == order.order_id ];
		products_sub.apply(set_freq,
			computed_hour = (order.order_dow * 24) + order.order_hour_of_day, axis=1);

	# apply get_products() on each order.
	orders.apply(get_products, axis=1);
	norm_products_freq = products_freq.apply(lambda d: d - np.median(d) / d.max(),axis=1)
	return norm_products_freq;
