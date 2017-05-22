#!/usr/bin/env python
'''
This file generates a  M x N matrix where;
	- M : number of products (49689 for products from 1 to 49688; and row 0 is empty.)
	- N : number of days per week times number of hours per day (24*7=168)
	- Item of M,N : the total number of product M ordered in the give hour N.
	- N is computed by ( for given order X, N = X.order_dow * 24 + X.order_hour_of_day ).

After Many tries for optimizing the complexity/time of the algorithm, it will work as follow :
/* START */
products_freq = 49689x168 Matrix initialized with 0 values.
For i from 1 to 206,209(number of user_ids) by 1000 (step):
	orders_S = subset of orders for users in the given range (i).
	products_S = subset of products_prior for orders_S.
	For order in orders_S :
		hour = order.order_dow * 24 + order.order_hour_of_day
		products = subset of products_S for order.
		for product in products:
			products_freq[ product.product_id ][ hour ] += 1

Save the Matrix to a pickle file so we won't run this script each time.
Print some logs.
/* END */
'''
import pandas as pd
import numpy as np
import pickle

#disable pandas warnings.
pd.options.mode.chained_assignment = None  # default='warn'

DA_orders = pd.read_csv('./data/orders.csv')
DA_order_products__prior = pd.read_csv('./data/order_products__prior.csv')

# define the MATRIX
# row 0 should be empty.
# 1 to 49688 product rows
# 0 to 167 scores columns by hour, computed by : day * 24 + hour
#	where "day" in range 0 to 6 , and hour in range 0 to 23.
products_freq = np.zeros(shape=[49689,168])
############################################################################
# reasonable step value is between 1,000 and 10,000
# it indicates the number of users.
# the higher step values the less loops but slower perfomance.
step = 1000
# tt > total_time, for log messages purpose.
tt = np.datetime64('now');
# for user_ids from 1 to 206,209
for i in range(0,207000,step):
	#time record for log messsages.
	t = np.datetime64('now');

	orders_S = DA_orders[DA_orders.user_id.isin(range(i,i+step))]
	products_S =\
		DA_order_products__prior[ DA_order_products__prior.order_id.isin(orders_S.order_id) ]
	# drop some columns to minimize the data size.
	orders_S.drop(['user_id','eval_set','order_number','days_since_prior_order'],axis=1,inplace=1)
	products_S.drop(['add_to_cart_order','reordered'],axis=1,inplace=1)
    #***************************************************************

	def set_freq(product, computed_hour):
		# takes a product series and computed_hour to update the matrix.
		products_freq[product.product_id][int(computed_hour)] += 1;

	def get_products(order):
		# takes order series to get the products then apply set_freq() for each product.
		products = products_S[ products_S.order_id == order.order_id ];
		products.apply(set_freq,
			computed_hour = (order.order_dow * 24) + order.order_hour_of_day, axis=1);

	# apply get_products() on each order.
	orders_S.apply(get_products, axis=1)
	#*************************************************************
	t = np.datetime64('now') - t
	#convert time to seconds from timedelta64 format.
	t = int(str(t).split(' ')[0])
	# print log message :
	print  "\n* completed users from {0} to {1}, with #orders = {2},#products = {3},\
 in {4} seconds or {5} minutes.".format(i, i+step-1, len(orders_S), len(products_S), t, t/60.0)
	#*************************************************************
tt = np.datetime64('now') - tt;
tt = int(str(tt).split(' ')[0])
print "\n\n* total time = {0} seconds or {1} minutes.".format(tt, tt/60.0)

# Save the matrix to a file, actually two files, one of them is for backup.
files = ["./data/products_freq.pkl","./data/products_freq.pkl.backup"]
for file_ in files :
	with open(file_, 'wb') as f :
		pickle.dump(products_freq, f)
###########################################################################
