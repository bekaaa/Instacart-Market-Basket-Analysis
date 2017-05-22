#!/usr/bin/env python

import pandas as pd
import numpy as np
import pickle

output_file = "./data/products_freq.pkl"
data_files = [
	"aisles",
	"departments",
	"order_products__prior",
	"orders",
	"products",
	"sample_submission"
	]
for file_name in data_files :
	exec( "DA_{0} = pd.read_csv('./data/{0}.csv')".format(file_name) )

# keep only columns "product-id, order-id, dow, hod"
DA_RESIZED_order_products__prior =\
	DA_order_products__prior.drop(['add_to_cart_order','reordered'],axis=1)
DA_RESIZED_orders = \
	DA_orders.drop(['user_id','eval_set','order_number','days_since_prior_order'],axis=1)

# take samples for test
DA_SAMPLE_orders = DA_RESIZED_orders.loc[1:1000]
DA_SAMPLE_order_products_prior = DA_RESIZED_order_products__prior[1:10000]

# row 0 should be empty.
# 1 to 49688 product rows
# 0 to 167 scores columns by hour, computed by : day * 24 + hour
#	where "day" in range 0 to 6 , and hour in range 0 to 23.
products_freq = np.zeros(shape=[49689,168])

# will take 37 hours
t = np.datetime64('now')

def set_freq(product, col_id):
	products_freq[product.product_id][int(col_id)] += 1;

def get_products(order):
	products = DA_RESIZED_order_products__prior[ \
		DA_RESIZED_order_products__prior.order_id == order.order_id]
	products.apply(set_freq, col_id = (order.order_dow * 24) + order.order_hour_of_day, axis=1)

DA_SAMPLE_orders.apply(get_products, axis=1)

t = np.datetime64('now') - t
print t

with open(output_file, 'wb') as f:
	pickle.dump(products_freq, f);
