#! /usr/bin/env python
import pandas as pd
import numpy as np
import pickle
import os.path

pred_dir = './data/predictions/'
training_data_file = './data/data/order_products__train.csv'
def evaluate(filename):
	filename = pred_dir + filename
	if not os.path.isfile(filename) : return -1,-1,-1
	products_true = pd.read_csv(training_data_file)
	products_pred = pd.read_csv(filename)
	precision, recall, F1 = [], [], [];
	def update_score(product_pred):
		order_id = product_pred.order_id
		products = product_pred.products
		#y_predicted = products.get_values()[0]
		y_predicted = products.split(' ')
		y_predicted = [ int(i) for i in y_predicted ]
		y_expected = products_true[ products_true.order_id == order_id ].product_id.tolist()
		true_positive = sum([ 1.0 for i in y_predicted if i in y_expected  ])
		prec = true_positive / len(y_predicted)
		rec  = true_positive / len(y_expected)
		try :
			f1   = 2.0 * prec * rec / ( prec + rec )
		except ZeroDivisionError :
			f1 = 0.0
		precision.append(prec)
		recall.append(rec)
		F1.append(f1)
		return
	products_pred.apply(update_score,axis=1)
	precision = np.mean(precision)
	recall    = np.mean(recall)
	F1        = np.mean(F1)
	print 'Precision : %0.4f\tRecall : %0.4f\tF1-score : %0.4f\n' % (precision, recall, F1)
	return precision, recall, F1
