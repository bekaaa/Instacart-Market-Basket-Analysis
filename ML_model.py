#!/usr/bin/env python

import pickle
import log
#from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier



# Machine learning Model
#--------------------------------------------------------------------
# - split data to train and test
seed       = 40
#train_size = 1000.0 / 131209
test_size  = .1 #1 - train_size
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = GradientBoostingClassifier()
print "fitting model"
model.fit(Xtrain, Ytrain)
print "predicting test data"
predictions = model.predict(Xtest)
print "accuracy : %f,\n %s" %\
	( accuracy_score(Ytest,predictions), classification_report(Ytest,predictions) )
#-------------------------------------------------------------------
# calculate score

def calc_score(order):
	y_true = DA_train[DA_train.order_id == order.order_id].product_id.tolist()
