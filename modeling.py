#!/home/pinoystat/Documents/python/environment/datascience/bin/python
# coding: utf-8

# For this script only:
import sys

# Modeling Script
# Import the preprocessing classes:


from preprocessing import *

# Importing libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

import re
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler, SplineTransformer
from sklearn.preprocessing import PowerTransformer 
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin 

#metrics
from sklearn.metrics import mean_squared_error, r2_score
#model selection
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

#load preprocessed dataset:
import joblib
#models
from sklearn.linear_model import ElasticNet, SGDRegressor
from sklearn.ensemble import StackingRegressor 
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor 
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

# check using keras:
import keras as ks
import tensorflow as tf

# X = joblib.load("transformed_train.pkl")
# print("X shape: {}".format(X.shape))


input_folder = "kaggle/input/linking-writing-processes-to-writing-quality/"
train_scores = pd.read_csv(input_folder +"train_scores.csv", delimiter = ",", header = 0)
train_logs = pd.read_csv(input_folder + "train_logs.csv",delimiter = ",",header = 0)
#test logs:
test_logs = pd.read_csv(input_folder +"test_logs.csv", delimiter = ",", header = 0)
scores = pd.Series(data = train_scores['score'].values, index = train_scores['id'].values, name = 'score')


# Pipeline to combine summary:
cp_pipe = Pipeline([('cp_tx', CursorPositionTransformer())])
wc_pipe = Pipeline([('wc_tx', WordCountTransformer())])
tc_pipe = Pipeline([('tc_tx', TextChangeTransformer())])
tc2_pipe = Pipeline([('tc2_tx', TextChangeTransformer2(14))])
tc3_pipe = Pipeline([('tc3_tx', TextChangeTransformer3(scores= scores, max_wl = 15))])
tc_act = make_pipeline(ActivityTransformer(scores))
tc_act_combo = make_pipeline(ComboActivityActionTime())
tc_event_combo = make_pipeline(EventComboTransformer())
event_cv_tx = make_pipeline(EventCountVectorizerTransformer())
event_cv_tx2 = make_pipeline(EventCountVectorizerTransformer2())
tc4_pipe = Pipeline([('tc4_tx', TextChangeVectorizer())])

# Creating the model:

# model = BaggingRegressor(estimator = xgb.XGBRegressor(tree_method = "hist", device = 'cpu', learning_rate = 0.1,
#                                                      subsample = 1, max_depth = 3, random_state = 11, n_jobs = -1,
#                                                       n_estimators = 1000),
#                          random_state = 11, n_jobs = -1,max_samples = .3, n_estimators = 100, bootstrap = True)
#
# model_parameters = {'device':['cpu'], 'tree_method':['hist'], 'learning_rate':[0.001, 0.01, 0.1],
#                     'subsample' : [1], 'max_depth' : [2,4,6,8], 'n_estimators':[100,200,300,400,500,600]}
# 
# model = xgb.XGBRegressor()


# combine now:
initial_pipe = FeatureUnion(
        transformer_list = [
            ('cp_pipe', cp_pipe),
            ("wc_pipe", wc_pipe), 
            ('tc_pipe', tc_pipe),
            ('tc2_pipe', tc2_pipe), 
            ('tc3_pipe', tc3_pipe),
            ('tc_act', tc_act),
            ('tc_act_combo', tc_act_combo), 
            ('tc_event_combo', tc_event_combo),
            ('event_cv_tx2', event_cv_tx2), 
            ('event_cv_tx', event_cv_tx)
            # ('tc4_pipe', tc4_pipe)
            ])



final_pipe = make_pipeline(initial_pipe, StandardScaler(with_mean = False))

# To transform the labels, Y:
power_tx = PowerTransformer()


# display scores
def display_scores(actual,predicted):
    actual = power_tx.inverse_transform(actual.reshape(-1,1)).ravel()
    predicted = power_tx.inverse_transform(predicted.reshape(-1,1)).ravel()
    scores = mean_squared_error(actual, predicted) 
    print("RMSE: {}".format(np.sqrt(scores)))
    r2 = r2_score(actual, predicted)
    print("R2: {}".format(r2))

# transform the predicted values into its nearest score
def give_nearest_score(x_predicted):
    score_list = np.arange(start = 0.5, stop = 6.5, step = 0.5)
    diff = np.abs(score_list - x_predicted)
    index_min = np.where(diff == np.min(diff))[0][0]
    return score_list[index_min]

# def display_scores_with_offset(actual,predicted):
#     actual = power_tx.inverse_transform(actual.reshape(-1,1)).ravel()
#     predicted = power_tx.inverse_transform(predicted.reshape(-1,1)).ravel()
# 
#     offset = np.array(list(map(give_nearest_score, predicted))) 
#     scores = mean_squared_error(actual, offset) 
#     print("RMSE (with offset) : {}".format(np.sqrt(scores)))
#     r2 = r2_score(actual, offset)
#     print("R2:(with offset) {}".format(r2))

def cross_validation(model, X,Y, cv = 5):
    cv_scores = cross_val_score(estimator=model, X=X, y=Y, scoring = "neg_mean_squared_error", 
                                verbose = 3, cv = cv, n_jobs = -1)

    print("cross-validated scores:")
    print(cv_scores)
    rmse = np.sqrt(-cv_scores)
    print("RMSE: {}".format(rmse))
    print("Mean RMSE: {}".format(np.mean(rmse)))


# Execution section:



# 1 for fitting again
print("sys argv 1 == {}".format(sys.argv[1]))
if int(sys.argv[1]) == 1:
    print("Preprocessing starts.")
    print("transforming train data.")
    X = final_pipe.fit_transform(train_logs)
    print("transforming test data.")
    xtest = final_pipe.transform(test_logs)
    joblib.dump(X, "transformed_train.pkl")
    joblib.dump(xtest, "transformed_test.pkl")

else:
    if int(sys.argv[1]) == 0:
        print("Loading pre-processed dataset")
        X = joblib.load('transformed_train.pkl')
        xtest = joblib.load('transformed_test.pkl')

    else:
        if int(sys.argv[1]) == 2:
            print("Preprocessing starts.")
            print("transforming train data.")
            X = final_pipe.fit_transform(train_logs)
            sys.exit()





# print("Grid searching")
# grid_search = GridSearchCV(model, model_parameters, cv = 5, scoring = 'neg_mean_squared_error', n_jobs = -1, verbose = 3)
# grid_search.fit(X,Y)
# joblib.dump(grid_search,"xgb_grid.pkl")
# print("Grid results saved")


# model = BaggingRegressor(estimator = xgb.XGBRegressor(tree_method = "hist", device = 'cpu', learning_rate = 0.1,
#                                                      subsample = 1, max_depth = 3, random_state = 11, n_jobs = -1,
#                                                       n_estimators = 1000),
#                          random_state = 11, n_jobs = -1,max_samples = .3, n_estimators = 100, bootstrap = True)
# 



base_model = xgb.XGBRegressor(tree_method = "hist", device = 'cpu', learning_rate = 0.01,
                 subsample = 1, max_depth = 3, random_state = 11, n_jobs = -1,
                         n_estimators = 300)


model = BaggingRegressor(estimator = base_model, 
                         random_state = 11, n_jobs = -1,max_samples = .8, n_estimators = 100, bootstrap = True)

# create a tensforflow model:
X = X.toarray()
# X = tf.convert_to_tensor(X)

print(type(X))
print("shape of X: {}".format(X.shape))
model = ks.Sequential()
model.add(ks.Input(shape =[X.shape[1],], name = "pasukan"))
model.add(ks.layers.Dense(24, activation = "relu", name = "Una"))
model.add(ks.layers.Dense(24, activation = "relu", name = "Pangalawa"))
model.add(ks.layers.Dense(10, activation = "relu", name = "Ikatlo"))
model.add(ks.layers.Dense(1, name = "Finale"))


print("Deployed model:")
print(model.summary())


# print("The best model: \n {}".format(model))
Y = scores.values
Y = power_tx.fit_transform(Y.reshape(-1,1)).ravel()
print("Y shape : {}".format(Y.shape))
print("X shape : {}".format(X.shape))

# print("Executing cross_validation of the best estimator model:")
# cross_validation(model, X,Y)
print("Fitting the model to the full dataset before submission.")

model.compile(
        optimizer = ks.optimizers.Adam(learning_rate = 0.01),
        loss = "mean_squared_error")
model.fit(X, Y, epochs = 150, verbose = 0,validation_split = 0.3)

test_results = model.evaluate(X,Y, verbose = 0)
print("test results")
print(test_results)

# score = model.evaluate(X, Y)
# print("test score:", score)

# model.fit(X,Y, batch_size = 128, epochs = 2, verbose = 1)
# evaluate the mode:
 
# # model performance:
print("Checking model performance on whole dataset.")
predictions = model.predict(X)
preds = power_tx.inverse_transform(predictions[:20,].reshape(-1,1)).ravel()
print(preds)
print(predictions.shape)
# 
display_scores(Y, predictions)
print("y labels")
print(np.array(scores)[:20])
# display_scores_with_offset(Y, predictions)

# test_ids = test_logs.id.unique()
# 
# pred_test = model.predict(xtest.dense())
# submission = pd.DataFrame({'id': test_ids,'score': pred_test})
# submission.to_csv("submission.csv", index = False)


