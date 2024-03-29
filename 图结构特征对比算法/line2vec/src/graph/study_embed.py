# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 13:29:23 2020

@author: NANA
"""
import pandas as pd
import networkx as nx
import numpy as np
import random
import itertools
import copy
import matplotlib
import matplotlib.pyplot as plt
from itertools import chain
import xgboost as xgb
import pandas as pd
import matplotlib as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix


def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain, label=predictors)
        # XGBoost有一个很有用的函数“cv”，这个函数可以在每一次迭代中使用交叉验证，并返回理想的决策树数量。
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)

        alg.set_params(n_estimators=cvresult.shape[0])
    # Fit the algorithm on the data
    alg.fit(dtrain, predictors, eval_metric='logloss')
    # Predict training set:
    dtrain_predictions = alg.predict(dtrain)
    dtrain_predprob = alg.predict_proba(dtrain)[:, 1]

    # Print model report:
    print ("\nModel Report")
    # print ("Accuracy : %.4g" % metrics.accuracy_score(predictors, dtrain_predictions))
    # print ("AUC Score (Train): %f" % metrics.roc_auc_score(predictors, dtrain_predprob))

    # feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    # print ("feat_imp", "*"*30)
    # print ("feature_name feature_importance_score")
    # print (feat_imp)
    return cvresult.shape[0]


# S3_1 为异质 S33_1为同质
all_predictors = []
PA = ['PA']
for i in range(129):
    if i == 0:
        continue
    all_predictors.append(str(i))
# all_predictors.append('M1')


data_test = pd.read_csv('test.csv')
data_train = pd.read_csv('train.csv')

param_test1 = {
'max_depth':[3,5,7,9],
'min_child_weight':[1,3,5],
'gamma':[i/10.0 for i in range(0,10)],
'subsample':[i/10.0 for i in range(6,11)],
'colsample_bytree':[i/10.0 for i in range(6,11)],
'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
real_mi = 1
real_ma = 6
real_g = 0
real_s = 1
real_c = 1
real_r = 0
# for r in param_test1['reg_alpha']:
     # for c in param_test1['colsample_bytree']:
auc_test = []
auc_train = []
pro = []
x_test, y_test = data_test[all_predictors], data_test['label']
x_train, y_train = data_train[all_predictors], data_train['label']
clf = XGBClassifier(
    learning_rate=0.01,  # 默认0.3 原来是0.01
    n_estimators=5000,  # 树的个数
    max_depth=6,  # 默认是6，原来是5
    min_child_weight=1,
    gamma=0,
    subsample=1,  # 默认是1，原来是0.8
    colsample_bytree=1,  # 默认是1，原来是0.8
    # eval_metric= 'logloss',
    objective='binary:logistic',  # 逻辑回归损失函数
    nthread=4,  # cpu线程
    scale_pos_weight=1,
    # reg_alpha = 0,
    seed=0)  # 随机种子默认为0，原来是27
best_n_estimators = modelfit(clf, x_train, y_train)
clf.fit(x_train, y_train)

y_pre = clf.predict(x_test)


y_pro = clf.predict_proba(x_test)[:, 1]
ser = pd.Series(y_pro.tolist())
y_pro_neg = clf.predict_proba(x_test)[:, 0]
print(y_pre)
# pax.to_csv('edge_predic.csv')
y_pro = clf.predict_proba(x_test)[:, 1]
y_pro_neg = clf.predict_proba(x_test)[:, 0]
for i in range(len(y_pro_neg)):
    if y_pro_neg[i] <= y_pro[i]:
        pro.append(1)
    else:
        pro.append(0)
max  = precision_score(y_test, pro)
max_1 = accuracy_score(y_test, y_pre)
print('accuracy:', accuracy_score(y_test, y_pre))
print('precision:', precision_score(y_test, pro))
# scikit-learn工具 中 roc_auc函数只需要提供实际标签和预测值即可
from xgboost import plot_importance


for m in param_test1['max_depth']:
    for mi in param_test1['min_child_weight']:
        pro = []
        x_test, y_test = data_test[all_predictors], data_test['label']
        x_train, y_train = data_train[all_predictors], data_train['label']
        clf = XGBClassifier(
            learning_rate=0.01,  # 默认0.3 原来是0.01
            n_estimators=5000,  # 树的个数
            max_depth=m,  # 默认是6，原来是5
            min_child_weight=mi,
            gamma=0,
            subsample=1,  # 默认是1，原来是0.8
            colsample_bytree=1,  # 默认是1，原来是0.8
            # eval_metric= 'logloss',
            objective='binary:logistic',  # 逻辑回归损失函数
            nthread=4,  # cpu线程
            scale_pos_weight=1,
            # reg_alpha = 0,
            seed=0)  # 随机种子默认为0，原来是27
        best_n_estimators = modelfit(clf, x_train, y_train)
        clf.fit(x_train, y_train)

        y_pre = clf.predict(x_test)

        print(best_n_estimators)

        y_pro = clf.predict_proba(x_test)[:, 1]
        ser = pd.Series(y_pro.tolist())
        y_pro_neg = clf.predict_proba(x_test)[:, 0]
        # pax.to_csv('edge_predic.csv')
        y_pro = clf.predict_proba(x_test)[:, 1]
        y_pro_neg = clf.predict_proba(x_test)[:, 0]
        for i in range(len(y_pro_neg)):
            if y_pro_neg[i] <= y_pro[i]:
                pro.append(1)
            else:
                pro.append(0)
        if accuracy_score(y_test, y_pre) > max_1:
            real_mi = mi
            real_ma = m
            max =precision_score(y_test, pro)
            max_1 = accuracy_score(y_test, y_pre)
        elif accuracy_score(y_test, y_pre) == max_1:
            if precision_score(y_test, pro) > max:
                real_mi = mi
                real_ma = m
                max = precision_score(y_test, pro)
                max_1 = accuracy_score(y_test, y_pre)
        print('accuracy:', accuracy_score(y_test, y_pre))
        print('precision:', precision_score(y_test, pro))

for g in param_test1['gamma']:
        pro =[]
        x_test, y_test = data_test[all_predictors], data_test['label']
        x_train, y_train = data_train[all_predictors], data_train['label']
        clf = XGBClassifier(
            learning_rate=0.01,  # 默认0.3 原来是0.01
            n_estimators=5000,  # 树的个数
            max_depth=real_ma,  # 默认是6，原来是5
            min_child_weight=real_mi,
            gamma=g,
            subsample=1,  # 默认是1，原来是0.8
            colsample_bytree=1,  # 默认是1，原来是0.8
            # eval_metric= 'logloss',
            objective='binary:logistic',  # 逻辑回归损失函数
            nthread=4,  # cpu线程
            scale_pos_weight=1,
            # reg_alpha = 0,
            seed=0)  # 随机种子默认为0，原来是27
        best_n_estimators = modelfit(clf, x_train, y_train)
        clf.fit(x_train, y_train)

        y_pre = clf.predict(x_test)

        print(best_n_estimators)

        y_pro = clf.predict_proba(x_test)[:, 1]
        ser = pd.Series(y_pro.tolist())
        y_pro_neg = clf.predict_proba(x_test)[:, 0]
        print(y_pre)
        # pax.to_csv('edge_predic.csv')
        y_pro = clf.predict_proba(x_test)[:, 1]
        y_pro_neg = clf.predict_proba(x_test)[:, 0]
        for i in range(len(y_pro_neg)):
            if y_pro_neg[i] <= y_pro[i]:
                pro.append(1)
            else:
                pro.append(0)
        if accuracy_score(y_test, y_pre) > max_1:
            real_g = g
            max = precision_score(y_test, pro)
            max_1 = accuracy_score(y_test, y_pre)
        elif accuracy_score(y_test, y_pre) == max_1:
            if precision_score(y_test, pro) > max:
                real_g = g
                max = precision_score(y_test, pro)
                max_1 = accuracy_score(y_test, y_pre)
        print('accuracy:', accuracy_score(y_test, y_pre))
        print('precision:', precision_score(y_test, pro))

for s in param_test1['subsample']:
    for c in param_test1['colsample_bytree']:
        pro = []
        x_test, y_test = data_test[all_predictors], data_test['label']
        x_train, y_train = data_train[all_predictors], data_train['label']
        clf = XGBClassifier(
            learning_rate=0.01,  # 默认0.3 原来是0.01
            n_estimators=5000,  # 树的个数
            max_depth=real_ma,  # 默认是6，原来是5
            min_child_weight=real_mi,
            gamma=real_g,
            subsample=s,  # 默认是1，原来是0.8
            colsample_bytree=c,  # 默认是1，原来是0.8
            # eval_metric= 'logloss',
            objective='binary:logistic',  # 逻辑回归损失函数
            nthread=4,  # cpu线程
            scale_pos_weight=1,
            # reg_alpha = 0,
            seed=0)  # 随机种子默认为0，原来是27
        best_n_estimators = modelfit(clf, x_train, y_train)
        clf.fit(x_train, y_train)

        y_pre = clf.predict(x_test)

        print(best_n_estimators)

        y_pro = clf.predict_proba(x_test)[:, 1]
        ser = pd.Series(y_pro.tolist())
        y_pro_neg = clf.predict_proba(x_test)[:, 0]
        print(y_pre)
        # pax.to_csv('edge_predic.csv')
        y_pro = clf.predict_proba(x_test)[:, 1]
        y_pro_neg = clf.predict_proba(x_test)[:, 0]
        for i in range(len(y_pro_neg)):
            if y_pro_neg[i] <= y_pro[i]:
                pro.append(1)
            else:
                pro.append(0)
        if accuracy_score(y_test, y_pre) > max_1:
            real_c = c
            real_s = s
            max = precision_score(y_test, pro)
            max_1 = accuracy_score(y_test, y_pre)
        elif accuracy_score(y_test, y_pre) == max_1:
            if precision_score(y_test, pro) > max:
                real_c = c
                real_s = s
                max = precision_score(y_test, pro)
                max_1 = accuracy_score(y_test, y_pre)
        print('accuracy:', accuracy_score(y_test, y_pre))
        print('precision:', precision_score(y_test, pro))

for r in param_test1['reg_alpha']:
    pro = []
    x_test, y_test = data_test[all_predictors], data_test['label']
    x_train, y_train = data_train[all_predictors], data_train['label']
    clf = XGBClassifier(
        learning_rate=0.01,  # 默认0.3 原来是0.01
        n_estimators=5000,  # 树的个数
        max_depth=real_ma,  # 默认是6，原来是5
        min_child_weight=real_mi,
        gamma=real_g,
        subsample=real_s,  # 默认是1，原来是0.8
        colsample_bytree=real_c,  # 默认是1，原来是0.8
        # eval_metric= 'logloss',
        objective='binary:logistic',  # 逻辑回归损失函数
        nthread=4,  # cpu线程
        scale_pos_weight=1,
        reg_alpha = r,
        seed=0)  # 随机种子默认为0，原来是27
    best_n_estimators = modelfit(clf, x_train, y_train)
    clf.fit(x_train, y_train)

    y_pre = clf.predict(x_test)

    print(best_n_estimators)

    y_pro = clf.predict_proba(x_test)[:, 1]
    ser = pd.Series(y_pro.tolist())
    y_pro_neg = clf.predict_proba(x_test)[:, 0]
    print(y_pre)
    # pax.to_csv('edge_predic.csv')
    y_pro = clf.predict_proba(x_test)[:, 1]
    y_pro_neg = clf.predict_proba(x_test)[:, 0]
    for i in range(len(y_pro_neg)):
        if y_pro_neg[i] <= y_pro[i]:
            pro.append(1)
        else:
            pro.append(0)
    if accuracy_score(y_test, y_pre) > max:
        real_r = r
        max = precision_score(y_test, pro)
        max_1 = accuracy_score(y_test, y_pre)
    elif accuracy_score(y_test, y_pre) == max_1:
        if precision_score(y_test, pro) > max:
            real_r = r
            max = precision_score(y_test, pro)
            max_1 = accuracy_score(y_test, y_pre)

print('precision:',max)
print('accuracy:',max_1)
# scikit-learn工具 中 roc_auc函数只需要提供实际标签和预测值即可
# scikit-learn工具 中 roc_auc函数只需要提供实际标签和预测值即可

print(clf.feature_importances_)
print(real_ma,real_mi,real_g,real_r,real_s,real_c)


plt.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
plt.show()

pro = []
x_test, y_test = data_test[all_predictors], data_test['label']
x_train, y_train = data_train[all_predictors], data_train['label']
clf = XGBClassifier(
    learning_rate=0.01,  # 默认0.3 原来是0.01
    n_estimators=5000,  # 树的个数
    max_depth=3,  # 默认是6，原来是5
    min_child_weight=1,
    gamma=0,
    subsample=1,  # 默认是1，原来是0.8
    colsample_bytree=0.6,  # 默认是1，原来是0.8
    # eval_metric= 'logloss',
    objective='binary:logistic',  # 逻辑回归损失函数
    nthread=4,  # cpu线程
    scale_pos_weight=1,
    reg_alpha=1e-05,
    seed=0)  # 随机种子默认为0，原来是27
best_n_estimators = modelfit(clf, x_train, y_train)
clf.fit(x_train, y_train)

y_pre = clf.predict(x_test)

print(best_n_estimators)

y_pro = clf.predict_proba(x_test)[:, 1]
ser = pd.Series(y_pro.tolist())
y_pro_neg = clf.predict_proba(x_test)[:, 0]
print(y_pre)
# pax.to_csv('edge_predic.csv')
y_pro = clf.predict_proba(x_test)[:, 1]
y_pro_neg = clf.predict_proba(x_test)[:, 0]
for i in range(len(y_pro_neg)):
    if y_pro_neg[i] <= y_pro[i]:
        pro.append(1)
    else:
        pro.append(0)
print(roc_auc_score(y_test, y_pro))
# if accuracy_score(y_test, y_pre) > max or precision_score(y_test, pro) > max:
#     max = precision_score(y_test, pro)
#     max_1 = accuracy_score(y_test, y_pre)
