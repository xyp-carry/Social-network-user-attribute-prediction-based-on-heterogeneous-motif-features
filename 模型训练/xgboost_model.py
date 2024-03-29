import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, auc, roc_curve, \
    roc_auc_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier
from sklearn import preprocessing


def learning_curve(train_loss=None, test_loss=None):
    return
    plt.figure(figsize=(10, 10))
    plt.title('loss')
    if train_loss:
        x_scale = [i for i in range(len(train_loss))]
        plt.plot(x_scale, train_loss, label='train', color='r')
    if test_loss:
        x_scale = [i for i in range(len(test_loss))]
        plt.plot(x_scale, test_loss, label='test', color='b')
    plt.legend()
    plt.show()


def Normal(data, columns):
    zscore = preprocessing.StandardScaler()
    z_score = zscore.fit_transform(data)
    z_score = pd.DataFrame(z_score, columns=columns)
    return z_score


rounds = 50
pos_weight = 1
pca_trigger = False
train_trigger = True
merge_features = False
# S3_1 为异质 S33_1为同质
all_predictors = []
PA = ['PA']
# motif_structure = ['M4_1', 'M4_2', 'M4_3', 'M4_4', 'M4_5', 'M4_6', 'M4_7', 'M4_8', 'M4_9', 'M4_10', 'M4_11']
# motif_label = ['111', '112', '121', '122', '211', '212', '221', '222']
# motif_structure3 = ['M3_1', 'M3_2', 'M3_3']
# motif_label3 = ['11', '12', '21', '22']
# xc = []
# for i in motif_structure:
#     all_predictors.append(i)
# for i in motif_structure:
#     for j in motif_label:
#         all_predictors.append(i + j)
# for i in motif_structure3:
#     for j in motif_label3:
#         all_predictors.append(i + j)
# for i in motif_structure3:
#     all_predictors.append(i)




def xunlian(x_train, y_train, x_test, y_test):
    real_mi = 1
    real_ma = 6
    real_g = 0
    real_s = 1
    real_c = 1
    real_r = 0
    pro = []
    param_test1 = {
        'max_depth': [3, 5, 7, 9],
        'min_child_weight': [1, 3, 5],
        'gamma': [i / 10.0 for i in range(0, 10)],
        'subsample': [i / 10.0 for i in range(6, 11)],
        'colsample_bytree': [i / 10.0 for i in range(6, 11)],
        'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
    }

    clf = XGBClassifier(
        learning_rate=0.01,  # 默认0.3 原来是0.01
        n_estimators=35,  # 树的个数
        max_depth=9,  # 默认是6，原来是5
        min_child_weight=1,
        gamma=0,
        subsample=1,  # 默认是1，原来是0.8
        colsample_bytree=1,  # 默认是1，原来是0.8
        # eval_metric= 'logloss',
        objective='multi:softprob',  # 逻辑回归损失函数
        num_class=2,
        nthread=4,  # cpu线程
        scale_pos_weight=pos_weight,
        reg_alpha=0,
        seed=0)  # 随机种子默认为0，原来是27

    clf.fit(x_train, y_train, eval_set=[(x_test, y_test), (x_train, y_train)])
    # learning_curve(test_loss=clf.evals_result()['validation_0']['logloss'])
    learning_curve(clf.evals_result()['validation_1']['mlogloss'], clf.evals_result()['validation_0']['mlogloss'])

    y_pre = clf.predict(x_test)
    print(y_pre.shape)
    yyy = []
    print(y_pre)
    for i in y_pre:
        if i[0] == 1:
            yyy.append(0)
        else:
            yyy.append(1)

    y_pre = yyy
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

    max = roc_auc_score(y_test, y_pro)
    max_1 = accuracy_score(y_test, y_pre)
    print('accuracy:', accuracy_score(y_test, y_pre))
    print('auc:', roc_auc_score(y_test, y_pro))
    print('recall', recall_score(y_test, y_pre))
    # scikit-learn工具 中 roc_auc函数只需要提供实际标签和预测值即可
    from xgboost import plot_importance

    for m in param_test1['max_depth']:
        for mi in param_test1['min_child_weight']:
            pro = []
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
                scale_pos_weight=pos_weight,
                reg_alpha=0,
                seed=0)  # 随机种子默认为0，原来是27

            clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=rounds)
            learning_curve(test_loss=clf.evals_result()['validation_0']['logloss'])

            y_pre = clf.predict(x_test)

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
                max = roc_auc_score(y_test, y_pro)
                max_1 = accuracy_score(y_test, y_pre)
            elif accuracy_score(y_test, y_pre) == max_1:
                if roc_auc_score(y_test, y_pro) > max:
                    real_mi = mi
                    real_ma = m
                    max = roc_auc_score(y_test, y_pro)
                    max_1 = accuracy_score(y_test, y_pre)
            print('accuracy:', accuracy_score(y_test, y_pre))
            print('precision:', roc_auc_score(y_test, y_pro))

    for g in param_test1['gamma']:
        pro = []
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
            scale_pos_weight=pos_weight,
            reg_alpha=0,
            seed=0)  # 随机种子默认为0，原来是27

        clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=rounds)
        learning_curve(test_loss=clf.evals_result()['validation_0']['logloss'])

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
        if accuracy_score(y_test, y_pre) > max_1:
            real_g = g
            max = roc_auc_score(y_test, y_pro)
            max_1 = accuracy_score(y_test, y_pre)
        elif accuracy_score(y_test, y_pre) == max_1:
            if roc_auc_score(y_test, y_pro) > max:
                real_g = g
                max = roc_auc_score(y_test, y_pro)
                max_1 = accuracy_score(y_test, y_pre)
        print('accuracy:', accuracy_score(y_test, y_pre))
        print('precision:', roc_auc_score(y_test, y_pro))

    for s in param_test1['subsample']:
        for c in param_test1['colsample_bytree']:
            pro = []
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
                scale_pos_weight=pos_weight,
                # reg_alpha = 0,
                seed=0)  # 随机种子默认为0，原来是27

            clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=rounds)
            learning_curve(test_loss=clf.evals_result()['validation_0']['logloss'])

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
            if accuracy_score(y_test, y_pre) > max_1:
                real_c = c
                real_s = s
                max = roc_auc_score(y_test, y_pro)
                max_1 = accuracy_score(y_test, y_pre)
            elif accuracy_score(y_test, y_pre) == max_1:
                if roc_auc_score(y_test, y_pro) > max:
                    real_c = c
                    real_s = s
                    max = roc_auc_score(y_test, y_pro)
                    max_1 = accuracy_score(y_test, y_pre)
            print('accuracy:', accuracy_score(y_test, y_pre))
            print('precision:', roc_auc_score(y_test, y_pro))

    for r in param_test1['reg_alpha']:
        pro = []
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
            scale_pos_weight=pos_weight,
            reg_alpha=r,
            seed=0)  # 随机种子默认为0，原来是27

        clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=rounds)
        learning_curve(test_loss=clf.evals_result()['validation_0']['logloss'])

        y_pre = clf.predict(x_test)

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
        if accuracy_score(y_test, y_pre) > max:
            real_r = r
            max = roc_auc_score(y_test, y_pro)
            max_1 = accuracy_score(y_test, y_pre)
        elif accuracy_score(y_test, y_pre) == max_1:
            if roc_auc_score(y_test, y_pro) > max:
                real_r = r
                max = roc_auc_score(y_test, y_pro)
                max_1 = accuracy_score(y_test, y_pre)

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
            scale_pos_weight=pos_weight,
            reg_alpha=real_r,
            seed=0)  # 随机种子默认为0，原来是27
        clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=rounds)
        learning_curve(test_loss=clf.evals_result()['validation_0']['logloss'])
        print(real_ma, real_mi, real_g, real_s, real_c, real_r)

    return clf


data_test = pd.read_csv('test_motif.csv', index_col='src')
data_val = pd.read_csv('test_motif.csv', index_col='src')
data_train = pd.read_csv('train_motif.csv', index_col='src')


all_predictors = list(data_train.columns)
all_predictors.remove('label')

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

x_test, y_test = data_test[all_predictors], data_test['label']
x_train, y_train = data_train[all_predictors], data_train['label']
x_val, y_val = data_val[all_predictors], data_val['label']

x_val = x_test
y_val = y_test

print(x_train)


if pca_trigger:
    x_test = pca.fit_transform(x_test)
    x_train = pca.fit_transform(x_train)
    x_val = pca.fit_transform(x_val)

if train_trigger:
    clf = xunlian(x_train, y_train, x_val, y_val)
else:
    clf = XGBClassifier(
        learning_rate=0.01,  # 默认0.3 原来是0.01
        n_estimators=5000,  # 树的个数
        max_depth=3,  # 默认是6，原来是5
        min_child_weight=3,
        gamma=0,
        subsample=0.6,  # 默认是1，原来是0.8
        colsample_bytree=1.0,  # 默认是1，原来是0.8
        # eval_metric= 'logloss',
        objective='binary:logistic',  # 逻辑回归损失函数
        nthread=4,  # cpu线程
        scale_pos_weight=1,
        reg_alpha=0,
        seed=0)  # 随机种子默认为0，原来是27
clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=rounds)
learning_curve(test_loss=clf.evals_result()['validation_0']['logloss'])

y_pre = clf.predict(x_test)
pro = []
y_pro = clf.predict_proba(x_test)[:, 1]
ser = pd.Series(y_pro.tolist())
y_pro_neg = clf.predict_proba(x_test)[:, 0]
y_pro = clf.predict_proba(x_test)[:, 1]
y_pro_neg = clf.predict_proba(x_test)[:, 0]
for i in range(len(y_pro_neg)):
    if y_pro_neg[i] <= y_pro[i]:
        pro.append(1)
    else:
        pro.append(0)

print(accuracy_score(y_test, y_pre))
print(roc_auc_score(y_test, pro))

print("ACC: {}".format(accuracy_score(y_test, y_pre)))
print("ROC: {}".format(roc_auc_score(y_test, y_pre)))
print("F1: {}".format(f1_score(y_test, y_pre)))
print("Precision: {}".format(precision_score(y_test, y_pre)))
print("Recall: {}\n".format(recall_score(y_test, y_pre)))

# print(clf.feature_importances_)
print(y_pro)

