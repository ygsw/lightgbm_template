'''
評価をkfoldの平均とする
early_stopping

備考
trainを分割してvalidを作っているが、学習データを少しでも増やしたいならtestを分割でも良い
'''
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold

import lightgbm as lgb

from tqdm import tqdm_notebook as tqdm

# データ
X = pd.DataFrame()

# 正解ラベル
y = np.array()

# 分割数
k = 3
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)

lgbm_params = {}

auc_list = []
precision_list = []
recall_list = []

# kfoldで学習
for train_index, test_index in skf.split(X, y):

    X_train = X.iloc[train_index]
    y_train = y[train_index]
    X_test = X.iloc[test_index]
    y_test = y[test_index]

    X_train_, X_valid, y_train_, y_valid = train_test_split(X_train, y_train, random_state=90, shuffle=True, stratify=y_train, test_size=0.2)

    # データセットを生成する
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    # 上記のパラメータでモデルを学習する
    model = lgb.train(lgbm_params, lgb_train,
                        valid_sets=lgb_eval,
                        num_boost_round=100000,
                        early_stopping_rounds=10)


    predict_proba = model.predict(X_test, num_iteration=model.best_iteration)
    predict = [0 if i < 0.5 else 1 for i in predict_proba]

    fpr, tpr, thr_arr = metrics.roc_curve(y_test, predict)
    auc = metrics.auc(fpr, tpr)
    precision = metrics.precision_score(y_test, predict)
    recall = metrics.recall_score(y_test, predict)
    
    print(auc, precision, recall)
    
    auc_list.append(auc)
    precision_list.append(precision)
    recall_list.append(recall)


print(np.mean(auc_list), np.mean(precision_list), np.mean(recall_list))
