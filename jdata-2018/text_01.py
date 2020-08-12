# -*- coding: utf-8 -*-
"""
Created on Tue May 29 09:19:57 2018

@author: MSIK
"""

import pandas as pd
import numpy as np
from datetime import date

import warnings

warnings.filterwarnings('ignore')

import lightgbm
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_predict

"""
处理sms数据
"""
smsTrain = pd.read_table('sms_train.txt', header=None)
smsTrain.columns = ['uid', 'opp_num', 'opp_head', 'opp_len', 'start_time', 'in_out']
smsText = pd.read_table('sms_test_a.txt', header=None)
smsText.columns = ['uid', 'opp_num', 'opp_head', 'opp_len', 'start_time', 'in_out']

smsDx = pd.concat([smsTrain, smsText])

"""
处理voice数据
"""
voiceTrain = pd.read_table('voice_train.txt', header=None)
voiceTrain.columns = ['uid', 'opp_num', 'opp_head', 'opp_len', 'start_time', 'end_time', 'call_type', 'in_out']
voiceText = pd.read_table('voice_test_a.txt', header=None)
voiceText.columns = ['uid', 'opp_num', 'opp_head', 'opp_len', 'start_time', 'end_time', 'call_type', 'in_out']

voiceVl = pd.concat([voiceTrain, voiceText])
"""
wa_train.txt 用户网站访问记录数据
"""
waTrain = pd.read_table('wa_train.txt', header=None)
# pandas中还有读取表格的通用函数read_table。
# read_table 函数:
# 功能: 从文件、url、文件型对象中加载带分隔符的数据，默认为'\t'。（read_csv默认分隔符是逗号）
# 可以通过制定sep 参数来修改默认分隔符。
waTrain.columns = ['uid', 'wa_name', 'visit_cnt', 'visit_dura', 'up_flow',
                   'down_flow', 'wa_type', 'date']
waTest = pd.read_table('wa_test_a.txt', header=None)
waTest.columns = ['uid', 'wa_name', 'visit_cnt', 'visit_dura', 'up_flow',
                  'down_flow', 'wa_type', 'date']  # 定义列的名称

# 假设第一天是周5
weekList = [5, 6, 7, 1, 2, 3, 4] * 7
weekList = weekList[:45]
# print(waTrain, '我是训练集')
# print(waTest, '我是测试集 ')
waData = pd.concat([waTrain, waTest])  # 合并两组txt数据
# print(waData, '我是合集')
# waData['weekday'] = waData['date'].astype('int').map(lambda x: weekList[x-1] if x >0 else 1)

# 标签uid_train.txt  0:4099 1:900
uidTrain = pd.read_table('uid_train.txt', header=None)
uidTrain.columns = ['uid', 'label']  # 定义列的名称

uidTest = pd.DataFrame()
uidTest['uid'] = range(5000, 7000)
uidTest.uid = uidTest.uid.apply(lambda x: 'u' + str(x).zfill(4))  # Python zfill() 方法返回指定长度的字符串，原字符串右对齐，前面填充0
# apply 是 pandas 库的一个很重要的函数，多和 groupby 函数一起用，也可以直接用于 DataFrame 和 Series 对象。
# 主要用于数据聚合运算，可以很方便的对分组进行现有的运算和自定义的运算。


feature = pd.concat([uidTrain.drop('label', axis=1), uidTest])


# 合并数据,并删除原数据的0

def make_user_wa_feature(waData, feature, smsDx, voiceVl):
    # user_visit_web_cate_num
    t0 = waData[waData.wa_type == 0][['uid', 'wa_name']]  # wa_type为网站与app区分 'uid','wa_name'手机号用户和网站名
    t0 = t0.groupby('uid')['wa_name'].nunique().reset_index()
    t1 = smsDx[smsDx.in_out == 0][['uid', 'opp_num']]
    t1 = t1.groupby('uid')['opp_num'].nunique().reset_index()
    t2 = voiceVl[voiceVl.in_out == 0][['uid', 'opp_num']]
    t2 = t2.groupby('uid')['opp_num'].nunique().reset_index()
    # nunique()即返回的是唯一值的个数
    # reset_index 函数意思就是重新排序，将索引修改为列
    feature = feature.merge(t0, on='uid', how='left')
    feature = feature.merge(t1, on='uid', how='left')
    feature = feature.merge(t2, on='uid', how='left')
    # merge将其连接
    # how 指的是合并(连接)的方式有inner(内连接),left(左外连接),right(右外连接),outer(全外连接);默认为inner
    # on 指的是用于连接的列索引名称。必须存在右右两个DataFrame对象中

    return feature


# 提取特征
# print(waData,'xixixi')
# print(feature,"jjjjj")
feature = make_user_wa_feature(waData, feature, smsDx, voiceVl)

# feature.to_csv('../data/feature_wa_03.csv', index=False)

# 训练集
train = feature[:4999].copy()
train = train.merge(uidTrain, on='uid', how='left')

# 打乱顺序
np.random.seed(201805)
idx = np.random.permutation(len(train))
train = train.iloc[idx]

X_train = train.drop(['uid', 'label'], axis=1).values
y_train = train.label.values

# 测试集
test = feature[4999:].copy()

X_test = test.drop(['uid'], axis=1).values

"""
lgb = lightgbm.LGBMClassifier(boosting_type='gbdt', 
          objective= 'binary',
          metric= 'auc',
          min_child_weight= 1.5,
          num_leaves = 2**5,
          lambda_l2= 10,
          subsample= 0.7,
          colsample_bytree= 0.5,
          colsample_bylevel= 0.5,
          learning_rate= 0.1,
          scale_pos_weight= 20,
          seed= 201805,
          nthread= 4,
          silent= True)
"""
lgb = lightgbm.LGBMClassifier(random_state=201805)


def fitModel(model, feature1):
    X = feature1.drop(['uid', 'label'], axis=1).values
    y = feature1.label.values

    lgb_y_pred = cross_val_predict(model, X, y, cv=5,
                                   verbose=2, method='predict')
    lgb_y_proba = cross_val_predict(model, X, y, cv=5,
                                    verbose=2, method='predict_proba')[:, 1]

    f1score = f1_score(y, lgb_y_pred)
    aucscore = roc_auc_score(y, lgb_y_proba)
    print('F1:', f1score,
          'AUC:', aucscore,
          'Score:', f1score * 0.4 + aucscore * 0.6)
    print(classification_report(y, lgb_y_pred))

    model.fit(X, y)

    featureList0 = list(feature1.drop(['uid', 'label'], axis=1))
    featureImportant = pd.DataFrame()
    featureImportant['feature'] = featureList0
    featureImportant['score'] = lgb.feature_importances_
    featureImportant.sort_values(by='score', ascending=False, inplace=True)
    featureImportant.reset_index(drop=True, inplace=True)
    print(featureImportant)


# 交叉验证模型
fitModel(lgb, train)