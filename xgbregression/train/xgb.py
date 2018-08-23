import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdate

print('Load raw data...')
#导入数据
train_df = pd.read_csv('bit.csv')
test = pd.read_csv('test.csv')
train_df['oc'] = (train_df['open']-train_df['close'])
test['oc'] = (test['open']-test['close'])

train_df['lh'] = (train_df['high']-train_df['low'])
test['lh'] = (test['high']-test['low'])
#筛选特征
select_feat = ['open','high','low','close','volume','money','num','oc','lh']
# select_feat = ['open','high','low','close','volume','money','num']

label_df = pd.DataFrame(index = train_df.index, columns=["label"])
label_df["label"] = np.log(train_df["label"])
train_df = train_df[select_feat]
test_df = test[select_feat]
print('Done!')

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
regr = xgb.XGBRegressor(
                 colsample_bytree=0.5,
                 gamma=0.0,
                 learning_rate=0.01,
                 max_depth=6,
                 min_child_weight=1.5,
                 n_estimators=2200,                                                                  
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.8,
                 seed=2018,
                 silent=1)

regr.fit(train_df, label_df)

# Run prediction on training set to get a rough idea of how well it does.
y_pred = regr.predict(train_df)
y_test = label_df
print("XGBoost score on training set: ", rmse(y_test, y_pred))

# Run prediction on the test set.
y_pred_xgb = regr.predict(test_df)
y_pred = np.exp(y_pred_xgb)



pred_df = pd.DataFrame(y_pred, index=test["id"], columns=["close"])
pred_df.to_csv('output.csv', header=True, index_label='id')

#显示
dateparse = lambda dates:pd.datetime.strptime(dates,'%Y-%m-%d %H:%M:%S')
data_pre = pd.read_csv('output.csv',encoding='utf-8',parse_dates=['id'],date_parser=dateparse)
data_tru = pd.read_csv('test.csv',encoding='utf-8',parse_dates=['id'],date_parser=dateparse)

table_pre = pd.pivot_table(data_pre,index=['id'],values=['close'])
table_tru = pd.pivot_table(data_tru,index=['id'],values=['close'])

fig = plt.figure()
#生成axis对象
ax = fig.add_subplot(111) #本案例的figure中只包含一个图表
#设置x轴为时间格式，这句非常重要，否则x轴显示的将是类似于‘736268’这样的转码后的数字格式
ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d %H:%M:%S'))

#设置x轴坐标值和标签旋转45°的显示方式
plt.xticks(pd.date_range(table_tru.index[0],table_tru.index[-1],freq='min'),rotation=45)
#x轴为table.index，y轴为价格
ax.plot(table_pre.index,table_pre['close'],color='r')
ax.plot(table_tru.index,table_tru['close'],color='b')
plt.show()