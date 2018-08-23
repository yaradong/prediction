import pandas as pd
import numpy as np
import xgboost as xgb

test = pd.read_csv('test.csv')
test['oc'] = (test['open']-test['close'])

test['lh'] = (test['high']-test['low'])
#筛选特征
select_feat = ['open','high','low','close','volume','money','num','oc','lh']

X_test = test[select_feat].values
dtest = xgb.DMatrix(X_test)

regr = xgb.Booster({'nthread':1}) #init model
regr.load_model("xgb.model") # load data

# Run prediction on the test set.
y_pred_xgb = regr.predict(dtest)
y_pred = np.exp(y_pred_xgb)
print(y_pred)
