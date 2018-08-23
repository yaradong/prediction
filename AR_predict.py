# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 16:38:16 2018

@author: yaradong
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.timeseries.python.timeseries import NumpyReader

raw_data = pd.read_table('C:/Users/yaradong/Desktop/bitcoin/huobikline.txt',encoding='gbk', sep=',')

raw_data = raw_data.sort_index(ascending=False)


price = np.array(raw_data['收盘价'])[0:200]
time = np.array(range(200))

data = {
   tf.contrib.timeseries.TrainEvalFeatures.TIMES: time,
   tf.contrib.timeseries.TrainEvalFeatures.VALUES: price,
}

reader = NumpyReader(data)

price1 = np.array(raw_data['收盘价'])[0:220]
time1 = np.array(range(220))

data1 = {
   tf.contrib.timeseries.TrainEvalFeatures.TIMES: time1,
   tf.contrib.timeseries.TrainEvalFeatures.VALUES: price1,
}


train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
    reader, batch_size=10, window_size=20)


ar = tf.contrib.timeseries.ARRegressor(
        periodicities=100, input_window_size=15, output_window_size=5,
        num_features=1,
        loss=tf.contrib.timeseries.ARModel.NORMAL_LIKELIHOOD_LOSS)

ar.train(input_fn=train_input_fn, steps=500)

evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
    # keys of evaluation: ['covariance', 'loss', 'mean', 'observed', 'start_tuple', 'times', 'global_step']
evaluation = ar.evaluate(input_fn=evaluation_input_fn, steps=1)

(predictions,) = tuple(ar.predict(
        input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
            evaluation, steps=20)))

plt.figure(figsize=(8, 5))
plt.plot(data1['times'].reshape(-1), data1['values'].reshape(-1), label='origin')
plt.plot(evaluation['times'].reshape(-1), evaluation['mean'].reshape(-1), label='evaluate')
plt.plot(predictions['times'].reshape(-1), predictions['mean'].reshape(-1), label='prediction')
plt.xlabel('time_step')
plt.ylabel('values')
plt.legend(loc=4)
plt.savefig('ar_predict_result.jpg')