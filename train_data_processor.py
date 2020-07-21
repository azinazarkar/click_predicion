import xlearn as xl
from sklearn.datasets import dump_svmlight_file

from pandas import read_csv
import pandas as pd
import gc
import numpy as np
from imblearn.over_sampling import SMOTE
oversample = SMOTE()


# num = 2
# dflist = []
# for i in range(1, num):
#     dflist.append(read_csv('./click_data_train_'+str(i+1)+'.csv', header=None, skiprows=range(0, 30_001), usecols=range(1, 15)))
#

# df = dflist[0]
#
# X = pd.get_dummies(df[[1,2,4,5,6,7,8,9,10,11,12,13,14]])
#
#
# mat = X.values
# y = df[3].values
# del X
# gc.collect()
# X_ov, y_ov = oversample.fit_resample(mat, y)
#
#
# dump_svmlight_file(X_ov, y_ov, 'train_data.libsvm')

#
# num = 2
# dflist = []
# for i in range(1, num):
#     dflist.append(read_csv('./click_data_train_'+str(i+3)+'.csv', header=None, skiprows=range(0, 30_001), usecols=range(1, 15)))
#
#
# df = dflist[0]
#
# X = pd.get_dummies(df[[1,2,4,5,6,7,8,9,10,11,12,13,14]])
#
#
# mat = X.values
# y = df[3].values
# del X
# gc.collect()
# X_ov, y_ov = oversample.fit_resample(mat, y)
#
#
# dump_svmlight_file(mat, y, 'validation_data.libsvm')


num = 2
dflist = []
for i in range(1, num):
    dflist.append(read_csv('./click_data_train_'+str(i+5)+'.csv', header=None, skiprows=range(0, 30_001), usecols=range(1, 15)))


df = dflist[0]

X = pd.get_dummies(df[[1,2,4,5,6,7,8,9,10,11,12,13,14]])


mat = X.values
y = df[3].values
del X
gc.collect()
X_ov, y_ov = oversample.fit_resample(mat, y)


dump_svmlight_file(mat, y, 'test_data.libsvm')



