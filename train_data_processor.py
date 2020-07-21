import xlearn as xl
from sklearn.datasets import dump_svmlight_file

from pandas import read_csv
import pandas as pd
import numpy as np

df = read_csv('./click_data_train_3.csv', header=None, skiprows=[0], usecols=range(1, 15))

X = pd.get_dummies(df[[1,2,4,5,6,7,8,9,10,11,12,13,14]])
mat = X.values
y = df[3].values.tolist()


dump_svmlight_file(mat, y, 'testdata.libsvm')




