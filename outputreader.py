
import pandas as pd
import numpy as np

data = pd.read_csv('./output.txt', sep='\n')

sigmoid = lambda x: 1 / (1 + np.exp(-x))
sigV = np.vectorize(sigmoid)

predictions = sigV(data)

