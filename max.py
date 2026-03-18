import pandas as pd
import pandas as pd
import numpy as np
from glob import glob
import argparse
import os
data = pd.read_csv('final_sub1.csv') # 将'file.csv'
rows = 480
pridict = []
for i in range(0,rows+1):
    row = data.iloc[i] #
    a = []
    a.append(row[1])
    a.append(row[2])
    a.append(row[3])
    a.append(row[4])
    a.append(row[5])
    a.append(row[6])
    max_value = max(a)  # get max value of array
    index = a.index(max_value)  # get index of max value
    pridict.append(index)
    print(a)
print(pridict)
df_sub = data
df_sub['pridict'] =pridict
df_sub.to_csv(f"final_sub.csv",index=False)