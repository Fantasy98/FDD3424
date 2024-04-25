import numpy  as np 
import pandas as pd 

df = pd.read_csv("Coarse_Search.csv")
df = df.sort_values(by=['lambda'])
df.to_csv('Sorted_Coarse.csv',sep='&',float_format='%.2e')

df = pd.read_csv("Finer_Search.csv")
df = df.sort_values(by=['lambda'])
df.to_csv('Sorted_Finer.csv',sep='&',float_format='%.2e')
