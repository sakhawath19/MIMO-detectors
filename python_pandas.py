# Import all libraries needed for the tutorial

from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd
import sys #needed to determine python version number
import matplotlib #needed to determine python matplotlib version
import os # it will be used to remove csv file

print('Python version:' + sys.version)
print('Matplotlib version:' + matplotlib.__version__)
print('Pandas version:' + pd.__version__)

# The initial set of baby names and birth dates
names = ['Bob', 'Jessica', 'Mary', 'John', 'Mel']
births = [968, 155, 77, 578, 973]

# To merge these two list we will use zip function
baby_data_set = list(zip(names,births))

# df is a dataframe object
df = pd.DataFrame(data=baby_data_set,columns=['Names','Births'])

location = r'C:\Python\Pandas_file\births_1880.csv'

# It will create csv file in same directory
#df.to_csv('births_1880.csv')

# It will create csv file in the preffered location. It is not exporting index or header
df.to_csv(location,index=False,header=False)

# Header and index was not exported, so header needed to set null while reading the file
#df = pd.read_csv(location,header=None)

# We have specified the columns name
df = pd.read_csv(location,names=['Names_','Births_'])

os.remove(location)
#os.remove(r'C:\Python\births_1880.csv')

print(df.dtypes)

# checking data type and outlier
print('Data type in births column',df.Births_.dtype)

sorted = df.sort_values(['Births_'],ascending=False)

print('Maximum number of same name:',df['Births_'].max())

print('data sorted according to births number in decending order \n',sorted)
print(sorted.head(1))

print(df)
