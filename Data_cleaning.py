#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%

df1 = pd.read_csv('finanical_report_2018.csv')
dfcopy = df1.copy()

# %%

def data_check(dataframe):
    dataframe.columns = dataframe.columns.str.strip()
    print(dataframe.info())
    print(f'\nrows n columns : {dataframe.shape}')
    print(f'\nnulls are : {dataframe.isnull().sum()}')
    print(f'\ndulicates are: {dataframe.duplicated().sum()}')
    

data_check(dfcopy)


# %%

#dfcopy['Size'] = dfcopy['Size'].astype(float)

try:
    dfcopy['Size'] = dfcopy['Size'].astype(float)
except:
    print(f'cant be converted due to an error')


#%%

try:
    dfcopy.loc[dfcopy['Size']== '#NUM!', 'Size'] = np.nan
    print(f'''converted the string value that was stopping conversion
into a null''')
    dfcopy['Size'] =dfcopy['Size'].astype(float)
    print(f'''\nnow Size got converted into float data type''')
except:
    print(f'still an error')    

# %%

data_check(dfcopy)


# %%

cols_with_nulls = ['Size','Stock Return']
for col in cols_with_nulls:
    sns.histplot(x=dfcopy[col], kde=True)
    plt.show()
    print(f'the mean : {dfcopy[col].mean()}')
    print(f'the median : {dfcopy[col].median()}')
    print(f'the mode : {dfcopy[col].mode()}')
    print() 

# %%

dfcopy['Stock Return'] = dfcopy['Stock Return'].fillna(dfcopy['Stock Return'].median())
dfcopy['Size'] = dfcopy['Size'].fillna(dfcopy['Size'].median())

print(dfcopy.isnull().sum())


#%%

dfcopy['VbyP'] = ((1-1.24*0.12)*dfcopy['Book Value'] +1.24*1.12* dfcopy['Operating Income after Depreciation'] - 1.24*0.12* dfcopy['Annual Dividends'])/ dfcopy['Market Value of Equity']
dfcopy['Investment_2018'] = dfcopy['Total Investment Expenditure'] - dfcopy['Amortization and Depreciation']


# %%

cols_to_visualize = dfcopy.columns.drop('Stock Code')


#%%


for cols in cols_to_visualize:
    sns.histplot(x=dfcopy[cols],kde=True)
    plt.show()


#%%

for cols in cols_to_visualize:
    sns.boxplot(x=dfcopy[cols])
    plt.show()

# %%

for cols in cols_to_visualize:
    print(dfcopy[cols].quantile([0.01,0.10,0.25, 0.50, 0.75, 0.90,0.99]))
    print(f'\n')


# %%
dfcopy.to_csv('financial_cleaned.csv', index=False)



# %%
