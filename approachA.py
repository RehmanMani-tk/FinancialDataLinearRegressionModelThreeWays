#%%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

#%%

df = pd.read_csv('financial_cleaned.csv')
dfA = df.copy()

def data_check(dataframe):
    dataframe.columns = dataframe.columns.str.strip()
    print(dataframe.info())
    print(f'\nrows n columns : {dataframe.shape}')
    print(f'\nnulls are : {dataframe.isnull().sum()}')
    print(f'\ndulicates are: {dataframe.duplicated().sum()}')
    

data_check(dfA)

#%%

dfA
cols_to_adjust = dfA.columns.drop('Stock Code')
for col in cols_to_adjust:
    Q1 = dfA[col].quantile(0.01)  
    Q3 = dfA[col].quantile(0.99)  
    getOutliers= dfA.loc[(dfA[col] < (Q1)) | (dfA[col] > (Q3)), col]
    print(f"{col} Outliers : {len(getOutliers)}")

# %%

Q1 = dfA[cols_to_adjust].quantile(0.01)
Q3 = dfA[cols_to_adjust].quantile(0.99)

dropping_condition = ((dfA[cols_to_adjust] >= Q1) & (dfA[cols_to_adjust] <= Q3)).all(axis=1)
dfA_clean = dfA[dropping_condition].copy()
print(f"Total rows dropped: {len(dfA) - len(dfA_clean)}")

# %%

print(dfA_clean.shape)
print(dfA_clean[cols_to_adjust].skew())

#%%

right_skewed_cols = dfA_clean.columns.drop(['Stock Code','Size','VbyP','Age','Leverage'])
for col in right_skewed_cols:
    negative_values = (dfA_clean[col] < 0).sum()
    print(f'{col}: {negative_values} negative values')

# %%
right_skewed_cols = dfA_clean.columns.drop(['Stock Code','Size','VbyP','Age','Leverage','Stock Return','Investment_2018'])

for col in right_skewed_cols:
    dfA_clean[col] = np.log1p(dfA_clean[col])
    sns.histplot(x=dfA_clean[col], kde=True)
    plt.show()
# %%
from sklearn.preprocessing import PowerTransformer

tranformation_met = PowerTransformer(method='yeo-johnson')
dfA_clean['Stock Return'] = tranformation_met.fit_transform(dfA_clean[['Stock Return']])
dfA_clean['Investment_2018'] = tranformation_met.fit_transform(dfA_clean[['Investment_2018']])


# %%
sns.histplot(x=dfA_clean['Stock Return'], kde=True)
plt.show()
# %%
sns.histplot(x=dfA_clean['Investment_2018'], kde=True)
plt.show()

# %%

target_df = pd.read_csv('Investment_2019.csv')
data_check(target_df)


# %%

merged_df = dfA_clean.merge(target_df, on='Stock Code', how='left')
print(merged_df.shape)
print(merged_df['Investment2019'].isnull().sum())

# %%

sns.histplot(merged_df['Investment2019'], kde=True)
print(merged_df['Investment2019'].skew())
print((merged_df['Investment2019'] < 0).sum())

# %%

merged_df['Investment2019'] = tranformation_met.fit_transform(merged_df[['Investment2019']])
sns.histplot(x=merged_df['Investment2019'], kde=True)
plt.show()

# %%

plt.figure(figsize=(12, 8))
sns.heatmap(merged_df.drop(columns='Stock Code').corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.show()


# %%

df_train = merged_df[merged_df['Investment2019'].notna()]
df_predict = merged_df[merged_df['Investment2019'].isna()]

print(f'Training rows: {len(df_train)}')
print(f'Rows to predict: {len(df_predict)}')



# %%

features = ['Market Value of Equity', 'Investment_2018', 'Total Investment Expenditure']

X = df_train[features]
y = df_train['Investment2019']

print(X.shape)
print(y.shape)

#%%

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#%%


from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#%%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'R² Score: {r2:.4f}')
print(f'MAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')

#%%

y_test_real = tranformation_met.inverse_transform(pd.DataFrame(y_test.values, columns=['Investment2019']))
y_pred_real = tranformation_met.inverse_transform(pd.DataFrame(y_pred, columns=['Investment2019']))

mae_real = mean_absolute_error(y_test_real, y_pred_real)
rmse_real = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

print(f'MAE (real £): {mae_real:,.2f}')
print(f'RMSE (real £): {rmse_real:,.2f}')

# %%
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('model', LinearRegression())
])


cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
print(f'CV R² Scores: {cv_scores}')
print(f'Mean R²: {cv_scores.mean():.4f}')
print(f'Std: {cv_scores.std():.4f}')


# %%

pipeline.fit(X, y)

X_pred = df_predict[features]
predicted_values = pipeline.predict(X_pred)

predicted_real = tranformation_met.inverse_transform(
    pd.DataFrame(predicted_values, columns=['Investment2019'])
)

results_A = df_predict[['Stock Code']].copy()
results_A['Predicted_Investment2019'] = predicted_real
results_A.to_csv('predictions_approach_A.csv', index=False)

# %%
