#%%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt


# %%
df = pd.read_csv('financial_cleaned.csv')
dfB = df.copy()

def data_check(dataframe):
    dataframe.columns = dataframe.columns.str.strip()
    print(dataframe.info())
    print(f'\nrows n columns : {dataframe.shape}')
    print(f'\nnulls are : {dataframe.isnull().sum()}')
    print(f'\ndulicates are: {dataframe.duplicated().sum()}')
    

data_check(dfB)

# %%

dfB
cols_to_adjust = dfB.columns.drop('Stock Code')
for col in cols_to_adjust:
    Q1 = dfB[col].quantile(0.01)  
    Q3 = dfB[col].quantile(0.99)  
    getOutliers= dfB.loc[(dfB[col] < (Q1)) | (dfB[col] > (Q3)), col]
    print(f"{col} Outliers : {len(getOutliers)}")

# %%

cols_to_adjust = dfB.columns.drop('Stock Code')

for col in cols_to_adjust:
    p01 = dfB[col].quantile(0.01)
    p99 = dfB[col].quantile(0.99)
    dfB[col] = dfB[col].clip(lower=p01, upper=p99)

print(f'Rows remaining : {len(dfB)}')

# %%

clipped_dfB = dfB.copy()

print(clipped_dfB.shape)
print(clipped_dfB.skew())

# %%

right_skewed_cols = clipped_dfB.columns.drop(['Stock Code', 'Size', 'VbyP', 'Age', 'Leverage', 'Stock Return', 'Investment_2018'])

for col in right_skewed_cols:
    clipped_dfB[col] = np.log1p(clipped_dfB[col])

from sklearn.preprocessing import PowerTransformer
pt_B = PowerTransformer(method='yeo-johnson')
clipped_dfB['Stock Return'] = pt_B.fit_transform(clipped_dfB[['Stock Return']])
clipped_dfB['Investment_2018'] = pt_B.fit_transform(clipped_dfB[['Investment_2018']])


# %%
for col in right_skewed_cols:
    sns.histplot(x=clipped_dfB[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

#%%

for col in ['Stock Return', 'Investment_2018']:
    sns.histplot(x=clipped_dfB[col], kde=True)
    plt.title(f'Distribution of {col} after Yeo-Johnson')
    plt.show()

# %%

df_target = pd.read_csv('Investment_2019.csv')
merged_dfB = clipped_dfB.merge(df_target, on='Stock Code', how='left')
print(merged_dfB.shape)
print(merged_dfB['Investment2019'].isnull().sum())

#%%
pt_B_target = PowerTransformer(method='yeo-johnson')
merged_dfB['Investment2019'] = pt_B_target.fit_transform(merged_dfB[['Investment2019']])


# %%

plt.figure(figsize=(12, 8))
sns.heatmap(merged_dfB.drop(columns='Stock Code').corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Approach B - Feature Correlation Matrix')
plt.show()

# %%
features = ['Market Value of Equity', 'Investment_2018', 'Total Market Value']

df_train_B = merged_dfB[merged_dfB['Investment2019'].notna()]
df_predict_B = merged_dfB[merged_dfB['Investment2019'].isna()]

X_B = df_train_B[features]
y_B = df_train_B['Investment2019']

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score

X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X_B, y_B, test_size=0.2, random_state=42)

model_B = LinearRegression()
model_B.fit(X_train_B, y_train_B)

y_pred_B = model_B.predict(X_test_B)

r2 = r2_score(y_test_B, y_pred_B)
mae = mean_absolute_error(y_test_B, y_pred_B)
rmse = np.sqrt(mean_squared_error(y_test_B, y_pred_B))

print(f'R² Score: {r2:.4f}')
print(f'MAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')

#%%

y_test_real_B = pt_B_target.inverse_transform(pd.DataFrame(y_test_B.values, columns=['Investment2019']))
y_pred_real_B = pt_B_target.inverse_transform(pd.DataFrame(y_pred_B, columns=['Investment2019']))

mae_real_B = mean_absolute_error(y_test_real_B, y_pred_real_B)
rmse_real_B = np.sqrt(mean_squared_error(y_test_real_B, y_pred_real_B))

print(f'MAE (real £): {mae_real_B:,.2f}')
print(f'RMSE (real £): {rmse_real_B:,.2f}')



# %%

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

pipeline_B = Pipeline([
    ('scaler', RobustScaler()),
    ('model', LinearRegression())
])

cv_scores_B = cross_val_score(pipeline_B, X_B, y_B, cv=5, scoring='r2')
print(f'CV R² Scores: {cv_scores_B}')
print(f'Mean R²: {cv_scores_B.mean():.4f}')
print(f'Std: {cv_scores_B.std():.4f}')

# %%
model_B.fit(X_B, y_B)

predicted_values_B = model_B.predict(df_predict_B[features])

predicted_real_B = pt_B_target.inverse_transform(pd.DataFrame(predicted_values_B, columns=['Investment2019']))

results_B = df_predict_B[['Stock Code']].copy()
results_B['Predicted_Investment2019'] = predicted_real_B
results_B.to_csv('predictions_approach_B.csv', index=False)

# %%
