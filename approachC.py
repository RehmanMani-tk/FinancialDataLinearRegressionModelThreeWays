#%%

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score

# %%
df = pd.read_csv('financial_cleaned.csv')
dfC = df.copy()

# %%
df_target = pd.read_csv('Investment_2019.csv')
merged_dfC = dfC.merge(df_target, on='Stock Code', how='left')
print(merged_dfC.shape)
print(merged_dfC['Investment2019'].isnull().sum())


#%%

import seaborn as sns
import matplotlib.pylab as plt

plt.figure(figsize=(12,8))
sns.heatmap(merged_dfC.drop(columns='Stock Code').corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Approach C - Feature Correlation Matrix (Raw Data)')
plt.show()




# %%
features = merged_dfC.columns.drop(['Stock Code', 'Investment2019']).tolist()

df_train_C = merged_dfC[merged_dfC['Investment2019'].notna()]
df_predict_C = merged_dfC[merged_dfC['Investment2019'].isna()]

X_C = df_train_C[features]
y_C = df_train_C['Investment2019']

X_train_C, X_test_C, y_train_C, y_test_C = train_test_split(X_C, y_C, test_size=0.2, random_state=42)

model_C = LinearRegression()
model_C.fit(X_train_C, y_train_C)

y_pred_C = model_C.predict(X_test_C)

r2 = r2_score(y_test_C, y_pred_C)
mae = mean_absolute_error(y_test_C, y_pred_C)
rmse = np.sqrt(mean_squared_error(y_test_C, y_pred_C))

print(f'R² Score: {r2:.4f}')
print(f'MAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')

# %%



cv_scores_C = cross_val_score(LinearRegression(), X_C, y_C, cv=5, scoring='r2')
print(f'CV R² Scores: {cv_scores_C}')
print(f'Mean R²: {cv_scores_C.mean():.4f}')
print(f'Std: {cv_scores_C.std():.4f}')

# %%
model_C.fit(X_C, y_C)
predicted_values_C = model_C.predict(df_predict_C[features])

results_C = df_predict_C[['Stock Code']].copy()
results_C['Predicted_Investment2019'] = predicted_values_C
results_C.to_csv('predictions_approach_C.csv', index=False)

# %%
