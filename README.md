A comparative study of outlier handling and transformation strategies for predicting company investment values using Linear Regression.

This project predicts Investment2019 for companies with missing values, using financial data from 2018.

Model: Linear Regression, evaluated with 5-fold cross-validation.

Three different approaches.

BEST (comparatively) Approach A : despite not having the single highest R2 score, it delivered the most consistent cross-validated performance.

Key findings : Dropping extreme outliers outperformed winsorisation , High R² does not mean a reliable model , Raw data still carried genuine signal just noisy.

Methedology : Data cleaning, outlier detection, data transformation (log and yeo-johnson), Feature selection, Modelling, Prediction.

Libraries : pandas, numpy, seaborn, matplotlib, scikit-learn.

Author : Rehman
