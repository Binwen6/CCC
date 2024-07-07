import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from scipy.stats import beta

# Set display options and font
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=16)

# Load data
df = pd.read_csv('..\Data\data1.csv')
features = df.columns.tolist()[:-1]

# Data preprocessing
X = df.drop(['Y(1=default, 0=non-default)'], axis=1)
y = df['Y(1=default, 0=non-default)']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)

# Evaluate model
y_pred_proba = xgb.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC: {auc}")

# Train another XGBoost model using DMatrix
dtrain = xgboost.DMatrix(X_train, label=y_train)
dall = xgboost.DMatrix(X)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
}

model = xgboost.train(params, dtrain, num_boost_round=100)
probabilities = model.predict(dall)

def probability_to_score(prob, base_score=850, min_score=300, alpha=0.35, bet=0.5):
    prob = np.clip(prob, 1e-15, 1-1e-15)
    cdf_value = 1 - beta.cdf(prob, alpha, bet)
    score_range = base_score - min_score
    score_raw = cdf_value * score_range
    score = min_score + score_raw
    return np.clip(score, min_score, base_score)

credit_scores = [probability_to_score(p) for p in probabilities]

# Visualize credit score distribution
plt.figure(figsize=(10, 6))
palette = sns.color_palette("Blues", 8)
sns.histplot(credit_scores, kde=True, color=palette[7])
sns.set_style('darkgrid')

plt.title('信用分的分布', fontproperties=font)
plt.xlabel('信用分', fontproperties=font)
plt.ylabel('频率', fontproperties=font)
plt.savefig('../Data/Q2_credit_score_distribution.png')
plt.show()

# Visualize credit score vs target
df['Credit_Score'] = credit_scores
plt.figure(figsize=(10, 6))
blues = sns.color_palette("Blues", 8)
greens = sns.color_palette("Greens", 8)
palette = {1: blues[3], 0: greens[3]}

sns.boxplot(x='Y(1=default, 0=non-default)', y='Credit_Score', data=df, palette=palette)

plt.title('信用分 vs 目标', fontproperties=font)
plt.xlabel('目标 (0: Good, 1: Bad)', fontproperties=font)
plt.ylabel('信用分', fontproperties=font)
plt.savefig('../Data/Q2_credit_score_vs_target.png')
plt.show()

# Print credit score statistics
print("\nCredit Score Statistics:")
print(df['Credit_Score'].describe())

# Save results
df.to_csv('../Data/Q2_german_credit_data_with_scores.csv', index=False)
print("\nResults saved to 'german_credit_data_with_scores.csv'")