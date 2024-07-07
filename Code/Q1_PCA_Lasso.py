import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

# Set font for Chinese characters
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=16)

# Load and preprocess data
df = pd.read_csv('..\Data\data1.csv')
X = df.drop('Y(1=default, 0=non-default)', axis=1)
y = df['Y(1=default, 0=non-default)']

# Standardize and apply PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Split data and apply Lasso regression
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Calculate feature importance
feature_importance = np.abs(lasso.coef_)
original_feature_importance = np.abs(pca.components_.T.dot(lasso.coef_))

# Select features
threshold = np.percentile(feature_importance, 50)
selected_features = feature_importance >= threshold
original_selected_features = original_feature_importance >= np.percentile(original_feature_importance, 50)

# Print results
print("Selected PCA components:", np.sum(selected_features))
print("Selected original features:", np.sum(original_selected_features))
print("Selected original feature indices:", np.where(original_selected_features)[0])
print("Selected original feature names:", X.columns[original_selected_features].tolist())

# Create and save feature importance DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': original_feature_importance
}).sort_values('Importance', ascending=False)

feature_importance_df.to_csv('..\Data\Q1_feature_importance.csv', index=False)
print("\nFeature importance scores have been saved to '..\Data\feature_importance.csv'")

print("\nFeature importance scores:")
for _, row in feature_importance_df.iterrows():
    print(f"{row['Feature']}: {row['Importance']}")

# Plot feature importance with thresholds
fixed_threshold = 0.010
k = 10
top_k_threshold = np.sort(original_feature_importance)[-k]
n = 0.5
sorted_importance = np.sort(original_feature_importance)[::-1]
cumsum_importance = np.cumsum(sorted_importance)
n_threshold = sorted_importance[np.where(cumsum_importance >= n * sum(original_feature_importance))[0][0]]

plt.figure(figsize=(15, 10))
indices = np.arange(len(original_feature_importance))
bars = plt.bar(indices, original_feature_importance, align='center')

plt.xticks(indices, X.columns, rotation='vertical')
plt.ylabel('特征量重要度', fontproperties=font)
plt.title('不同阈值下的特征量重要度', fontproperties=font)

plt.axhline(y=fixed_threshold, color='r', linestyle='--', linewidth=2)
plt.text(len(indices), fixed_threshold, 'Fixed Threshold', color='red', va='bottom', ha='right')
plt.axhline(y=top_k_threshold, color='g', linestyle='--', linewidth=2)
plt.text(len(indices), top_k_threshold, f'Top {k} Threshold', color='green', va='bottom', ha='right')
plt.axhline(y=n_threshold, color='b', linestyle='--', linewidth=2)
plt.text(len(indices), n_threshold, f'Top {n:.1%} Weight Sum Threshold', color='blue', va='bottom', ha='right')

for i, bar in enumerate(bars):
    if original_feature_importance[i] >= fixed_threshold:
        bar.set_color('#C70039')
    elif original_feature_importance[i] >= top_k_threshold:
        bar.set_color('green')
    elif original_feature_importance[i] >= n_threshold:
        bar.set_color('#0000CD')

plt.savefig('../Data/Q1_feature_importance_with_thresholds.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()