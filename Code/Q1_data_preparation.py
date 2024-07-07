import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from matplotlib.font_manager import FontProperties

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

# Function to calculate WOE and IV
def calculate_woe_iv(df, feature, target, min_samples=0.05, alpha=0.5):
    df = df[[feature, target]].copy()
    
    if df[feature].dtype in ['int64', 'float64'] and df[feature].nunique() > 10:
        df[feature] = pd.qcut(df[feature], q=10, duplicates='drop')
    
    grouped = df.groupby(feature)[target].agg(['count', 'sum']).reset_index()
    grouped.columns = [feature, 'total', 'bad']
    grouped['good'] = grouped['total'] - grouped['bad']
    
    total_bad = grouped['bad'].sum()
    total_good = grouped['good'].sum()
    
    grouped['bad_rate'] = (grouped['bad'] + alpha) / (total_bad + alpha * grouped.shape[0])
    grouped['good_rate'] = (grouped['good'] + alpha) / (total_good + alpha * grouped.shape[0])
    
    grouped['woe'] = np.log(grouped['good_rate'] / grouped['bad_rate'])
    grouped['iv'] = (grouped['good_rate'] - grouped['bad_rate']) * grouped['woe']
    
    grouped['woe'] = grouped['woe'].clip(-20, 20)
    grouped['iv'] = grouped['iv'].clip(0, 1)
    
    return grouped['iv'].sum()

# Process data_1.csv
df1 = pd.read_csv('../Data/data_1.csv')
le = LabelEncoder()
for col in ['X1', 'X3', 'X5', 'X6', 'X7', 'X8', 'X9', 'X11', 'X12']:
    df1[col] = le.fit_transform(df1[col])

for col in ['X2', 'X4', 'X10']:
    df1[col] = pd.cut(df1[col].astype(float), 5, labels=False)

for col in ['X13', 'X14', 'X15']:
    df1[col] = df1[col].apply(lambda x: 1 if x == 2 else 0)

df1.to_csv('../Data/data1.csv', index=False)

# Process data_2.csv
df2 = pd.read_csv('../Data/data_2.csv')
for col in ['X4', 'X5', 'X6', 'X10', 'X12']:
    df2[col] = le.fit_transform(df2[col])

scaler = StandardScaler()
df2[['X2', 'X3', 'X7', 'X13', 'X14']] = scaler.fit_transform(df2[['X2', 'X3', 'X7', 'X13', 'X14']])
df2.to_csv('../Data/data2.csv', index=False)

# Correlation analysis
spearman_corr = df1.corr(method='spearman')
pearson_corr = df1.corr()

spearman_corr.to_csv('../Data/Q1_spearman_correlation_matrix.csv')
pearson_corr.to_csv('../Data/Q1_pearson_correlation_matrix.csv')

sorted_spearman_corr = spearman_corr.stack().sort_values(ascending=False).drop_duplicates()
sorted_pearson_corr = pearson_corr.stack().sort_values(ascending=False).drop_duplicates()

sorted_pearson_corr.to_csv('../Data/Q1_unique_sorted_pearson_correlation.csv')
sorted_spearman_corr.to_csv('../Data/Q1_unique_sorted_spearman_correlation.csv')

# Heatmap plotting
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=16)

def plot_heatmap(corr, title, filename):
    columns = corr.columns.tolist()
    index = corr.index.tolist()
    columns = [columns[-1]] + columns[:-1]
    index = [index[-1]] + index[:-1]
    corr_reordered = corr.loc[index, columns]
    
    plt.figure(figsize=(20, 17))
    sns.heatmap(corr_reordered, annot=False, cmap='RdBu', vmin=-1, vmax=1)
    
    last_row = corr.iloc[-1]
    modified_last_row = pd.Series([last_row.iloc[-1]] + last_row.iloc[:-1].tolist(), index=last_row.index)
    
    for i, val in enumerate(modified_last_row):
        plt.text(i + 0.5, 0.65, f'{val:.2f}', ha='center', va='bottom')
    
    plt.title(title, fontproperties=font)
    plt.savefig(filename)
    plt.tight_layout()
    plt.show()

plot_heatmap(spearman_corr, '斯皮尔曼相关性热图', '../Data/Q1_Spearman_Correlation_Heatmap.png')
plot_heatmap(pearson_corr, '皮尔逊相关性热图', '../Data/Q1_Pearson_Correlation_Heatmap.png')

# IV calculation for both datasets
def calculate_iv_for_dataset(data_path):
    data = pd.read_csv(data_path)
    data.rename(columns={'Y(1=default, 0=non-default)': 'target'}, inplace=True)
    
    scaler = StandardScaler()
    continuous_features = data.select_dtypes(include=['int64', 'float64']).columns.drop('target')
    data[continuous_features] = scaler.fit_transform(data[continuous_features])
    
    iv_values = {feature: calculate_woe_iv(data, feature, 'target') for feature in data.columns if feature != 'target'}
    iv_values = dict(sorted(iv_values.items(), key=lambda x: x[1], reverse=True))
    
    print("IV values for each feature:")
    for feature, iv in iv_values.items():
        print(f"{feature}: {iv}")
    
    selected_features = [feature for feature, iv in iv_values.items() if iv > 0.02]
    print("\nSelected features:")
    print(selected_features)

calculate_iv_for_dataset('../Data/data_1.csv')
calculate_iv_for_dataset('../Data/data_2.csv')