import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Font settings
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=16)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def preprocess_data(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def evaluate_model(model, X, y):
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    accuracy = cv_scores.mean()
    auc = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()
    f1 = cross_val_score(model, X, y, cv=5, scoring='f1').mean()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    type1_error = fp / (fp + tn)
    type2_error = fn / (fn + tp)
    
    return accuracy, auc, f1, type1_error, type2_error

def analyze_dataset(df, target_column, dataset_name):
    X, y = preprocess_data(df, target_column)
    
    models = {
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        '决策树': DecisionTreeClassifier(),
        '随机森林': RandomForestClassifier(),
        'SVM': SVC(probability=True)
    }
    
    results = []
    
    for model_name, model in models.items():
        accuracy, auc, f1, type1_error, type2_error = evaluate_model(model, X, y)
        results.append({
            'Dataset': dataset_name,
            '模型': model_name,
            '精确度': accuracy,
            'AUC': auc,
            'F1-分数': f1,
            '第一类错误': type1_error,
            '第二类错误': type2_error
        })
    
    return pd.DataFrame(results)

# Load and analyze datasets
german_df = pd.read_csv('../Data/data1.csv')
australian_df = pd.read_csv('../Data/data2.csv')

german_results = analyze_dataset(german_df, 'Y(1=default, 0=non-default)', 'German Credit')
australian_results = analyze_dataset(australian_df, 'Y(1=default, 0=non-default)', 'Australian Credit')

all_results = pd.concat([german_results, australian_results])

# Visualization
fig, axs = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('模型性能指标比较', fontproperties=font, y=0.99)

metrics = [
    ('精确度', axs[0, 0]),
    ('AUC', axs[0, 1]),
    ('第一类错误', axs[1, 0]),
    ('第二类错误', axs[1, 1])
]

for metric, ax in metrics:
    sns.barplot(x='模型', y=metric, hue='Dataset', data=all_results, ax=ax, palette='Blues_d')
    ax.set_title(f'{metric}的比较 ', fontproperties=font)
    ax.set_xlabel('模型', fontproperties=font)
    ax.set_ylabel(metric, fontproperties=font)
    ax.legend(title='Dataset')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('../Data/Q3_Performance_Metrics_Comparison.png')

# Save results
all_results.to_csv('../Data/Q3_credit_model_comparison_results.csv', index=False)
print("\nResults saved to 'Q3_credit_model_comparison_results.csv'")
print("Visualization saved to 'Q3_Performance_Metrics_Comparison.png'")