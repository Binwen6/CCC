import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# Set display options and font
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=16)

# Load data
df1 = pd.read_csv("../Data/data1.csv")
df2 = pd.read_csv("../Data/data2.csv")

# Prepare data
X1 = df1.loc[:, ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X9', 'X10', 'X21']]
y1 = df1['Y(1=default, 0=non-default)']
X2 = df2.loc[:, 'X1':'X14'].drop(columns=['X1','X11'])
y2 = df2['Y(1=default, 0=non-default)']

# Feature selection
selector = SelectKBest(f_classif, k=9)
X1_selected = selector.fit_transform(X1, y1)
X2_selected = selector.fit_transform(X2, y2)

# Define classifiers
classifiers = {
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='linear', probability=True, random_state=42)
}

def train_and_evaluate_model(X, y, classifier_name, classifier):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    accuracy = accuracy_score(y_test, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    type1_error = fp / (fp + tn)
    type2_error = fn / (fn + tp)
    
    return cm, fpr, tpr, roc_auc, accuracy, type1_error, type2_error

def plot_results(dataset_name, classifier_name, cm, fpr, tpr, roc_auc):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='PuBu')
    plt.title(f'混淆矩阵\n{classifier_name} on {dataset_name}', fontproperties=font)
    plt.ylabel('真值标签', fontproperties=font)
    plt.xlabel('预测标签', fontproperties=font)
    
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率', fontproperties=font)
    plt.ylabel('真正例率', fontproperties=font)
    plt.title(f'ROC曲线\n{classifier_name} on {dataset_name}', fontproperties=font)
    plt.legend(loc="lower right")
    
    plt.savefig(f'../Data/Q3_{classifier_name}_{dataset_name}.png')
    plt.tight_layout()
    plt.show()

results = {
    'Dataset': [], 'Classifier': [], 'Accuracy': [], 'AUC': [], 'Type1-error': [], 'Type2-error': []
}

total_iterations = len(classifiers) * 2 - 1  # Subtract 1 for SVM on df2
with tqdm(total=total_iterations, desc='Training Progress') as pbar:
    for dataset_name, X, y in [("df1", X1_selected, y1), ("df2", X2_selected, y2)]:
        for name, classifier in classifiers.items():
            if dataset_name == "df2" and name == "SVM":
                continue
            
            cm, fpr, tpr, roc_auc, accuracy, type1_error, type2_error = train_and_evaluate_model(X, y, name, classifier)
            print(f"\n{name} on {dataset_name}:")
            print(f"Confusion Matrix:\n{cm}")
            print(f"AUC: {roc_auc:.3f}")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Type I Error: {type1_error:.3f}")
            print(f"Type II Error: {type2_error:.3f}")
            
            results['Dataset'].append(dataset_name)
            results['Classifier'].append(name)
            results['Accuracy'].append(accuracy)
            results['AUC'].append(roc_auc)
            results['Type1-error'].append(type1_error)
            results['Type2-error'].append(type2_error)
            
            plot_results(dataset_name, name, cm, fpr, tpr, roc_auc)
            
            pbar.update(1)

results_df = pd.DataFrame(results)
results_df.to_csv('../Data/Q3_model_performance_metrics.csv', index=False)
print("\nResults have been saved to '../Data/Q3_model_performance_metrics.csv'")