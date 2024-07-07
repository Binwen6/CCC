import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set up matplotlib parameters
plt.style.use('seaborn')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Load and prepare data
df = pd.read_csv("../Data/Q2_german_credit_data_with_scores.csv")
scores = df['Credit_Score'].values
scaler = MinMaxScaler()
normalized_scores = scaler.fit_transform(scores.reshape(-1, 1)).flatten()

# Define objective function and constraint
def objective(boundaries, scores):
    boundaries = np.sort(np.concatenate(([0], boundaries, [1])))
    groups = np.digitize(scores, boundaries[1:-1])
    return sum(np.var(scores[groups == i]) for i in range(len(boundaries)-1))

def constraint(boundaries):
    return np.diff(np.concatenate(([0], boundaries, [1])))

# Optimize boundaries
n_groups = 5
initial_boundaries = np.linspace(0, 1, n_groups+1)[1:-1]
result = minimize(objective, initial_boundaries, args=(normalized_scores,),
                  method='SLSQP', constraints={'type': 'ineq', 'fun': constraint})

optimal_boundaries = np.sort(np.concatenate(([0], result.x, [1])))

# Assign credit ratings
credit_ratings = np.digitize(normalized_scores, optimal_boundaries[1:-1])
credit_ratings = 5 - n_groups + credit_ratings 
df['Credit_Rating'] = credit_ratings

# Save results
df.to_csv('../Data/Q4_german_credit_data_with_ratings_nonlinear.csv', index=False)

# Print results
print(df[['Credit_Score', 'Credit_Rating']])
print("\nOptimal boundaries:")
for i, boundary in enumerate(optimal_boundaries[1:-1], 1):
    print(f"Boundary {i}: {scaler.inverse_transform([[boundary]])[0][0]:.2f}")

# Plotting functions
def plot_credit_score_vs_rating():
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=df['Credit_Rating'].min(), vmax=df['Credit_Rating'].max())
    
    scatter = ax.scatter(df['Credit_Score'], df['Credit_Rating'], 
                         c=df['Credit_Rating'], cmap=cmap, norm=norm, 
                         marker='o', alpha=0.6, s=50)
    
    plt.colorbar(scatter, label='信用等级', ax=ax)
    ax.set_xlabel('信用评分', fontsize=12)
    ax.set_ylabel('信用等级', fontsize=12)
    ax.set_title('信用评分 vs. 信用等级', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    for boundary in optimal_boundaries[1:-1]:
        ax.axvline(scaler.inverse_transform([[boundary]])[0][0], 
                   color='#FF6347', linestyle='--', linewidth=2, 
                   label='边界线' if boundary == optimal_boundaries[1] else "")
    
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('../Data/Q4_Credit_Score_vs_Credit_Rating.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_credit_analysis():
    cluster_centers = df.groupby('Credit_Rating')['Credit_Score'].mean()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    cmap = plt.cm.viridis
    norm = plt.Normalize(df['Credit_Rating'].min(), df['Credit_Rating'].max())

    # Subplot 1: Credit Score vs Credit Rating
    scatter = ax1.scatter(df['Credit_Score'], df['Credit_Rating'], c=df['Credit_Rating'], cmap=cmap, norm=norm, alpha=0.6)
    ax1.set_xlabel('信用评分', fontsize=12)
    ax1.set_ylabel('信用等级', fontsize=12)
    ax1.set_title('信用评分 vs. 信用等级', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)

    for rating, center in cluster_centers.items():
        ax1.scatter(center, rating, color='#6A5ACD', s=150, marker='x', edgecolor='black', linewidth=1)
        ax1.annotate(f'({center:.2f}, {rating})', (center, rating), xytext=(5, 5), 
                     textcoords='offset points', color='red', fontweight='bold')

    for boundary in optimal_boundaries[1:-1]:
        ax1.axvline(scaler.inverse_transform([[boundary]])[0][0], color='#6A5ACD', linestyle='--')

    # Subplot 2: Box plot of Credit Scores by Credit Rating
    sns.boxplot(x='Credit_Rating', y='Credit_Score', data=df, ax=ax2, palette='viridis')
    ax2.set_xlabel('信用等级', fontsize=12)
    ax2.set_ylabel('信用评分', fontsize=12)
    ax2.set_title('各信用等级的信用评分分布', fontsize=14)

    # Subplot 3: Default Rates by Credit Rating
    default_rates = df.groupby('Credit_Rating')['Y(1=default, 0=non-default)'].mean()
    bars = ax3.bar(default_rates.index, default_rates.values, color=cmap(norm(default_rates.index)))
    ax3.set_xlabel('信用等级', fontsize=12)
    ax3.set_ylabel('违约率', fontsize=12)
    ax3.set_title('各信用等级的违约率', fontsize=14)
    ax3.set_ylim(0, max(default_rates.values) * 1.1)

    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2%}',
                 ha='center', va='bottom', fontweight='bold')

    fig.colorbar(scatter, ax=ax3, orientation='vertical', label='信用等级')
    plt.tight_layout()
    plt.savefig('../Data/Q4_Credit_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(default_rates)

# Generate plots
plot_credit_score_vs_rating()
plot_credit_analysis()