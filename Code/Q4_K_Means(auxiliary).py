import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from matplotlib.font_manager import FontProperties

# Font and display settings
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=16)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Load data
df = pd.read_csv("../Data/Q2_german_credit_data_with_scores.csv")
X = df[['Credit_Score']]

# K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['Credit_Rating'] = kmeans.fit_predict(X)

# Calculate and sort cluster default rates
cluster_default_rates = df.groupby('Credit_Rating')['Y(1=default, 0=non-default)'].mean().sort_values(ascending=False)

# Reassign credit ratings based on default rates
credit_score_mapping = {old_score: new_score for new_score, old_score in enumerate(cluster_default_rates.index, 0)}
df['Credit_Rating'] = df['Credit_Rating'].map(credit_score_mapping)

# Save results
df.to_csv('../Data/Q4_german_credit_data_with_ratings.csv', index=False)

# Plotting functions
def plot_kmeans_clusters():
    plt.figure(figsize=(10, 6))
    centers = kmeans.cluster_centers_.flatten()
    colors = ['#8B0000', '#1E90FF', '#556B2F', '#F5DEB3', '#9370DB']
    
    for i, color in enumerate(colors):
        plt.scatter(X[df['Credit_Rating'] == i]['Credit_Score'], [0] * len(X[df['Credit_Rating'] == i]['Credit_Score']), 
                    label=f'聚类 {i+1}', color=color, alpha=0.5)
    
    plt.scatter(centers, [0] * len(centers), c='black', marker='x', s=200, linewidths=3, zorder=3)
    for i, center in enumerate(centers):
        plt.text(center, -0.01, f'{center:.2f}', ha='center')
    
    plt.legend(title='聚类')
    plt.xlabel('信用分数', fontproperties=font)
    plt.ylabel('聚类', fontproperties=font)
    plt.title('信用评分的K均值聚类', fontproperties=font)
    plt.tight_layout()
    plt.savefig('../Data/Q4_kmeans_credit_scores.png')
    plt.show()

def plot_credit_score_boxplot():
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")
    custom_palette = sns.color_palette(["#492D22", "#FBD26A", "#002FA7", "#228B22", "#FF6347"])
    
    sns.boxplot(x='Credit_Rating', y='Credit_Score', palette=custom_palette, data=df)
    sns.swarmplot(x='Credit_Rating', y='Credit_Score', data=df, palette=custom_palette, size=3, alpha=0.5)
    
    plt.title('各聚类的信用评分分布', fontproperties=font)
    plt.xlabel('信用评级', fontproperties=font)
    plt.ylabel('信用评分', fontproperties=font)
    plt.ylim(df['Credit_Score'].min() - 10, df['Credit_Score'].max() + 10)
    plt.tight_layout()
    plt.savefig('../Data/Q4_kmeans_credit_scores_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_default_rates():
    plt.figure(figsize=(12, 6))
    colors = sns.color_palette("RdYlGn", n_colors=len(df['Credit_Rating'].unique()))
    grouped_data = df.groupby('Credit_Rating')['Y(1=default, 0=non-default)'].mean().sort_index()
    ax = grouped_data.plot(kind='bar', color=colors)
    
    for i, v in enumerate(grouped_data):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    plt.title('信用等级与违约率关系')
    plt.xlabel('信用等级')
    plt.ylabel('平均违约率')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('../Data/Q4_credit_rating_default_rates.png', dpi=300, bbox_inches='tight')
    plt.show()

# Execute plotting functions
plot_kmeans_clusters()
plot_credit_score_boxplot()
plot_default_rates()

print("\n各信用等级的平均违约率：")
print(cluster_default_rates)