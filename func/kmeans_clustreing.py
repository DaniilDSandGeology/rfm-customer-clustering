import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

from typing import Tuple, Optional

# ====================== 1. Поиск оптимального K для KMeans ======================


def find_optimal_kmeans(X, k_range=range(2, 11)):
    """
    Простая функция для поиска лучшего количества кластеров с помощью KMeans
    """
    results = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)

        results.append({
            'K': k,
            'Silhouette': sil,
            'Calinski_Harabasz': ch,
            'Davies_Bouldin': db
        })

    results_df = pd.DataFrame(results)
    return results_df


# ====================== 2. Визуализация всех метрик ======================
def plot_kmeans_metrics(results_df):
    """Простая и красивая визуализация всех метрик"""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Silhouette Score (чем выше — тем лучше)
    axes[0, 0].plot(results_df['K'], results_df['Silhouette'],
                    'o-', color='blue', linewidth=2, markersize=8)
    axes[0, 0].set_title('Silhouette Score (выше — лучше)',
                         fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Количество кластеров (K)')
    axes[0, 0].set_ylabel('Silhouette Score')
    axes[0, 0].grid(True)

    # 2. Calinski-Harabasz Score (чем выше — тем лучше)
    axes[0, 1].plot(results_df['K'], results_df['Calinski_Harabasz'],
                    'o-', color='orange', linewidth=2, markersize=8)
    axes[0, 1].set_title(
        'Calinski-Harabasz Score (выше — лучше)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Количество кластеров (K)')
    axes[0, 1].set_ylabel('Calinski-Harabasz')
    axes[0, 1].grid(True)

    # 3. Davies-Bouldin Score (чем ниже — тем лучше)
    axes[1, 0].plot(results_df['K'], results_df['Davies_Bouldin'],
                    'o-', color='green', linewidth=2, markersize=8)
    axes[1, 0].set_title('Davies-Bouldin Score (ниже — лучше)',
                         fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Количество кластеров (K)')
    axes[1, 0].set_ylabel('Davies-Bouldin')
    axes[1, 0].grid(True)

    # 4. Все метрики на одном графике (нормированные)
    # Нормализуем для удобного сравнения
    df_norm = results_df.copy()
    df_norm['Calinski_Harabasz'] = df_norm['Calinski_Harabasz'] / \
        df_norm['Calinski_Harabasz'].max()
    df_norm['Davies_Bouldin'] = 1 - \
        (df_norm['Davies_Bouldin'] /
         df_norm['Davies_Bouldin'].max())  # инвертируем

    axes[1, 1].plot(df_norm['K'], df_norm['Silhouette'],
                    'o-', label='Silhouette', color='blue')
    axes[1, 1].plot(df_norm['K'], df_norm['Calinski_Harabasz'],
                    'o-', label='Calinski (норм.)', color='orange')
    axes[1, 1].plot(df_norm['K'], df_norm['Davies_Bouldin'],
                    'o-', label='Davies-Bouldin (инверт.)', color='green')

    axes[1, 1].set_title('Сравнение всех метрик (нормировано)',
                         fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Количество кластеров (K)')
    axes[1, 1].set_ylabel('Нормированное значение')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    __all__ = ['find_optimal_kmeans', 'plot_kmeans_metrics']
