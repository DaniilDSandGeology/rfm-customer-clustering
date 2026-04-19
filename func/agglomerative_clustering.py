import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import Tuple


def find_optimal_agglomerative(
    X,
    k_range=range(2, 11),
    linkage: str = 'ward'
) -> pd.DataFrame:
    """
    Поиск оптимального количества кластеров с помощью AgglomerativeClustering.

    Параметры:
        X (array-like): Предобработанные данные
        k_range (range): Диапазон количества кластеров для проверки
        linkage (str): Тип связи ('ward', 'complete', 'average', 'single')

    Возвращает:
        pd.DataFrame: Таблица с метриками для каждого K
    """
    results = []

    for k in k_range:
        agg = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        labels = agg.fit_predict(X)

        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)

        results.append({
            'K': k,
            'Silhouette': round(sil, 4),
            'Calinski_Harabasz': round(ch, 2),
            'Davies_Bouldin': round(db, 4)
        })

    results_df = pd.DataFrame(results)
    return results_df


def plot_agglomerative_metrics(results_df: pd.DataFrame) -> None:
    """
    Визуализирует метрики качества для Agglomerative Clustering.
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # 1. Silhouette Score
    axes[0, 0].plot(results_df['K'], results_df['Silhouette'],
                    'o-', color='blue', linewidth=2.5, markersize=8)
    axes[0, 0].set_title('Silhouette Score (чем выше — тем лучше)',
                         fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Количество кластеров (K)')
    axes[0, 0].set_ylabel('Silhouette Score')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Calinski-Harabasz Score
    axes[0, 1].plot(results_df['K'], results_df['Calinski_Harabasz'],
                    'o-', color='orange', linewidth=2.5, markersize=8)
    axes[0, 1].set_title('Calinski-Harabasz Score (чем выше — тем лучше)',
                         fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Количество кластеров (K)')
    axes[0, 1].set_ylabel('Calinski-Harabasz')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Davies-Bouldin Score
    axes[1, 0].plot(results_df['K'], results_df['Davies_Bouldin'],
                    'o-', color='green', linewidth=2.5, markersize=8)
    axes[1, 0].set_title('Davies-Bouldin Score (чем ниже — тем лучше)',
                         fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Количество кластеров (K)')
    axes[1, 0].set_ylabel('Davies-Bouldin')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Сравнение всех метрик (нормированное)
    df_norm = results_df.copy()
    df_norm['Calinski_Harabasz'] = df_norm['Calinski_Harabasz'] / \
        df_norm['Calinski_Harabasz'].max()
    df_norm['Davies_Bouldin'] = 1 - \
        (df_norm['Davies_Bouldin'] / df_norm['Davies_Bouldin'].max())

    axes[1, 1].plot(df_norm['K'], df_norm['Silhouette'],
                    'o-', label='Silhouette', color='blue', linewidth=2)
    axes[1, 1].plot(df_norm['K'], df_norm['Calinski_Harabasz'],
                    'o-', label='Calinski (норм.)', color='orange', linewidth=2)
    axes[1, 1].plot(df_norm['K'], df_norm['Davies_Bouldin'],
                    'o-', label='Davies-Bouldin (инвертировано)', color='green', linewidth=2)

    axes[1, 1].set_title('Сравнение всех метрик (нормировано)',
                         fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Количество кластеров (K)')
    axes[1, 1].set_ylabel('Нормированное значение')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Для удобного импорта
__all__ = ['find_optimal_agglomerative', 'plot_agglomerative_metrics']
