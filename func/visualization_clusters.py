import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional


def plot_rfm_cluster_distributions(
    df: pd.DataFrame,
    cluster_col: str = 'kmeans',
    figsize: tuple = (18, 14),
    title: str = 'Анализ распределений RFM-метрик по кластерам'
) -> None:
    """
    Создаёт комплексную визуализацию RFM-метрик по кластерам:
        - 3 Violin plot'а (Recency, LogMonetary, LogFrequency)
        - 1 3D Scatter plot с цветовой кодировкой кластеров

    Параметры:
        df (pd.DataFrame): Датафрейм с колонками:
            - 'Recency', 'LogMonetary', 'LogFrequency', cluster_col
        cluster_col (str): Название колонки с номерами кластеров
        figsize (tuple): Размер фигуры
        title (str): Общий заголовок графика
    """
    if not all(col in df.columns for col in ['Recency', 'LogMonetary', 'LogFrequency', cluster_col]):
        raise ValueError(
            f"Датафрейм должен содержать колонки: Recency, LogMonetary, LogFrequency, {cluster_col}")

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.96)

    # Метрики для violin plot'ов
    metrics = [
        ('Recency', 'Recency (дни с последней покупки)'),
        ('LogMonetary', 'Log(Monetary) — покупательская способность'),
        ('LogFrequency', 'Log(Frequency) — частота покупок')
    ]

    # ==================== 1–3. Violin Plots ====================
    for idx, (metric, plot_title) in enumerate(metrics):
        row, col = idx // 2, idx % 2
        ax = fig.add_subplot(2, 2, idx + 1)

        # Violin plot
        sns.violinplot(
            data=df,
            x=cluster_col,
            y=metric,
            ax=ax,
            palette='Set2',
            inner='quartile',
            cut=0,
            linewidth=1.2
        )

        # Средние значения
        means = df.groupby(cluster_col)[metric].mean()
        for i, mean_val in means.items():
            ax.scatter(
                i, mean_val,
                color='red',
                s=100,
                marker='D',
                zorder=5,
                edgecolors='black',
                linewidth=1.5
            )

        ax.set_title(plot_title, fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Кластер', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.grid(True, alpha=0.3)

    # ==================== 4. 3D Scatter Plot ====================
    ax3d = fig.add_subplot(2, 2, 4, projection='3d')

    # Цвета для кластеров
    n_clusters = df[cluster_col].nunique()
    palette = sns.color_palette('Set2', n_colors=n_clusters)

    # Цвет каждой точки
    colors = [palette[int(cluster)] for cluster in df[cluster_col]]

    # Построение 3D графика
    scatter = ax3d.scatter(
        df['Recency'],
        df['LogMonetary'],
        df['LogFrequency'],
        c=colors,
        s=60,
        alpha=0.75,
        edgecolors='w',
        linewidth=0.4
    )

    ax3d.set_xlabel('Recency (дни)', fontsize=11, labelpad=10)
    ax3d.set_ylabel('Log(Monetary)', fontsize=11, labelpad=10)
    ax3d.set_zlabel('Log(Frequency)', fontsize=11, labelpad=10)

    ax3d.set_title('3D распределение клиентов по RFM-метрикам\n(цвет = кластер)',
                   fontsize=13, fontweight='bold', pad=20)

    # Легенда для 3D
    unique_clusters = sorted(df[cluster_col].unique())
    for i, cluster_id in enumerate(unique_clusters):
        ax3d.scatter([], [], [], color=palette[i], s=80,
                     label=f'Кластер {cluster_id}')

    ax3d.legend(
        title='Кластеры',
        loc='upper left',
        bbox_to_anchor=(0.82, 0.85),
        fontsize=10,
        title_fontsize=11
    )

    # Угол обзора 3D графика
    ax3d.view_init(elev=25, azim=45)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


# Для удобного импорта
__all__ = ['plot_rfm_cluster_distributions']
