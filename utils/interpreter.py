import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import zscore


def interpret_nd(df, selected_cols):
    data = df[selected_cols].dropna()
    insights = []

    if len(selected_cols) < 2:
        return [("error", "Please select at least 2 variables.")]

    if data.empty:
        return [("error", "No valid rows after removing missing values.")]

    # -------------------------
    # Pairwise correlations
    # -------------------------
    corr = data.corr()
    for i in range(len(selected_cols)):
        for j in range(i + 1, len(selected_cols)):
            a = selected_cols[i]
            b = selected_cols[j]
            c = corr.loc[a, b]

            if pd.notna(c) and abs(c) > 0.6:
                insights.append(("correlation", a, b, float(c)))

    # -------------------------
    # Dominant axis by variance
    # -------------------------
    variances = data.var()
    dominant = variances.idxmax()
    insights.append(("dominant_axis", dominant))

    # -------------------------
    # PCA
    # -------------------------
    try:
        n_components = min(len(selected_cols), data.shape[1], data.shape[0])
        if n_components >= 2:
            pca = PCA(n_components=n_components)
            pca.fit(data)
            insights.append(("pca", float(pca.explained_variance_ratio_[0])))
    except Exception:
        pass

    # -------------------------
    # Clustering
    # -------------------------
    try:
        n_clusters = min(3, len(data))
        if n_clusters >= 2:
            KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(data)
            insights.append(("cluster", n_clusters))
    except Exception:
        pass

    # -------------------------
    # Outliers
    # -------------------------
    try:
        z = np.abs(zscore(data, nan_policy="omit"))
        if isinstance(z, np.ndarray):
            outliers = (z > 3).any(axis=1).sum()
            insights.append(("outliers", int(outliers)))
    except Exception:
        pass

    return insights
