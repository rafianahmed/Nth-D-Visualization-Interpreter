import numpy as np
from sklearn.decomposition import PCA


def compare_views(df, view1, view2):
    d1 = df[list(view1)].dropna()
    d2 = df[list(view2)].dropna()

    results = []

    set1 = set(view1)
    set2 = set(view2)

    only_1 = sorted(list(set1 - set2))
    only_2 = sorted(list(set2 - set1))
    common = sorted(list(set1 & set2))

    if only_1:
        results.append(f"Only in first view: {', '.join(only_1)}.")
    if only_2:
        results.append(f"Only in second view: {', '.join(only_2)}.")
    if common:
        results.append(f"Common variables: {', '.join(common)}.")

    # Sample size after dropping missing values
    n1 = len(d1)
    n2 = len(d2)
    results.append(f"First view uses {n1} complete rows after removing missing data.")
    results.append(f"Second view uses {n2} complete rows after removing missing data.")

    # PCA comparison
    try:
        if len(d1) >= 2 and len(view1) >= 2:
            p1 = PCA(n_components=1).fit(d1).explained_variance_ratio_[0]
        else:
            p1 = None
    except Exception:
        p1 = None

    try:
        if len(d2) >= 2 and len(view2) >= 2:
            p2 = PCA(n_components=1).fit(d2).explained_variance_ratio_[0]
        else:
            p2 = None
    except Exception:
        p2 = None

    if p1 is not None and p2 is not None:
        results.append(f"First view primary PCA component explains {p1 * 100:.1f}% of variance.")
        results.append(f"Second view primary PCA component explains {p2 * 100:.1f}% of variance.")

        if p1 > p2:
            results.append("First view captures a more dominant overall structure.")
        elif p2 > p1:
            results.append("Second view captures a more dominant overall structure.")
        else:
            results.append("Both views capture a similar level of dominant structure.")

    # Average variance comparison
    try:
        v1 = d1.var().mean()
        v2 = d2.var().mean()

        results.append(f"First view average variance is {v1:.4f}.")
        results.append(f"Second view average variance is {v2:.4f}.")

        if v1 > v2:
            results.append("First view has greater spread across its selected variables.")
        elif v2 > v1:
            results.append("Second view has greater spread across its selected variables.")
        else:
            results.append("Both views have similar spread.")
    except Exception:
        pass

    # Correlation comparison
    try:
        if len(view1) > 1:
            corr1 = d1.corr().abs()
            mask1 = np.triu(np.ones(corr1.shape), k=1).astype(bool)
            vals1 = corr1.where(mask1).stack()
            avg_corr1 = vals1.mean() if not vals1.empty else 0
        else:
            avg_corr1 = 0

        if len(view2) > 1:
            corr2 = d2.corr().abs()
            mask2 = np.triu(np.ones(corr2.shape), k=1).astype(bool)
            vals2 = corr2.where(mask2).stack()
            avg_corr2 = vals2.mean() if not vals2.empty else 0
        else:
            avg_corr2 = 0

        results.append(f"First view average absolute correlation is {avg_corr1:.2f}.")
        results.append(f"Second view average absolute correlation is {avg_corr2:.2f}.")

        if avg_corr1 > avg_corr2:
            results.append("First view shows tighter internal relationships among its variables.")
        elif avg_corr2 > avg_corr1:
            results.append("Second view shows tighter internal relationships among its variables.")
        else:
            results.append("Both views show similar internal relationship strength.")
    except Exception:
        pass

    return results
