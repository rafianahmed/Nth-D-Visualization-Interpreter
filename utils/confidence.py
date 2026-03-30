import numpy as np

def confidence_score_nd(df, selected_cols):
    data = df[selected_cols].dropna()
    n = len(data)

    # Not enough data
    if n < 30:
        return 30

    # -------------------------
    # Data completeness
    # -------------------------
    completeness = 1 - df[selected_cols].isna().mean().mean()

    # -------------------------
    # Correlation strength
    # -------------------------
    corr = data.corr().abs()

    if len(selected_cols) > 1:
        mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
        corr_vals = corr.where(mask).stack()
        avg_corr = corr_vals.mean() if not corr_vals.empty else 0
    else:
        avg_corr = 0

    # -------------------------
    # Variance quality
    # -------------------------
    variance_quality = data.std().replace(0, np.nan).notna().mean()

    # -------------------------
    # Final score (weighted)
    # -------------------------
    score = (
        0.4 * completeness * 100 +
        0.35 * avg_corr * 100 +
        0.25 * variance_quality * 100
    )

    return int(min(100, max(0, score)))
