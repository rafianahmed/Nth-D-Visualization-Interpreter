def enrich_language(insights, domain):
    output = []

    for item in insights:
        base = None

        if isinstance(item, str):
            base = item

        elif isinstance(item, (list, tuple)) and len(item) > 0:
            if item[0] == "correlation" and len(item) >= 4:
                _, a, b, c = item
                tone = "positive" if c > 0 else "negative"
                base = f"{a} and {b} show a strong {tone} relationship."

            elif item[0] == "dominant_axis" and len(item) >= 2:
                base = f"Most variation is driven by {item[1]}."

            elif item[0] == "pca" and len(item) >= 2:
                base = f"Primary direction explains {item[1] * 100:.1f}% of variation."

            elif item[0] == "cluster":
                base = "Data forms multiple structural regimes."

            elif item[0] == "outliers" and len(item) >= 2:
                base = f"{item[1]} extreme cases detected."

            else:
                base = str(item)

        else:
            base = str(item)

        if domain == "Finance":
            base += " This has implications for risk–return dynamics."
        elif domain == "Supply Chain":
            base += " This impacts operational efficiency and resilience."
        elif domain == "ML":
            base += " This reflects latent feature structure."
        else:
            base += " This provides a general analytical interpretation."

        output.append(base)

    return output
