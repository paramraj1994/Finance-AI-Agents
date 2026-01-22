import pandas as pd

def build_model(base_revenue, years, assumptions):
    projections = []

    revenue = base_revenue

    for year in years:
        revenue = revenue * (1 + assumptions["revenue_growth"])
        ebitda = revenue * assumptions["ebitda_margin"]
        tax = ebitda * assumptions["tax_rate"]
        pat = ebitda - tax

        projections.append({
            "Year": year,
            "Revenue": revenue,
            "EBITDA": ebitda,
            "PAT": pat
        })

    return pd.DataFrame(projections)
