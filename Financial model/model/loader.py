import pandas as pd

def load_financials(file_path):
    df = pd.read_excel(file_path)

    # Drop fully empty rows
    df = df.dropna(how="all")

    # Forward-fill year headers if needed
    df.columns = [str(c) for c in df.columns]

    return df
