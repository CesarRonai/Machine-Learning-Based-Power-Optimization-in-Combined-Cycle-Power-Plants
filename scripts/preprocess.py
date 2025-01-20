import pandas as pd

def load_and_preprocess_data(file_path):
    """Load dataset and preprocess it."""
    df = pd.read_excel(file_path)

    # Remover valores nulos
    df.dropna(inplace=True)

    # Converter tipos de dados se necessário
    df = df.astype(float)

    return df

if __name__ == "__main__":
    df = load_and_preprocess_data(r"C:\Users\Usuário\CCPP\Folds5x2_pp.xlsx")

    print(df.head())
