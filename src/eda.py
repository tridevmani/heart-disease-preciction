from src.utils import save_plot

def perform_eda(df):

    import seaborn as sns
    import matplotlib.pyplot as plt

    print(df.info())
    print(df.describe())

    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    plt.figure(figsize=(12,10))
    sns.heatmap(numeric_df.corr(), annot=True)
    plt.title("Correlation Heatmap")
    save_plot("heatmap")