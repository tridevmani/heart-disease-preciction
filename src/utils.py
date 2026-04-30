import os
import matplotlib.pyplot as plt

def save_plot(filename):
    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig(f"outputs/plots/{filename}.png")
    plt.close()