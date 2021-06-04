import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_loss(ds_result, column_to_display):
    plt.figure(figsize=(16,9), dpi=80)
    plt.title('Loss Distribution', fontsize=16)
    # sns.distplot(ds_result[column_to_display], bins = 20, kde = True, color = "blue")
    sns.histplot(ds_result[column_to_display], bins = 20, kde = True, color = "blue")
    plt.xlim([0.0,.5])
    plt.show()

sns.set(color_codes=True, rc={'figure.figsize':(11, 4)})

