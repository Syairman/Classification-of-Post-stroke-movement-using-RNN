import pandas as pd
import matplotlib.pyplot as plt
Data = pd.read_csv(r'D:\Tugas Akhir\Final Dev-4\Results\20181210_DATA_COMPARISON.csv')
Data.boxplot(figsize=[6.4,4.8])
plt.ylabel('Akurasi')
plt.xlabel('Arsitektur Model')
plt.savefig(r'D:\Tugas Akhir\Final Dev-4\Results\20181210_DATA_COMPARISON.png')