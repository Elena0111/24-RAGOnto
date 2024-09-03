# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

"""# Comparing RAG and RAGOnto's values on each metric"""

columns = ["context_precision", "faithfulness", "answer_relevancy", "context_recall", "context_entity_recall", "answer_similarity", "summary_score"]

table1= pd.read_csv("/content/drive/MyDrive/Innovazione Digitale/Results/cyber_answer_evaluations.csv", skiprows=2)
table2= pd.read_csv("/content/drive/My Drive/Innovazione Digitale/Results/cyber_onto_answer_evaluations.csv", skiprows=2)


df1 = pd.DataFrame(table1, columns=columns)
df2 = pd.DataFrame(table2, columns=columns)


df = pd.concat({'RAG cybersecurity': df1.melt(), 'RAGOnto cybersecurity': df2.melt()}, names=['source', 'old_index'])

df = df.reset_index(level=0).reset_index(drop=True)
plt.xticks(rotation=45)
sns.boxplot(data=df, x='variable', y='value', hue='source',palette='turbo')
plt.legend(loc='lower left', title='Legend', fontsize='x-small')
plt.savefig("boxplot_cybersecurity.png")
plt.show()

columns = ["context_precision", "faithfulness", "answer_relevancy", "context_recall", "context_entity_recall", "answer_similarity", "summary_score"]

tab1= pd.read_csv("/content/drive/MyDrive/Innovazione Digitale/Results/salmon_answer_evaluations.csv", skiprows=2)
tab2= pd.read_csv("/content/drive/My Drive/Innovazione Digitale/Results/salmon_onto_answer_evaluations.csv", skiprows=2)


df1 = pd.DataFrame(tab1, columns=columns)
df2 = pd.DataFrame(tab2, columns=columns)

df = pd.concat({'RAG salmon': df1.melt(), 'RAGOnto salmon': df2.melt()}, names=['source', 'old_index'])
df = df.reset_index(level=0).reset_index(drop=True)
plt.xticks(rotation=45)
sns.boxplot(data=df, x='variable', y='value', hue='source',palette='turbo')
plt.legend(loc='lower left', title='Legend', fontsize='x-small')
plt.savefig("boxplot_salmon.png")
plt.show()

"""# Comparing RAG and RAGOnto's means of metrics on each query"""

df1 = pd.read_csv("/content/drive/My Drive/Innovazione Digitale/" + "Results/salmon_answer_evaluations.csv",skiprows=2)
df2 = pd.read_csv("/content/drive/My Drive/Innovazione Digitale/" + "Results/salmon_onto_answer_evaluations.csv",skiprows=2)

numeric_df1 = df1.select_dtypes(include=['number'])
numeric_df2 = df2.select_dtypes(include=['number'])

mean1 = numeric_df1.mean(axis=1)
mean2 = numeric_df2.mean(axis=1)

means_df = pd.DataFrame({
    'Salmon RAG': mean1,
    'Salmon RAGOnto': mean2
})

ax = means_df.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Query')
plt.ylabel('Mean')
ax.set_xticklabels(range(1, len(means_df) + 1))
ax.set_yticks(np.linspace(0, 1, 11, dtype=float))
ax.set_yticklabels([round(val, 1) for val in np.linspace(0, 1, 11, dtype=float)])
plt.xticks(rotation=0)
plt.legend(loc = "upper left", fontsize = "x-small", title='Legend')
plt.savefig("SalmonMetricsMeans.png")
plt.show()

df1 = pd.read_csv("/content/drive/My Drive/Innovazione Digitale/" + "Results/cyber_answer_evaluations.csv",skiprows=2)
df2 = pd.read_csv("/content/drive/My Drive/Innovazione Digitale/" + "Results/cyber_onto_answer_evaluations.csv",skiprows=2)
df2.drop(columns=['Unnamed: 0'], inplace=True)

numeric_df1 = df1.select_dtypes(include=['number'])
numeric_df2 = df2.select_dtypes(include=['number'])

mean1 = numeric_df1.mean(axis=1)
mean2 = numeric_df2.mean(axis=1)

means_df = pd.DataFrame({
    'Cybersecurity RAG': mean1,
    'Cybersecurity RAGOnto': mean2
})

ax = means_df.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Query')
plt.ylabel('Mean')
ax.set_xticklabels(range(1, len(means_df) + 1))
ax.set_yticks(np.linspace(0, 1, 11, dtype=float))
ax.set_yticklabels([round(val, 1) for val in np.linspace(0, 1, 11, dtype=float)])
plt.xticks(rotation=0)
plt.legend(loc = "upper left", fontsize = "x-small", title='Legend')
plt.savefig("CybersecurityMetricsMeans.png")
plt.show()