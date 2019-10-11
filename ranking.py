import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

cut_off = 0.99
edge = (3000, 30)  # true range
importance = np.loadtxt('./sample_data/importance_test_1_0_c1.csv', delimiter=',')  # shap distribution of an sample


m, n = importance.shape
m_step = edge[0] / m
n_step = edge[1] / n
df = pd.DataFrame(importance)
ax = sns.heatmap(df)
vals = ax.get_yticks()
ax.set_yticklabels(['{:.2f}'.format(x*3000/512) for x in vals])
vals = ax.get_xticks()
ax.set_xticklabels(['{:.2f}'.format(x*30/512) for x in vals])
plt.imsave('./test_1_0_importance.png', df.values, cmap='jet')  # output image for shap
plt.show()

sample = pd.read_excel('./sample_data/test_1_0.xls')  # corresponding original feature list of that sample
sample['importance'] = 0
num_dict = {}
for i, f in sample.iterrows():
    m = int(f['m/z']//m_step)
    n = int(f['t']//n_step)
    if m >= importance.shape[0]:
        m -= 1
    if n >= importance.shape[1]:
        n -= 1
    num_dict[(m, n)] = num_dict.get((m, n), 0) + 1
    pass
for i, f in sample.iterrows():
    m = int(f['m/z']//m_step)
    n = int(f['t']//n_step)
    if m >= importance.shape[0]:
        m -= 1
    if n >= importance.shape[1]:
        n -= 1
    sample.ix[i, 'importance'] = importance[m, n] / num_dict.get((m, n), 0)
sample.to_csv('./test_1_0_ranking.csv')  # output importance rankings for features
