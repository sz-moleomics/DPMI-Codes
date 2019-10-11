import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve


file = './sample_data/result.csv'  # the test result from test.py
test_num = [193, 191, 188, 188, 187, 186, 185, 184, 184, 184]  # number of test samples for each class

test_num = [0].extend(test_num)
split_index = np.cumsum(test_num)
csv = np.loadtxt(file, delimiter=',')
confusion_matrix = np.zeros([10, 10], dtype='int')
confusion_raw_loss = np.zeros([10, 10])

# confusion matrix
for i in range(csv.shape[0]):
    ground_truth = np.zeros(10)
    label = int(csv[i, 0])
    ground_truth[label] = 1
    pred = int(csv[i, 1])
    confusion_matrix[label, pred] += 1
    if label == pred:
        confusion_raw_loss[label, :] += csv[i, 2::] - ground_truth

num = np.sum(confusion_matrix, axis=1).reshape([10, 1])
print(confusion_matrix)
print(confusion_raw_loss)
sns.heatmap(np.abs(confusion_raw_loss)/num, cmap='coolwarm', annot=confusion_matrix, fmt='d',
            vmax=np.max(np.abs(confusion_raw_loss/num)), vmin=-np.max(np.abs(confusion_raw_loss/num)))
plt.title('Confusion Matrix (colored by loss)')
np.savetxt('./temp.csv', np.abs(confusion_raw_loss)/num)

csv_sort = csv[np.argsort(csv[:, 0]), :]
plt.figure()
sns.heatmap(csv_sort[:, 2::], cmap='RdYlBu_r')
plt.title('Class Probabilities for Each Sample')

df = np.zeros([csv.shape[0], 3])
df[:, 0] = np.linspace(1, csv_sort.shape[0], csv_sort.shape[0])
df[:, 1] = csv_sort[df[:, 0].astype('int')-1, 2+csv_sort[:, 0].astype('int')]
df_bool = csv_sort[:, 0] == csv_sort[:, 1]
plt.figure()
csv_sort[df[:, 0].astype('int')-1, 2+csv_sort[:, 0].astype('int')] = -1
best_left = np.max(csv_sort[:, 2::], axis=1)
plt.vlines(df[:, 0], 0, best_left, colors='grey', linewidth=.5)
sns.scatterplot(x=df[:, 0], y=df[:, 1], hue=df_bool, s=5, linewidth=0)
plt.xlim([0, 1870])
plt.ylim([0, 1.1])
plt.xlabel('Index')
plt.ylabel('Probability of true class')
plt.legend(loc='center right')

out = np.zeros([90, 4])
for i in range(90):
    thresh = 0.1 + 0.01 * i
    if thresh == 1:
        thresh = 0.99
    correct_and_sure = np.count_nonzero(np.logical_and(csv[:, 0] == csv[:, 1], np.max(csv[:, 2::], axis=1) >= thresh))
    wrong_and_sure = np.count_nonzero(np.logical_and(csv[:, 0] != csv[:, 1], np.max(csv[:, 2::], axis=1) >= thresh))
    unsure = np.count_nonzero(np.max(csv[:, 2::], axis=1) < thresh)
    print(correct_and_sure/1870, wrong_and_sure/1870, unsure/1870)
    out[i, :] = (thresh, correct_and_sure, wrong_and_sure, unsure)

df[:, 2] = best_left

# ROC
plt.figure()
for i in range(10):
    y = csv[:, 0:2] * 1
    y[csv[:, 0] == i, 0] = 1
    y[csv[:, 0] != i, 0] = 0
    fpr, tpr, _ = roc_curve(y[:, 0], csv[:, 2+i], pos_label=1)
    plt.plot(fpr, tpr)
plt.title('ROC')
plt.show()

