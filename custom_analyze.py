import copy

import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np

Ldef = 2  # inspect layer position
original_dataset_name = 'cifar10'

df_path = f'./results/res.csv'

df = pd.read_csv(df_path)
df = df.sort_values(by='poisoned', ascending=True)

clean_df = df[~df['poisoned']]
poisoned_df = df[df['poisoned']]

metrics = df['anomaly_metric']
labels = df['poisoned']
metrics = list(metrics.astype('float'))
labels = list(labels.astype('int'))

train_ratio = 0.1
REPEAT_TIMES = 10
TPRs = []
FPRs = []

for _time in range(REPEAT_TIMES):
    metrics_temp = copy.deepcopy(metrics)

    print(f'Trial {_time} start!\n')
    # randomly select 30% poisoned/benign models as the Dev set to parameterize the method,
    # i.e., determine the threshold
    poisoned_models_for_parameterize_index = np.random.randint(low=0, high=len(clean_df) - 1,
                                                               size=int(train_ratio * len(clean_df)))
    benign_models_for_parameterize_index = poisoned_models_for_parameterize_index + len(clean_df)

    median_dev_metric = np.array(1.0)
    metrics_temp = abs(metrics_temp - median_dev_metric)

    # make the dev set (to parameterize) and test set (for evaluation)
    labels_dev, metrics_dev, labels_test, metrics_test = [], [], [], []
    for _id in range(len(labels)):
        if _id in np.concatenate(
                (poisoned_models_for_parameterize_index, benign_models_for_parameterize_index)).tolist():
            labels_dev.append(labels[_id])
            metrics_dev.append(metrics_temp[_id])
        else:
            labels_test.append(labels[_id])
            metrics_test.append(metrics_temp[_id])

    # parameterization (determine the threshold of anomaly metric)
    print('--------Method Parameterization on Dev Set---------')
    # compute roc on dev set
    fpr_dev, tpr_dev, thresholds_dev = roc_curve(labels_dev, metrics_dev)
    roc_auc_dev = auc(fpr_dev, tpr_dev)
    print(f'Dev AUC={roc_auc_dev}')

    # select the best threshold
    fpr_dev_005_distances = abs(fpr_dev - 0.05)
    temp_index = 0
    max_index_dev = None
    try:
        while 1:
            temp_index = fpr_dev_005_distances.tolist().index(min(fpr_dev_005_distances), temp_index + 1)
            # print(f'temp_index:{temp_index}')
    except ValueError:
        max_index_dev = temp_index
        print(f'max_index_dev:{max_index_dev}')

    if fpr_dev[max_index_dev] > 0.05:
        threshold_param = (thresholds_dev[max_index_dev] + thresholds_dev[max_index_dev + 1]) / 2.
    else:
        threshold_param = (thresholds_dev[max_index_dev] + thresholds_dev[max_index_dev - 1]) / 2.

    # threshold_param = 1.0
    print(f'The best threshold on the dev set: {threshold_param}')

    # test the parameterized method on the test set
    print('--------Method Evaluation on Test Set---------')
    selected_metric_threshold = threshold_param  # ours:1.0, DF-TND: 100
    y_pred = (np.array(metrics_test) > selected_metric_threshold).astype(int).tolist()
    y_true = labels_test


    def perf_measure(y_true, y_pred):
        TP, FP, TN, FN = 0, 0, 0, 0
        for i in range(len(y_true)):
            if y_true[i] == 1 and y_pred[i] == 1:
                TP += 1
            if y_true[i] == 0 and y_pred[i] == 1:
                FP += 1
            if y_true[i] == 0 and y_pred[i] == 0:
                TN += 1
            if y_true[i] == 1 and y_pred[i] == 0:
                FN += 1
        return TP, FP, TN, FN


    TP, FP, TN, FN = perf_measure(y_true, y_pred)
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    print(f'TPR:{TPR}, FPR:{FPR}\n')
    TPRs.append(TPR)
    FPRs.append(FPR)

print(f'{REPEAT_TIMES} TPRs: {TPRs}')
print(f'{REPEAT_TIMES} FPRs: {FPRs}')
ave_TPR = np.average(TPRs)
ave_FPR = np.average(FPRs)
print(f'Average: '
      f'TPR/FPR = {round(ave_TPR, 2)}/{round(ave_FPR, 2)}')

best_TFPR = max(np.array(TPRs) - np.array(FPRs)).tolist()
best_index = (np.array(TPRs) - np.array(FPRs)).tolist().index(best_TFPR)
print(f'Result: '
      f'TPR/FPR = {round(TPRs[best_index], 2)}/{round(FPRs[best_index], 2)}')
