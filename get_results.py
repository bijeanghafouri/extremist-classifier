import pandas as pd
import os


# path = '/scratch2/chiyuwei/political_leaning/classifier_model/multiple_features/with_undersampling'
path = '/Users/chiyuwei/PycharmProjects/political_leaning/classifier_model/multiple_features/with_undersampling'
neg_pos = ['negative_theta', 'positive_theta']
for theta in neg_pos:
    path_all = os.path.join(path, theta)
    results = pd.DataFrame(columns=['threshold', 'accuracy', 'f1-score'])
    for root, dirs, files in os.walk(path_all):
        for file in files:
            if file == 'report_all.csv':
                df = pd.read_csv(os.path.join(root, file), index_col=0)
                acc = df.loc['Average', 'accuracy']
                f1 = df.loc['Average', 'f1-score']
                threshold = float(root.split('/')[-1])
                results = results.append({
                    'threshold': threshold,
                    'accuracy': float(acc),
                    'f1-score': float(f1)
                }, ignore_index=True)
    results = results.sort_values(by='threshold', ascending=True)
    results.to_csv(os.path.join(path_all, 'results_acc_f1.csv'), index=False)

