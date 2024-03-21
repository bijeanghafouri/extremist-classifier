import pandas as pd
import numpy as np
import math


def save_pred(index, df, class_save_path):
    df.to_csv(class_save_path + f'pred_true_{index}')


class Report:
    def __init__(self, n):
        self.df_report = pd.DataFrame(np.zeros((n + 4, 4)),
                                      columns=['precision', 'recall', 'f1-score', 'accuracy'],
                                      index=['0', '1', '2', '3', '4', 'Average', 'Standard deviation',
                                             'Confidence Interval low', 'Confidence Interval high'])

        self.df_report_class_0 = pd.DataFrame(np.zeros((n + 4, 3)),
                                              columns=['precision', 'recall', 'f1-score'],
                                              index=['0', '1', '2', '3', '4', 'Average', 'Standard deviation',
                                                     'Confidence Interval low', 'Confidence Interval high'])

        self.df_report_class_1 = pd.DataFrame(np.zeros((n + 4, 3)),
                                              columns=['precision', 'recall', 'f1-score'],
                                              index=['0', '1', '2', '3', '4', 'Average', 'Standard deviation',
                                                     'Confidence Interval low', 'Confidence Interval high'])
        self.number = n

    def get_results(self, i, report_once):
        self.df_report.iloc[i, 0] = report_once['macro avg']['precision']
        self.df_report.iloc[i, 1] = report_once['macro avg']['recall']
        self.df_report.iloc[i, 2] = report_once['macro avg']['f1-score']
        self.df_report.iloc[i, 3] = report_once['accuracy']

    def get_class_report(self, i, report_once):
        if '0' in report_once:
            self.df_report_class_0.iloc[i, 0] = report_once['0']['precision']
            self.df_report_class_0.iloc[i, 1] = report_once['0']['recall']
            self.df_report_class_0.iloc[i, 2] = report_once['0']['f1-score']
        if '1' in report_once:
            self.df_report_class_1.iloc[i, 0] = report_once['1']['precision']
            self.df_report_class_1.iloc[i, 1] = report_once['1']['recall']
            self.df_report_class_1.iloc[i, 2] = report_once['1']['f1-score']

    def save_report(self, report_save_path, class_save_path):
        self.df_report.iloc[-4, :] = list(self.df_report.iloc[:-4, ].mean())
        self.df_report.iloc[-3, :] = list(self.df_report.iloc[:-4, ].std(axis=0, ddof=1))
        self.df_report.iloc[-2, :] = list(
            self.df_report.iloc[-4, ] - 1.96 * self.df_report.iloc[-3, ] / math.sqrt(self.number))
        self.df_report.iloc[-1, :] = list(
            self.df_report.iloc[-4, ] + 1.96 * self.df_report.iloc[-3, ] / math.sqrt(self.number))

        self.df_report_class_0.iloc[-4, :] = list(self.df_report_class_0.iloc[:-4, ].mean())
        self.df_report_class_0.iloc[-3, :] = list(self.df_report_class_0.iloc[:-4, ].std(axis=0, ddof=1))
        self.df_report_class_0.iloc[-2, :] = list(
            self.df_report_class_0.iloc[-4, ] - 1.96 * self.df_report_class_0.iloc[-3, ] / math.sqrt(self.number))
        self.df_report_class_0.iloc[-1, :] = list(
            self.df_report_class_0.iloc[-4, ] + 1.96 * self.df_report_class_0.iloc[-3, ] / math.sqrt(self.number))

        self.df_report_class_1.iloc[-4, :] = list(self.df_report_class_1.iloc[:-4, ].mean())
        self.df_report_class_1.iloc[-3, :] = list(self.df_report_class_1.iloc[:-4, ].std(axis=0, ddof=1))
        self.df_report_class_1.iloc[-2, :] = list(
            self.df_report_class_1.iloc[-4, ] - 1.96 * self.df_report_class_1.iloc[-3, ] / math.sqrt(self.number))
        self.df_report_class_1.iloc[-1, :] = list(
            self.df_report_class_1.iloc[-4, ] + 1.96 * self.df_report_class_1.iloc[-3, ] / math.sqrt(self.number))

        # Round to 4 decimal places
        self.df_report = self.df_report.round(4)
        self.df_report_class_0 = self.df_report_class_0.round(4)
        self.df_report_class_1 = self.df_report_class_1.round(4)

        self.df_report.to_csv(report_save_path)
        self.df_report_class_0.to_csv(class_save_path + 'report_class_0.csv')
        self.df_report_class_1.to_csv(class_save_path + 'report_class_1.csv')
