import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.model_selection import KFold
from imblearn.over_sampling import RandomOverSampler
from transformers import RobertaForSequenceClassification
from torch.utils.data import DataLoader
from Report import Report
from Report import save_pred
from Dataset import CustomDataset
from Trainer import Trainer
from Predictor import Predictor
import sys
import torch.nn as nn
import logging
from logger_config import configure_logger


def preprocess_data(df_train, df_val, df_test):
    ros = RandomOverSampler(random_state=42)

    X_resampled, y_resampled = ros.fit_resample(df_train['text'].values.reshape(-1, 1),
                                                df_train['label'].values)

    df_train_oversample = pd.DataFrame()
    df_train_oversample['text'] = pd.Series(X_resampled.flatten())
    df_train_oversample['label'] = pd.Series(y_resampled)

    return df_train_oversample, df_val, df_test


if __name__ == '__main__':
    percentage = int(sys.argv[1])
    EPOCHS = 20
    LR = 2e-5
    n = 5
    batch_size = 256
    kf = KFold(n_splits=n, shuffle=True)
    configure_logger("my_logger")
    logger = logging.getLogger("my_logger")
    df_neg = pd.read_csv('/scratch2/chiyuwei/political_leaning/data/multiple_features/us-presidential-2020-01_multiple_features_negative_raw.csv')
    df_pos = pd.read_csv('/scratch2/chiyuwei/political_leaning/data/multiple_features/us-presidential-2020-01_multiple_features_positive_raw.csv')

    for positive_or_negative in ['p', 'n']:
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if positive_or_negative == 'p':
            file = 'positive_theta'
            df_all = df_pos
        else:
            file = 'negative_theta'
            df_all = df_neg

        folder_path = f'/scratch2/chiyuwei/political_leaning/classifier_model/multiple_features/with_undersampling/{file}/'
        model_save_path = folder_path + f'{percentage}/'
        report_save_path = folder_path + f'{percentage}/report_all.csv'
        class_save_path = folder_path + f'{percentage}/'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not os.path.exists(class_save_path):
            os.makedirs(class_save_path)

        quantile = df_all['theta'].quantile(float(percentage) / 100)
        label_list = df_all['theta'].apply(lambda x: 1 if x >= quantile else 0)
        df_all['label'] = label_list

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        rp = Report(n)
        logger.info('================ Start Training ================')
        for index, (train_index, test_index) in enumerate(kf.split(df_all)):
            df_train = df_all.iloc[train_index]
            df_test = df_all.iloc[test_index]
            df_train, df_val = train_test_split(df_train, test_size=0.2)

            df_train_oversample, df_val_undersample, df_test_undersample = preprocess_data(df_train, df_val, df_test)

            train_dataset = CustomDataset(df_train_oversample)
            val_dataset = CustomDataset(df_val_undersample)
            test_dataset = CustomDataset(df_test_undersample)

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            validation_dataloader = DataLoader(val_dataset, batch_size=batch_size)
            prediction_dataloader = DataLoader(test_dataset, batch_size=batch_size)

            model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(device)

            trainer = Trainer(model=model,
                              train_dataloader=train_dataloader,
                              validation_dataloader=validation_dataloader,
                              device=device,
                              learning_rate=LR,
                              epochs=EPOCHS,
                              model_save_path=model_save_path + f'model_folder_{index}.pt')
            model_after_train = trainer.train()

            predictor = Predictor(model_state_dict=model_after_train,
                                  dataloader=prediction_dataloader,
                                  device=device)
            report, df_true_pred = predictor.predict()

            rp.get_results(index, report)
            rp.get_class_report(index, report)
            save_pred(index, df_true_pred, class_save_path)
        rp.save_report(report_save_path, class_save_path)

