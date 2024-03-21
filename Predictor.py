import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import logging
from logger_config import configure_logger
from transformers import RobertaForSequenceClassification


class Predictor:
    def __init__(self, model_state_dict, dataloader, device):
        self.model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
        self.model.load_state_dict(model_state_dict)
        self.model = self.model.to(device)
        self.model.eval()
        self.dataloader = dataloader
        self.device = device
        configure_logger("my_logger")
        self.logger = logging.getLogger("my_logger")

    def predict(self):
        self.logger.info('Predicting labels for {:,} test sentences...'.format(len(self.dataloader.dataset)))

        # Tracking variables
        predictions, true_labels = [], []

        # Predict
        for batch in self.dataloader:
            input_ids = batch[0]['input_ids'].to(self.device)
            attention_mask = batch[0]['attention_mask'].to(self.device)
            labels = batch[1].to(self.device)

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                result = self.model(input_ids,
                                    token_type_ids=None,
                                    attention_mask=attention_mask,
                                    return_dict=True)

            logits = result.logits

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()

            # Store predictions and true labels
            predictions.extend(np.argmax(logits, axis=1).flatten())
            true_labels.extend(label_ids.flatten())

        self.logger.info('    DONE.')
        # Use int list as target names
        report_eva = classification_report(true_labels, predictions, target_names=['0', '1'], output_dict=True)
        data = {'True Labels': true_labels, 'Predictions': predictions}
        df = pd.DataFrame(data)
        return report_eva, df
