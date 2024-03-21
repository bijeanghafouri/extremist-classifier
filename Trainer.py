import time
import datetime
import torch
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import logging
from logger_config import configure_logger
from copy import deepcopy


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


class Trainer:
    def __init__(self, model, train_dataloader, validation_dataloader, device, learning_rate, epochs, model_save_path):
        self.model = model
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=len(self.train_dataloader) * epochs * 0.1,
                                                         num_training_steps=len(self.train_dataloader) * epochs)
        self.device = device
        self.epochs = epochs
        self.model_save_path = model_save_path
        configure_logger("my_logger")
        self.logger = logging.getLogger("my_logger")

    def train(self):
        training_stats = []
        total_t0 = time.time()
        best_model = None
        best_val_loss = float("inf")
        patience = 0
        for epoch_i in range(0, self.epochs):
            self.logger.info("")
            self.logger.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            self.logger.info('Training...')
            t0 = time.time()
            total_train_loss = 0
            self.model.train()

            for step, batch in enumerate(self.train_dataloader):
                if step % 100 == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)
                    self.logger.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.train_dataloader),
                                                                                elapsed))

                b_input_ids = batch[0]["input_ids"].to(self.device)
                b_input_mask = batch[0]["attention_mask"].to(self.device)
                b_labels = batch[1].to(self.device)

                result = self.model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask,
                                    labels=b_labels,
                                    return_dict=True)

                loss = result.loss
                logits = result.logits
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.model.zero_grad()
                self.scheduler.step()

            avg_train_loss = total_train_loss / len(self.train_dataloader)
            training_time = format_time(time.time() - t0)

            self.logger.info("")
            self.logger.info("  Average training loss: {0:.2f}".format(avg_train_loss))
            self.logger.info("  Training epcoh took: {:}".format(training_time))

            self.logger.info("")
            self.logger.info("Running Validation...")
            t0 = time.time()
            self.model.eval()

            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            for batch in self.validation_dataloader:
                b_input_ids = batch[0]["input_ids"].to(self.device)
                b_input_mask = batch[0]["attention_mask"].to(self.device)
                b_labels = batch[1].to(self.device)

                with torch.no_grad():
                    result = self.model(b_input_ids,
                                        token_type_ids=None,
                                        attention_mask=b_input_mask,
                                        labels=b_labels,
                                        return_dict=True)

                loss = result.loss
                logits = result.logits
                total_eval_loss += loss.item()
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                total_eval_accuracy += flat_accuracy(logits, label_ids)

            avg_val_accuracy = total_eval_accuracy / len(self.validation_dataloader)
            self.logger.info("  Accuracy: {0:.2f}".format(avg_val_accuracy))
            avg_val_loss = total_eval_loss / len(self.validation_dataloader)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = deepcopy(self.model.state_dict())
                torch.save(best_model, self.model_save_path)
                patience = 0
            else:
                patience += 1
            if patience > 3:
                self.logger.info("Early stopping")
                break
            validation_time = format_time(time.time() - t0)

            self.logger.info("  Validation Loss: {0:.2f}".format(avg_val_loss))
            self.logger.info("  Validation took: {:}".format(validation_time))

            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

        self.logger.info("")
        self.logger.info("Training complete!")
        self.logger.info("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

        return best_model
