from torch import nn
import torch
from transformers import BertModel


class BertClassifier(nn.Module):
    # def __init__(self, dropout=0.5, num_extra_features=9):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        # self.extra_feature_layer = nn.Linear(num_extra_features, num_extra_features)
        # self.linear = nn.Linear(768 + num_extra_features, 1)
        # self.linear = nn.Linear(768, 1)
        self.linear = nn.Linear(768, 2)
        # self.sigmoid = nn.Sigmoid()

        self.relu = nn.ReLU()
    # def forward(self, input_ids, attention_mask, extra_features):
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)

        # extra_features_output = self.extra_feature_layer(extra_features)

        # combined_output = torch.cat((dropout_output, extra_features_output), dim=1)
        # linear_output = self.linear(combined_output)
        linear_output = self.linear(dropout_output)
        # final_output = self.sigmoid(linear_output)
        final_output = self.relu(linear_output)
        return final_output



