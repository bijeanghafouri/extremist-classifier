import torch
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)


def clean_text(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_len=256):
        self.labels = [int(label) for label in df['label']]
        self.texts = df['text'].tolist()
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sent = self.texts[idx]
        sent = clean_text(sent)
        encoded_dict = tokenizer.encode_plus(
                    sent,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    truncation=True,
                    padding='max_length',
                    return_attention_mask=True,
                    return_tensors='pt',
        )
        input_id = encoded_dict['input_ids'].squeeze()
        attention_mask = encoded_dict['attention_mask'].squeeze()
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        return {"input_ids": input_id, "attention_mask": attention_mask}, label


