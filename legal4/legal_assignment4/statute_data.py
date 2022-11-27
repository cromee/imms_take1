import pytorch_lightning as pl
import torch


class StatutesDataModule(pl.LightningDataModule):
    def __init__(self, statute, tokenizer, data, batch_size, max_input_len):
        super().__init__()
        self.statute = statute
        self.tokenizer = tokenizer
        self.data = data
        self.batch_size = batch_size
        self.max_input_len = max_input_len
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=self._single_label_collate_fn
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data["validation"],
            batch_size=self.batch_size * 2,
            shuffle=False,
            collate_fn=self._single_label_collate_fn
        )
    
    def _single_label_collate_fn(self, batch):
        input_texts = [x["facts"] for x in batch]
        label_texts = [x["statutes"][0] for x in batch]
        
        model_inputs = self.tokenizer(
            input_texts,
            text_target=label_texts,
            max_length=self.max_input_len,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        batch_label_ids = []
        for x in batch:
            batch_label_ids.append([self.statute.index(s) if s in self.statute else len(self.statute) - 1 for s in x['statutes']])
            
        return model_inputs, batch_label_ids
    
   
def get_data_module(statute, tokenizer, data, batch_size, max_input_len):
    return StatutesDataModule(statute, tokenizer, data, batch_size, max_input_len)
