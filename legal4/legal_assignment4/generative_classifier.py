import random

import torch
import pytorch_lightning as pl
import numpy as np

from functools import reduce

from sklearn.metrics import roc_curve, auc


class StatutesClassifier(pl.LightningModule):
    def __init__(self, statute, backbone, tokenizer, learning_rate=0.00001,
               multi_label=False, pos_weight=None, threshold=0.5):
        super().__init__()
        self.statute = statute
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.multi_label = multi_label
        self.pos_weight = pos_weight
        self.threshold = threshold

    def forward(self, batch):
        model_inputs, _ = batch
        outputs = self.backbone(**model_inputs)
        
        loss = outputs.loss
        logits = outputs.logits

        return loss, logits

    def training_step(self, batch, batch_idx):
        loss, logits = self.forward(batch)
        self.log("train_loss", loss)
        return {"loss": loss, "logits": logits}

    def validation_step(self, batch, batch_idx):
        return self._evaluation_step(batch) 

    def validation_epoch_end(self, outputs):
        self._evaluation_epoch_end(outputs)

    def predict_step(self, batch, batch_idx):
        _, logits = self.forward(batch)
        pr_label_ids = logits.argmax(-1)
        prs = [self.statute[int(label_id)] for label_id in pr_label_ids]
        return prs

    def _evaluation_step(self, batch):
        loss, logits = self.forward(batch)
        _, batch_label_ids = batch

        gts = []
        prs = []
        for label_ids in batch_label_ids:
            gts.append([self.statute[label_id] for label_id in label_ids])

        pr_tokens = logits.argmax(-1)
        prs = [self.tokenizer.decode(pr_token, skip_special_tokens=True) for pr_token in pr_tokens]

        return {
            "loss": loss.item(),
            "gts": gts,
            "prs": prs,
        }

    def _evaluation_epoch_end(self, outputs):
        """
        outputs = list of {'loss', 'gts', 'prs'}
        """
        ave_loss = np.mean([x["loss"] for x in outputs])
        gts = reduce(lambda x,y: x + y, [x["gts"] for x in outputs], [])
        prs = reduce(lambda x,y: x + y, [x["prs"] for x in outputs], [])
        acc = self._compute_single_label_metric(gts, prs)

        target_ids = random.sample(range(len(gts)), 5)
        target_ids = [0,1,2,3,4]
        print("="*50)
        print(f"ave_loss: {ave_loss}")
        print(f"ACC: {acc}")
        print("GT" + "-"*40)
        for target_id in target_ids:
            print(f"GT: {gts[target_id]}\t\t\tPR: {prs[target_id]}")
            
        self.log("acc", acc)
        self.log("val_loss", ave_loss)

    def _compute_single_label_metric(self, gts, prs):
        previous_acc = sum([False if pr == len(self.statute) else pr in gt for gt, pr in zip(gts, prs)]) / len(gts)
        return sum([pr in gt for gt, pr in zip(gts, prs)]) / len(gts)

    def configure_optimizers(self):
        grouped_params = [
          {
              "params": list(filter(lambda p: p.requires_grad, self.parameters())),
              "lr": self.learning_rate,
          },
        ]

        optimizer = torch.optim.AdamW(
            grouped_params,
            lr=self.learning_rate, 
        )
        return {"optimizer": optimizer}
    

def get_classifier(statute, model, tokenizer, learning_rate):
    return StatutesClassifier(statute, model, tokenizer, learning_rate)
