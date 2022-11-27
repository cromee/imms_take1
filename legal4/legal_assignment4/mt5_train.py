import argparse

import datasets
import pytorch_lightning as pl
import torch

from transformers import MT5Model, T5Tokenizer, MT5ForConditionalGeneration
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from statute_data import get_data_module
from generative_classifier import get_classifier


def load_mt5_model(name):
    model = MT5ForConditionalGeneration.from_pretrained(name)
    tokenizer = T5Tokenizer.from_pretrained(name)
    return model, tokenizer


def read_dataset(dataset_card="lbox/lbox_open", task="statute_classification", cache_dir="~/.cache/huggingface/datasets"):
    data = datasets.load_dataset(dataset_card, task, cache_dir=cache_dir)
    train_statues = []
    for example in data['train']:
        train_statues += example['statutes']
    print(f"Number of train labels: {len(set(train_statues))}")

    unknown_cnt = 0
    for example in data['validation']:
        if len(set(example['statutes']) - set(train_statues)) > 0:
            unknown_cnt += 1
    print(f"Percent of examples with unknown labels: {unknown_cnt / len(data['validation']) * 100.} %")

    statutes = list(sorted(list(set(train_statues)))) + ['Unknown']
    return data, statutes


def create_trainer(epoch: int, logging: bool):
    n_gpus = torch.cuda.device_count()
    
    import os
    job_name = os.environ['JOB_ID'][:8]
    if logging:
        logger = [
            TensorBoardLogger("/tensorboard", name=job_name),
            CSVLogger("/gpfs-volume/logs", name=job_name)
        ]
    checkpoint_writer = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath="/gpfs-volume/checkpoints",
        filename="legal-assign4-" + job_name + "-{epoch:02d}-{val_loss:.2f}"
    )
    trainer = pl.Trainer(
        max_epochs=epoch,
        gpus=n_gpus,
        logger=logger,
        callbacks=[checkpoint_writer],
        fast_dev_run=not True, 
        limit_train_batches=1.0,
        limit_val_batches=1.0,
    )
    return trainer
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size')
    parser.add_argument('--max_input_len', type=int, default=256, help='Max input length')
    parser.add_argument('--epoch', type=int, default=8, help='Epochs to train')
    parser.add_argument('--lr', type=float, default=0.00003, help='Learning rate')
    parser.add_argument('--logging', action='store_true', help='Flag to save logs')
    parser.add_argument('--mt5', choices=['small', 'base'], default='small', help='MT5 size')
    return parser.parse_args()


def main():
    args = parse_args()
    print('Options:', args.__dict__)
    name = f'google/mt5-{args.mt5}'
    batch_size, max_input_len = args.batch_size, args.max_input_len
    lr = args.lr
    epoch = args.epoch
    log_mode = args.logging

    data, class_names = read_dataset()
    model, tokenizer = load_mt5_model(name)

    data_module = get_data_module(class_names, tokenizer, data, batch_size=batch_size, max_input_len=max_input_len)
    model = get_classifier(class_names, model, tokenizer, learning_rate=lr)

    # trainer
    trainer = create_trainer(epoch, log_mode)

    import time
    start = int(time.time())
    trainer.fit(model, data_module)
    print('Elapsed', int(time.time()) - start, 's')

    
if __name__ == '__main__':
    main()
