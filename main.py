import sys

import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from callbacks import MetricsCallback
from dataset import TaggingDataModule
from models import BaselineLSTM


def print_stats(node, tp, tn, fp, fn):
    print(f"Node {node}")
    print(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}")
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    print(f"Accuracy: {acc}, Precision: {prec}, Recall: {rec}")
    return acc, prec, rec


def run_lstm_baseline(datamodule, tensorboard_path="runs", nr_epochs=50):
    metrics = MetricsCallback()
    early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(monitor="train_loss")
    logger = TensorBoardLogger(tensorboard_path)

    model = BaselineLSTM()
    trainer = pl.Trainer(
        deterministic=True,
        accelerator="auto",
        max_epochs=nr_epochs,
        log_every_n_steps=1,
        logger=logger,
        callbacks=[lr_logger, early_stop_callback, metrics, checkpoint_callback],
    )
    trainer.fit(model, datamodule=datamodule)

    checkpoint = torch.load(checkpoint_callback.best_model_path)
    torch.save(checkpoint, "pretrained/checkpoint.ckpt")


def main(filepath):
    data_train = pd.read_csv(filepath)
    data_train["label"].replace(to_replace=["sfa", "sha", "sya", "vna", "dfa"], value="malicious", inplace=True)
    print("Creating dataset...")
    dataset = TaggingDataModule(data_train)
    print("Training LSTM...")
    run_lstm_baseline(dataset, nr_epochs=100)
    print("Done!")


if __name__ == "__main__":
    file = sys.argv[1]
    main(file)
