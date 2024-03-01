import sys
import pandas as pd
import torch
import pytorch_lightning as pl
from dataset import TaggingDataModule
from models import BaselineLSTM


def main(path_to_checkpoint_file):
    # load data -- to be changed with path to test data
    data = pd.read_csv("data/intrusion_big_train.csv")

    # remove extra data (according to seq_len = 50) -- only for simplicity
    extra_data = data.shape[0] % 50
    data = data[:-extra_data]

    # replace bad labels
    data["label"].replace(to_replace=["sfa", "sha", "sya", "vna", "dfa"], value="malicious", inplace=True)

    # create datasets
    dataset = TaggingDataModule(data)
    dataset.setup()
    dataset.data_train.len += 1
    dataloader = dataset.train_dataloader()

    # load pretrained model
    model = BaselineLSTM()
    model.load_from_checkpoint(path_to_checkpoint_file)

    # perform inference and complete data
    predictions = []
    for idx, batch in enumerate(dataloader):
        print(f"Test batch: {idx+1}", end="\r")
        x, y = batch
        y_model = model.forward(x)
        y_model_f = torch.flatten(y_model, end_dim=1)
        y_pred_f = y_model_f.argmax(dim=-1)
        predictions.append(y_pred_f)

    predictions = torch.cat(predictions, dim=0).numpy()

    data_pred = data.copy()
    data_pred["label"] = predictions
    data_pred["label"].replace([0, 1], ["benign", "malicious"], inplace=True)
    data_pred.to_csv("data/intrusion_big_train_pred.csv")


if __name__ == "__main__":
    path_to_checkpoint_file = sys.argv[1]
    main(path_to_checkpoint_file)
