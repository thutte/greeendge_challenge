import torch

import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl


class BaselineLSTM(pl.LightningModule):
    def __init__(self, num_features=16, h_s=64, num_layers=1, l_r=1e-3):
        super().__init__()
        self.hidden_size = h_s
        self.num_features = num_features
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=num_features, hidden_size=h_s, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=h_s, out_features=2)

        self.learning_rate = l_r
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        b_s = x.shape[0]
        hidden = (torch.zeros((self.num_layers, b_s, self.hidden_size)).type_as(x), torch.zeros((self.num_layers, b_s, self.hidden_size)).type_as(x))

        output, _ = self.lstm(x, hidden)
        output = self.fc(output)

        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_model = self.forward(x)
        y_f = y.flatten().long()
        y_model_f = torch.flatten(y_model, end_dim=1)
        loss = self.criterion(y_model_f, y_f)
        self.log("train_loss", loss, on_epoch=True, batch_size=512, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_model = self.forward(x)
        y_f = y.flatten().long()
        y_model_f = torch.flatten(y_model, end_dim=1)
        loss = self.criterion(y_model_f, y_f)
        self.log("test_loss", loss, on_epoch=True, batch_size=512, prog_bar=True)
        y_pred_f = y_model_f.argmax(dim=-1)
        accuracy = torch.eq(y_pred_f, y_f).to(float).mean()
        self.log("test_accuracy", accuracy, on_epoch=True, batch_size=512, prog_bar=True)
        return

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss"}}
