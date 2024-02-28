from pytorch_lightning import Callback


class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_loss = None

    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        if 'train_loss' in logs:
            self.train_loss = logs['train_loss'].cpu().squeeze()
