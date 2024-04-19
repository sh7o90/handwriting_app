import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torchmetrics.functional import accuracy
import timm
import prepare_dataset
import pickle

# EfficientNetV2_EMNISTモデルの定義
class EfficientNetV2_EMNIST(pl.LightningModule):
    def __init__(self, num_classes=47):  # クラス数を47に変更
        super().__init__()
        self.efficientnet = timm.create_model('tf_efficientnetv2_b0', pretrained=True)
        # 最終層の出力数を変更
        in_features = self.efficientnet.classifier.in_features
        self.efficientnet.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)

    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy(y.softmax(dim=-1), t, task='multiclass', num_classes=47, top_k=1), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', accuracy(y.softmax(dim=-1), t, task='multiclass', num_classes=47, top_k=1), on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', accuracy(y.softmax(dim=-1), t, task='multiclass', num_classes=47, top_k=1), on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath="logs/my_exp",
    filename="efficientnetv2_emnist_{epoch:02d}-{val_loss:.2f}",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
)

# 学習の実行
pl.seed_everything(0)
net = EfficientNetV2_EMNIST()
logger = CSVLogger(save_dir='logs', name='my_exp')
# trainer = pl.Trainer(max_epochs=3, accelerator="gpu", deterministic=False, logger=logger)
trainer = pl.Trainer(
    max_epochs=3,
    accelerator="gpu",
    deterministic=False,
    logger=logger,
    callbacks=[checkpoint_callback]
)
# trainer.fit(net, iter(prepare_dataset.train_loader), iter(prepare_dataset.val_loader))
trainer.fit(net, prepare_dataset.train_loader_transposed, prepare_dataset.val_loader_transposed)
trainer.test(dataloaders=prepare_dataset.test_loader_transposed)

# 学習済みモデルの保存
with open("efficientnetv2_emnist.pkl", "wb") as f:
    pickle.dump(net.state_dict(), f)

print("Model saved to efficientnetv2_emnist.pkl")
