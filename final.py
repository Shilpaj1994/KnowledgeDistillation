from datasets import load_dataset, DatasetDict, Features, Image as DatasetsImage, ClassLabel
from transformers import AutoImageProcessor, AutoModelForImageClassification, ViTConfig, ViTForImageClassification
from transformers import ViTImageProcessor, ViTConfig, ViTForImageClassification
from transformers import TrainingArguments, Trainer
import evaluate
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import io
from knowledge_distillation import OxfordPetsDataset, ViT


class OxfordPetsDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def setup(self, stage=None):
        # Load dataset
        dataset = load_dataset("pcuenq/oxford-pets")
        
        # Create label mappings
        unique_labels = list(set(example["label"] for example in dataset["train"]))
        self.label_to_id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        
        # Encode labels
        dataset = dataset.map(lambda x: {"label": self.label_to_id[x["label"]]})
        
        # Split dataset
        split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
        self.train_data = split_dataset["train"]
        self.val_data = split_dataset["test"]

    def train_dataloader(self):
        train_dataset = OxfordPetsDataset(self.train_data, self.transform)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        val_dataset = OxfordPetsDataset(self.val_data, self.transform)
        return DataLoader(val_dataset, batch_size=self.batch_size)

class LitViT(pl.LightningModule):
    def __init__(self, num_classes=37, learning_rate=1e-4):
        super().__init__()
        self.learning_rate = learning_rate
        self.vit = ViT(img_size=224, num_classes=num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.vit(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

# Replace the training loop with Lightning training
def train_model():
    # Initialize data module
    data_module = OxfordPetsDataModule(batch_size=32)
    
    # Initialize model
    model = LitViT(num_classes=37)
    
    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator='auto',  # Automatically detect GPU/CPU
        devices=1,
        logger=TensorBoardLogger('lightning_logs/', name='vit'),
        callbacks=[
            ModelCheckpoint(
                dirpath='checkpoints',
                filename='vit-{epoch:02d}-{val_loss:.2f}',
                monitor='val_loss',
                mode='min'
            )
        ]
    )
    
    # Train the model
    trainer.fit(model, data_module)

train_model()