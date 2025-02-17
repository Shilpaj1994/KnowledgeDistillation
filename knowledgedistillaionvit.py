#!/usr/bin/env python3

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


def teacher_eval():
    """
    Function to run the teacher model
    """
    # Datasets
    dataset = load_dataset("pcuenq/oxford-pets")
    id2label = {int_label: str_label for int_label, str_label in enumerate(sorted(list(set(dataset['train']['label']))))}
    label2id = {str_label: int_label for int_label, str_label in enumerate(sorted(list(set(dataset['train']['label']))))}

    # Teacher Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_model_name = "asusevski/vit-base-patch16-224-oxford-pets"
    teacher_model = AutoModelForImageClassification.from_pretrained(teacher_model_name).to(device)
    teacher_model.eval()
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    accuracy = evaluate.load("accuracy")

    # Model Output
    from PIL import Image
    import io
    import matplotlib.pyplot as plt
    # Decode the image bytes
    image_bytes = dataset["train"][0]["image"]["bytes"]
    image = Image.open(io.BytesIO(image_bytes))  # Convert bytes to a PIL image
    # Process the image
    inputs = processor(image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move tensors to the device
    # Perform inference with the teacher model
    model_logits = teacher_model(**inputs).logits
    model_prediction = torch.argmax(model_logits, dim=1).item()
    # Create a side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Plot the original image
    axes[0].imshow(image)
    axes[0].axis('off')
    axes[0].set_title("Original Image")
    # Plot the image with the prediction
    axes[1].imshow(image)
    axes[1].axis('off')
    axes[1].set_title(f"Prediction: {id2label[model_prediction]}")
    # Display the plots
    plt.tight_layout()
    plt.show()
    print(f"Output: {id2label[model_prediction]}")

    # Print Teacher model parameters
    total_params = sum(p.numel() for p in teacher_model.parameters())
    print(f"Total Teacher Parameters: {total_params:,}")
    print("-" * 50)
    print("Teacher Model Summary:")
    print(teacher_model)



from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
import io

dataset = load_dataset("pcuenq/oxford-pets")
print(dataset, type(dataset))

# Extract unique labels and create a mapping
unique_labels = list(set(example["label"] for example in dataset["train"]))
label_to_id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
id_to_label = {idx: label for label, idx in label_to_id.items()}

# Map string labels to integers in the dataset
def encode_label(example):
    example["label"] = label_to_id[example["label"]]
    return example

dataset = dataset.map(encode_label)

# Create a train-validation split (e.g., 80% train, 20% validation)
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_data = dataset["train"]
val_data = dataset["test"]

# Define the image transformations for ViT input
img_size = 224  # Match the img_size used in ViT
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
])

# Dataset class for preprocessing
class OxfordPetsDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Access the image and label directly
        entry = self.data[idx]
        image_data = entry['image']['bytes']  # Extract image bytes
        label = entry['label']  # Integer label
        
        # Convert bytes to a PIL Image
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Apply the transformations
        image = self.transform(image)
        
        return image, label

# Create PyTorch datasets and dataloaders
batch_size = 32
train_dataset = OxfordPetsDataset(data=train_data, transform=transform)
val_dataset = OxfordPetsDataset(data=val_data, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Verify data loading
for images, labels in train_loader:
    print(f"Batch size: {images.size()}, Labels: {labels.size()}")
    print(f"First batch labels: {labels}")
    break

from transformer import PatchEmbedding, EncoderLayer

class ViT(nn.Module):
    """Creates a Vision Transformer architecture with ViT-Base hyperparameters by default."""
    # 2. Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 img_size:int=224, # Training resolution from Table 3 in ViT paper
                 in_channels:int=3, # Number of channels in input image
                 patch_size:int=16, # Patch size
                 num_transformer_layers:int=6, # Layers from Table 1 for ViT-Base
                 embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 num_heads:int=6, # Heads from Table 1 for ViT-Base
                 attn_dropout:float=0, # Dropout for attention projection
                 mlp_dropout:float=0.1, # Dropout for dense/MLP layers
                 embedding_dropout:float=0.1, # Dropout for patch and position embeddings
                 num_classes:int=1000): # Default for ImageNet but can customize this
        super().__init__() # don't forget the super().__init__()!

        # 3. Make the image size is divisble by the patch size
        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."

        # 4. Calculate number of patches (height * width/patch^2)
        self.num_patches = (img_size * img_size) // patch_size**2

        # 5. Create learnable class embedding (needs to go at front of sequence of patch embeddings)
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)

        # 6. Create learnable position embedding
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim),
                                               requires_grad=True)

        # 7. Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # 8. Create patch embedding layer
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)

        # 9. Create Transformer Encoder blocks (we can stack Transformer Encoder blocks using nn.Sequential())
        # Note: The "*" means "all"
        self.transformer_encoder = nn.Sequential(*[EncoderLayer(n_heads=1, inner_transformer_size=768, inner_ff_size=3072, seq_len=1, dropout=0.1) for _ in range(num_transformer_layers)])

        # 10. Create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    # 11. Create a forward() method
    def forward(self, x):

        # 12. Get batch size
        batch_size = x.shape[0]

        # 13. Create class token embedding and expand it to match the batch size (equation 1)
        class_token = self.class_embedding.expand(batch_size, -1, -1) # "-1" means to infer the dimension (try this line on its own)

        # 14. Create patch embedding (equation 1)
        x = self.patch_embedding(x)

        # 15. Concat class embedding and patch embedding (equation 1)
        x = torch.cat((class_token, x), dim=1)

        # 16. Add position embedding to patch embedding (equation 1)
        x = self.position_embedding + x

        # 17. Run embedding dropout (Appendix B.1)
        x = self.embedding_dropout(x)

        # 18. Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
        x = self.transformer_encoder(x)

        # 19. Put 0 index logit through classifier (equation 4)
        x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index

        return x

print("Loading Model..")
model = ViT(img_size=224, num_classes=37)  # Assuming 37 classes in the dataset
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 5

# Training loop
for epoch in range(num_epochs):
    model.train()
    print(f"Epoch: {epoch}")
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

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

if __name__ == "__main__":
    train_model()
