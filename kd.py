import torch
import torch.nn.functional as F
from final import OxfordPetsDataModule, LitViT
import pytorch_lightning as pl
from transformers import AutoModelForImageClassification, ViTImageProcessor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

class DistillationModel(pl.LightningModule):
    def __init__(self, num_classes=37, learning_rate=1e-4, alpha=0.5, temperature=3.0):
        super().__init__()
        # Student model
        self.student = LitViT(num_classes=num_classes)
        
        # Teacher model
        teacher_model_name = "asusevski/vit-base-patch16-224-oxford-pets"
        self.teacher = AutoModelForImageClassification.from_pretrained(teacher_model_name)
        self.teacher.eval()  # Set teacher to evaluation mode
        
        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # Standard ViT preprocessing parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.temperature = temperature
        
    def forward(self, x):
        return self.student(x)
    
    def _preprocess_images_for_teacher(self, images):
        # Normalize to [0, 1] range
        images = (images + 1) / 2.0
        # Apply ImageNet normalization
        images = (images - self.mean.to(images.device)) / self.std.to(images.device)
        return images
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        
        # Preprocess images for teacher
        processed_images = self._preprocess_images_for_teacher(images)
        
        # Get teacher predictions
        with torch.no_grad():
            teacher_logits = self.teacher(processed_images).logits
        
        # Get student predictions
        student_logits = self.student(images)
        
        # Calculate soft targets (distillation loss)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        student_logits_temp = student_logits / self.temperature
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits_temp, dim=1),
            soft_targets,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Calculate hard targets (standard cross-entropy loss)
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Combine losses
        loss = (self.alpha * distillation_loss) + ((1 - self.alpha) * hard_loss)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('distillation_loss', distillation_loss, prog_bar=True)
        self.log('hard_loss', hard_loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.student(images)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2, 
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

def train_with_distillation():
    # Initialize data module
    data_module = OxfordPetsDataModule(batch_size=32)
    
    # Initialize distillation model
    model = DistillationModel(
        num_classes=37,
        learning_rate=1e-4,  # Lower learning rate
        alpha=0.7,  # More weight to soft targets
        temperature=2.0  # Lower temperature for clearer signal
    )
    
    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator='auto',
        devices=1,
        logger=TensorBoardLogger('lightning_logs/', name='distillation'),
        callbacks=[
            ModelCheckpoint(
                dirpath='distillation_checkpoints',
                filename='distill-{epoch:02d}-{val_loss:.2f}',
                monitor='val_loss',
                mode='min'
            )
        ],
        gradient_clip_val=1.0,
    )
    
    # Train the model
    trainer.fit(model, data_module)

if __name__ == "__main__":
    train_with_distillation()
