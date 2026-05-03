import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from pathlib import Path

# ==========================================
# 1. DATASET DEFINITION
# ==========================================
class CASIABDataset(Dataset):
    """
    Custom PyTorch Dataset for loading CASIA-B Gait Energy Images.
    Expects directory structure: root_dir / subject_id / condition / angle.png
    """
    def __init__(self, data_dir, subject_list, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        # Map original subject IDs (e.g. '001') to integer labels (0 to num_subjects-1)
        # This is necessary because CrossEntropyLoss expects labels in range [0, C-1]
        self.subject_to_idx = {subj: idx for idx, subj in enumerate(sorted(subject_list))}
        
        for subj in subject_list:
            subj_dir = self.data_dir / subj
            if not subj_dir.exists():
                print(f"Warning: Folder for subject {subj} not found at {subj_dir}")
                continue
                
            # Iterate through conditions (nm-01, nm-02, bg-01, etc.)
            for condition_dir in subj_dir.iterdir():
                if not condition_dir.is_dir(): continue
                
                # Iterate through all PNG images (angles) in the condition folder
                for img_path in condition_dir.glob('*.png'):
                    self.samples.append((img_path, self.subject_to_idx[subj]))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image as Grayscale (1 channel)
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ==========================================
# 2. DATA LOADERS & TRANSFORMATIONS
# ==========================================
def get_dataloaders(data_dir, batch_size=64):
    """
    Creates DataLoaders for Training (first 74 subjects) and Testing (remaining 50 subjects).
    """
    # Standard CASIA-B Split: 001-074 for Train, 075-124 for Test
    train_subjects = [f"{i:03d}" for i in range(1, 75)]
    test_subjects = [f"{i:03d}" for i in range(75, 125)]
    
    # Define Transformations
    # GEIs are 1-channel images. We resize them to 64x64 to keep the model lightweight.
    # Normalization mean=0.5, std=0.5 brings pixel values to [-1, 1] range.
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])
    
    # Create Dataset Objects
    train_dataset = CASIABDataset(data_dir, train_subjects, transform=transform)
    test_dataset = CASIABDataset(data_dir, test_subjects, transform=transform)
    
    # Create DataLoaders
    # num_workers=4 is usually a good balance for disk reading speed vs CPU overhead
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # shuffle=False is standard for test sets
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader, len(train_subjects)

# ==========================================
# 3. BASELINE CNN MODEL
# ==========================================
class BaselineGaitCNN(nn.Module):
    """
    ResNet18 Architecture modified for 1-channel Gait Energy Images.
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # Load a base ResNet18 (without pretrained weights since we are training from scratch on GEIs)
        self.resnet = models.resnet18(weights=None)
        
        # Modify the first convolutional layer to accept 1 channel (grayscale) instead of 3 (RGB)
        # ResNet18's original conv1: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Store the original fully connected layer out_features (512 for ResNet18)
        num_ftrs = self.resnet.fc.in_features
        
        # Replace the final fully connected layer with an Identity layer
        # This makes self.resnet(x) output the 512-dimensional embedding directly
        self.resnet.fc = nn.Identity()
        
        # Add our own separate classification layer on top
        self.classifier = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x, return_embedding=False):
        # Extract features (the 512-d embedding)
        emb = self.resnet(x)
        
        # Get class logits
        logits = self.classifier(emb)
        
        if return_embedding:
            return logits, emb
            
        return logits

# ==========================================
# 4. SCRIPT VERIFICATION (TEST BLOCK)
# ==========================================
if __name__ == "__main__":
    # Define parameters
    GEI_DATA_DIR = "./dataset/GEI_Data"
    BATCH_SIZE = 64
    
    print("Initializing DataLoaders...")
    train_loader, test_loader, num_train_classes = get_dataloaders(GEI_DATA_DIR, batch_size=BATCH_SIZE)
    
    print(f"Train set size: {len(train_loader.dataset)} images")
    print(f"Test set size: {len(test_loader.dataset)} images")
    print(f"Number of training classes: {num_train_classes}")
    
    # Fetch one batch to verify
    try:
        images, labels = next(iter(train_loader))
        print(f"\nBatch images shape: {images.shape} (Batch, Channel, Height, Width)")
        print(f"Batch labels shape: {labels.shape}")
        
        # Initialize Model
        print("\nInitializing BaselineGaitCNN...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        model = BaselineGaitCNN(num_classes=num_train_classes).to(device)
        images = images.to(device)
        
        # Test a forward pass
        logits, embeddings = model(images, return_embedding=True)
        print(f"Model logits shape: {logits.shape} -> Expected: (Batch Size, {num_train_classes})")
        print(f"Model embeddings shape: {embeddings.shape} -> Expected: (Batch Size, 512)")
        print("\n✅ Script verification successful! Dataset and Model are ready.")
        
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        print("Make sure your GEI_Data directory is properly populated.")
