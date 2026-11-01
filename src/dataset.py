import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=32):
    # Standard ImageNet normalization (required for pretrained ResNet)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Training transforms (adds random flips to prevent overfitting)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    # Validation/Test transforms (no flips, just resize)
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # Load datasets from your processed folder
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=val_transforms)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=val_transforms)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader