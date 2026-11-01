import torch
import torch.nn as nn
import torch.optim as optim
import os
from dataset import get_dataloaders
from model import get_model

def train():
    DATA_DIR = "../data/processed/sdnet_split"
    BATCH_SIZE = 32
    EPOCHS = 5 # Start with 5 to see how it goes
    LR = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {DEVICE}")

    train_loader, val_loader, _ = get_dataloaders(DATA_DIR, BATCH_SIZE)
    
    model = get_model().to(DEVICE)
    
    # SDNET is imbalanced. This weights the loss so the model doesn't just guess "uncracked"
    # Adjust these numbers if your dataset ratio is different
    class_weights = torch.tensor([3.0, 1.0]).to(DEVICE) 
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Only optimize the parameters of the final layer (since we froze the rest)
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 50 == 0: # Print every 50 batches
                print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        print(f"--- Epoch {epoch+1} Summary: Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}% ---")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("../models", exist_ok=True)
            torch.save(model.state_dict(), "../models/baseline_resnet18_seed42.pth")
            print("--> Saved best model to models/")

if __name__ == "__main__":
    train()