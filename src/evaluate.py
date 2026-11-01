import torch
from sklearn.metrics import classification_report, confusion_matrix
from dataset import get_dataloaders
from model import get_model

def evaluate():
    DATA_DIR = "../data/processed/sdnet_split"
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the test loader (Pavements only)
    _, _, test_loader = get_dataloaders(DATA_DIR, BATCH_SIZE)
    
    # Load the model
    model = get_model().to(DEVICE)
    model.load_state_dict(torch.load("../models/baseline_resnet18_seed42.pth"))
    model.eval()

    all_preds = []
    all_labels = []

    print("Running inference on Pavement (Test) set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n" + "="*40)
    print("PAVEMENT HOLDOUT EVALUATION RESULTS")
    print("="*40)
    
    # ImageFolder assigns 0 to the first alphabetical folder ('cracked') and 1 to 'uncracked'
    print(classification_report(all_labels, all_preds, target_names=["cracked", "uncracked"]))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    evaluate()