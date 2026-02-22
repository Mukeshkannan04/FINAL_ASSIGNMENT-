import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from dataset import get_dataloaders
from model import AirDrawModel

def evaluate():
    # 1. Setup
    device = torch.device("cpu") # CPU is fine for testing
    print("Loading test data...")
    # Batch size 1 so we can see individual predictions
    _, test_loader = get_dataloaders('data/processed', batch_size=1) 
    
    # 2. Load the Trained Model
    model = AirDrawModel()
    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        print("Loaded 'best_model.pth' successfully.")
    except FileNotFoundError:
        print("Error: 'best_model.pth' not found. Did you run train.py?")
        return

    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("\n--- Sample Predictions ---")
    
    # 3. Run Inference
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            pred_val = predicted.item()
            true_val = labels.item()
            
            all_preds.append(pred_val)
            all_labels.append(true_val)
            
            # Print the first 10 examples to the screen
            if i < 10:
                status = "✅" if pred_val == true_val else "❌"
                print(f"Sample {i+1}: True Digit [{true_val}] -> Predicted [{pred_val}] {status}")

    # 4. Generate Report
    print("\n--- Classification Report ---")
    # This shows Precision/Recall for every digit
    print(classification_report(all_labels, all_preds, zero_division=0))
    
    # 5. Plot Confusion Matrix
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('AirDraw Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Saved chart to 'confusion_matrix.png'. Open it to see errors!")

if __name__ == "__main__":
    evaluate()