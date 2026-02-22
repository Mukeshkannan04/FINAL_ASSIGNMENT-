import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import AirDrawModel

def train():
    # 1. Load Data
    print("Loading data...")
    train_loader, test_loader = get_dataloaders('data/processed', batch_size=8)
    
    # 2. Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    model = AirDrawModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 30 # Increased to 30 for better convergence
    best_acc = 0.0

    # 3. Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 4. Evaluation Loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Accuracy: {acc:.1f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model.pth")

    print(f"\nTraining Complete! Best Accuracy: {best_acc:.1f}%")
    print("Model saved as 'best_model.pth'")

if __name__ == "__main__":
    train()