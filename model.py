import torch
import torch.nn as nn

class AirDrawModel(nn.Module):
    def __init__(self, num_classes=10):
        super(AirDrawModel, self).__init__()
        
        # CNN: Extracts features from the sensor curves
        self.cnn = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # LSTM: Understands the sequence of movement
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        
        # Classifier: Outputs the digit (0-9)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input shape: [Batch, 6, 200]
        x = self.cnn(x)              # Output: [Batch, 64, 50]
        x = x.transpose(1, 2)        # Swap for LSTM: [Batch, 50, 64]
        out, _ = self.lstm(x)
        # Use the output of the last time step
        return self.fc(out[:, -1, :])