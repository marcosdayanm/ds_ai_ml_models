import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

# Verificar si hay soporte MPS (GPU Apple)
device_cpu = torch.device("cpu")
device_mps = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Datos sintéticos
x = torch.randn(5000, 1, 28, 28)
y = torch.randint(0, 10, (5000,))
dataset = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)

# CNN sencilla
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 3, 1)
        self.fc = nn.Linear(16*26*26, 10)
    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def train(device):
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(2):
        for xb, yb in dataset:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    return time.time() - start

print(f"Tiempo CPU: {train(device_cpu):.2f} s")
print(f"Tiempo MPS: {train(device_mps):.2f} s")
