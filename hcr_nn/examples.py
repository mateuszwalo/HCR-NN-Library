from hcr_layers import InformationBottleneckLoss, DynamicEMA
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.ema = DynamicEMA(0.6)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.ema(x)
        x = torch.relu(self.linear2(x))
        return self.linear3(x)

#Information Bottleneck criterion example usage
class SmallIBNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.bottleneck = nn.Linear(hidden_dim, hidden_dim // 2)
        self.decoder = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        x_enc = F.relu(self.encoder(x))
        t = F.relu(self.bottleneck(x_enc))
        y_pred = self.decoder(t)
        return x, t, y_pred

def IB_loss_example():
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_dim = 10
    hidden_dim = 16
    output_dim = 5

    model = SmallIBNet(input_dim, hidden_dim, output_dim).to(device)
    ib_loss = InformationBottleneckLoss(beta=1.0).to(device)
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X = torch.randn(64, input_dim, device=device)
    Y_true = torch.randn(64, output_dim, device=device)

    for epoch in range(10):
        X_in, T, Y_pred = model(X)        
        reconstruction_loss = mse_loss(Y_pred, Y_true)
        information_loss = ib_loss(X_in, T, Y_true)
        total_loss = reconstruction_loss + 0.01 * information_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1:02d} | Total Loss: {total_loss.item():.4f} | "
            f"Recon: {reconstruction_loss.item():.4f} | IB: {information_loss.item():.4f}")

def dema_example():
    model = DifferentiableModel(input_dim=10, hidden_dim=20, output_dim=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()

    for step in range(200):
        inputs = torch.randn(16, 10)
        targets = torch.randn(16, 3)

        preds = model(inputs)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            print(f"Step {step}, loss: {loss.item():.4f}")

#dema_example()