from hcr_layers import InformationBottleneckLoss, DynamicEMA, EntropyAndMutualInformation, PropagationEstimation, MeanEstimationBaseline
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class FeatureNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 32),
            nn.ReLU(),
            nn.Linear(32, d)
        )

    def forward(self, x):
        return self.net(x)

class FeatureNet2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class PropModel(nn.Module):
    def __init__(self, y, z, a, d):
        super().__init__()
        self.feature_net = FeatureNet(d)

        self.propagation = PropagationEstimation(
            y=y,
            z=z,
            a=a,
            feature_fn=self.feature_net
        )

        self.head = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self):
        p = self.propagation()
        p = p.view(1, 1)
        return self.head(p)

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
        inputs = torch.randn(160, 10)
        targets = torch.randn(160, 3)

        preds = model(inputs)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            print(f"Step {step}, loss: {loss.item():.4f}")

def ent_mi_example():
    class InfoRegularizedModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, mi_weight=0.1):
            super().__init__()
            self.encoder = nn.Linear(input_dim, hidden_dim)
            self.decoder = nn.Linear(hidden_dim, output_dim)
            self.info = EntropyAndMutualInformation()
            self.mi = EntropyAndMutualInformation(compute_mi=True)
            self.mi_weight = mi_weight

        def forward(self, x):
            hidden = torch.relu(self.encoder(x))
            output = self.decoder(hidden)
            entropy_loss = self.info(hidden)
            # For MI, we need a paired input; here we just reuse for demonstration
            mutual_info_loss = self.mi(hidden, output)
            total_loss = output.mean() - entropy_loss + self.mi_weight * mutual_info_loss
            return output, entropy_loss, mutual_info_loss, total_loss

    model = InfoRegularizedModel(input_dim=10, hidden_dim=32, output_dim=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for step in range(100):
        x = torch.randn(16, 10)
        y = torch.randn(16, 3)
        out, entropy, mi, loss_info = model(x)
        mse_loss = criterion(out, y)
        total_loss = mse_loss + 0.1 * entropy - 0.05 * mi

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step} | Loss: {total_loss.item():.4f} | H: {entropy.item():.4f} | MI: {mi.item():.4f}")

def propagation_example():
    d = 8
    y = torch.randn(d)
    z = torch.randn(d)
    a = torch.randn(2, d, d)

    target = torch.tensor([[1.0]])

    model = PropModel(y, z, a, d)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for step in range(500):
        optimizer.zero_grad()

        pred = model()
        loss = loss_fn(pred, target)

        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step}: loss={loss.item():.4f}")

def meanest():
    B = 128
    input_dim = 10
    feature_dim = 6

    x = torch.randn(B, input_dim)
    y = torch.randn(B, input_dim)
    z = torch.randn(B, input_dim)

    target = torch.randn(feature_dim, feature_dim, feature_dim)

    feature_fn = FeatureNet2(input_dim, feature_dim)
    model = MeanEstimationBaseline(feature_fn=feature_fn, feature_dim=feature_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for step in range(300):
        optimizer.zero_grad()

        A = model(x, y, z)
        loss = criterion(A, target)

        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print("step", step, "loss", loss.item())

meanest()