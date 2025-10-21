import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Example Custom Dataset
# -------------------------
class CustomDataset(Dataset):
    def __init__(self, data_path):
        # Load your data here
        self.data = torch.randn(100, 10)  # Dummy
        self.labels = torch.randn(100, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# -------------------------
# Example Architectures
# -------------------------
class NetA(nn.Module):
    def __init__(self, input_size=10, hidden_size=32, output_size=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)


class NetB(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, output_size=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)


# -------------------------
# Config Loader/Saver
# -------------------------
class Config:
    def __init__(self, config_path=None):
        if config_path:
            with open(config_path, "r") as f:
                self.cfg = yaml.safe_load(f)
        else:
            self.cfg = {}

    def save(self, path):
        with open(path, "w") as f:
            yaml.dump(self.cfg, f)

    def build_model(self):
        arch = self.cfg["model"]["arch"]
        if arch == "NetA":
            return NetA()
        elif arch == "NetB":
            return NetB()
        else:
            raise ValueError(f"Unknown model architecture {arch}")

    def build_dataset(self):
        return CustomDataset(self.cfg["data"]["path"])

    def build_optimizer(self, model):
        lr = self.cfg["training"]["learning_rate"]
        opt_type = self.cfg["training"]["optimizer"]
        if opt_type == "Adam":
            return optim.Adam(model.parameters(), lr=lr)
        elif opt_type == "AdamW":
            return optim.AdamW(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer {opt_type}")

    def build_loss(self):
        loss_type = self.cfg["training"]["loss"]
        if loss_type == "MSE":
            return nn.MSELoss()
        elif loss_type == "ME":
            return lambda pred, target: torch.mean(torch.abs(pred - target))  # simple MAE
        else:
            raise ValueError(f"Unknown loss {loss_type}")

    def build_scheduler(self, optimizer):
        if self.cfg["training"].get("scheduler", False):
            return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return None


# -------------------------
# Example YAML Config
# -------------------------
default_config = {
    "model": {
        "arch": "NetA"  # Options: NetA, NetB
    },
    "data": {
        "path": "dataset.pt"
    },
    "training": {
        "optimizer": "Adam",   # Options: Adam, AdamW
        "loss": "MSE",         # Options: MSE, ME
        "learning_rate": 0.001,
        "scheduler": True
    }
}

if __name__ == "__main__":
    # Save default config
    cfg = Config()
    cfg.cfg = default_config
    cfg.save("config.yaml")

    # Load and build objects
    cfg = Config("config.yaml")
    model = cfg.build_model()
    dataset = cfg.build_dataset()
    optimizer = cfg.build_optimizer(model)
    loss_fn = cfg.build_loss()
    scheduler = cfg.build_scheduler(optimizer)

    print(model)
    print(loss_fn)
    print(optimizer)
    print(scheduler)
