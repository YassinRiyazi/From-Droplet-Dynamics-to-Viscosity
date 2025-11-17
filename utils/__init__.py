import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

def set_randomness(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
