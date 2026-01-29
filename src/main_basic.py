import mlbrs
import yaml
import argparse
import pathlib
from torch.utils.data import DataLoader
import torch
from pathlib import Path
from datetime import datetime
import json

from typing import Any
from tqdm import tqdm


def load_config(config_path: pathlib.Path) -> dict[str, Any]:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def train_model(config, train_dataset, model):
    # run training
    device = config["training"].get("device", "cpu")
    epochs = config["training"].get("epochs", 10)
    batch_size = config["training"].get("batch_size", 32)
    learning_rate = config["training"].get("learning_rate", 0.001)

    # train model
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    training_loss = []
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.4f}")
        training_loss.append(total_loss / len(loader))
    return training_loss, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MLbRS with specified config.")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    # load config
    config = load_config(args.config)

    # Initialize model and torchdataset from config
    if "model" in config:
        model = mlbrs.model.Model.from_config(config["model"])
        print("Model initialized:", model)
    else:
        raise ValueError("Model configuration not found in the config file.")

    if "training_dataset" in config:
        training_dataset = mlbrs.dataset.TorchDataset.from_config(config["training_dataset"])
        print("TorchDataset initialized with length:", len(training_dataset))
    else:
        raise ValueError("TorchDataset configuration not found in the config file.")

    if "test_dataset" in config:
        test_dataset = mlbrs.dataset.TorchDataset.from_config(config["test_dataset"])
        print("Test TorchDataset initialized with length:", len(test_dataset))
    else:
        raise ValueError("Test TorchDataset configuration not found in the config file.")

    if "training" not in config:
        raise ValueError("Training configuration not found in the config file.")

    # set up paths
    outpath = config.get("output_path", "./results")
    outpath = Path(outpath)
    experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    full_outpath = outpath / experiment_name
    full_outpath.mkdir(parents=True, exist_ok=True)

    with open(full_outpath / "config_used.yaml", "w") as f:
        yaml.dump(config, f)

    # train model
    training_loss, model = train_model(config, training_dataset, model)
    f1 = mlbrs.evaluation.evaluate_f1_score(
        model,
        test_dataset,
        batch_size=config["training"].get("batch_size", 32),
        device=config["training"].get("device", "cpu"),
    )
    print(f"F1 Score on test set: {f1:.4f}")
    # Save the trained model
    model_path = full_outpath / "trained_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved at: {model_path}")

    # Save results to JSON
    results = {
        "training_loss": training_loss,
        "f1_score": f1,
        "epochs": config["training"].get("epochs", 10),
        "batch_size": config["training"].get("batch_size", 32),
        "learning_rate": config["training"].get("learning_rate", 0.001),
    }

    json.dump(results, open(full_outpath / "results.json", "w"), indent=4)
