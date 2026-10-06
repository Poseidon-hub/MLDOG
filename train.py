import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

from dataset import SELECTED_BREEDS, load_datasets
from model import build_model, get_device


def main():
    device = get_device()
    print("device:", device)

    transform = ResNet18_Weights.DEFAULT.transforms()
    train_ds, test_ds = load_datasets("./data", transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    images, labels = next(iter(train_loader))
    print(images.shape)
    print(labels.shape)

    model = build_model(len(SELECTED_BREEDS), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(5):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        print(f"epoch={epoch + 1} loss={loss.item():.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    print(f"accuracy = {correct / total:.2%}")
    torch.save(
        {"model_state": model.state_dict(), "classes": SELECTED_BREEDS},
        "dog_breeds.pth",
    )


if __name__ == "__main__":
    main()
