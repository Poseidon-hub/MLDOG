import torch
from PIL import Image
from torchvision.models import ResNet18_Weights

from model import build_model, get_device


def main():
    device = get_device()
    print("device:", device)

    checkpoint = torch.load("dog_breeds.pth", map_location=device)
    classes = checkpoint["classes"]
    transform = ResNet18_Weights.DEFAULT.transforms()

    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    print(model)
    print("Основные части ResNet18:")
    for name in ("conv1", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"):
        print(f"{name}: {type(getattr(model, name)).__name__}")

    image = Image.open("dog.jpg").convert("RGB")
    image_tensor = transform(image)
    print(image_tensor.shape)
    image_tensor = image_tensor.unsqueeze(0)
    print(image_tensor.shape)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        class_id = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, class_id].item()

    print(classes[class_id], f"{confidence:.1%}")


if __name__ == "__main__":
    main()
