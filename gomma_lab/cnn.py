import torch
from torchvision import models
from PIL import Image


def classify_image(image_path):
    # Load pretrained ResNet50 model
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.eval()

    # Preprocess image
    transform = weights.transforms()
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    # Predict image
    with torch.no_grad():
        output = model(image_tensor)

    predicted = output.argmax(dim=1).item()
    labels = weights.meta["categories"]

    print("Predicted Label:", labels[predicted])

    return labels[predicted]
