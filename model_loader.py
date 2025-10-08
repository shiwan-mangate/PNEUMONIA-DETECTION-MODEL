import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

# ----------------------------
# 1️⃣ ResNet Class
# ----------------------------
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2, resnet_version=50, dropout_rate=0.5, trainable_layers=1, input_channels=1):
        super().__init__()

        # Load pre-trained ResNet
        if resnet_version == 18:
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif resnet_version == 34:
            self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif resnet_version == 50:
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif resnet_version == 101:
            self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        elif resnet_version == 152:
            self.model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        else:
            raise ValueError("Unsupported ResNet version!")

        # Adjust input channels
        if input_channels != 3:
            old_conv = self.model.conv1
            self.model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if input_channels == 1:
                self.model.conv1.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
            else:
                self.model.conv1.weight.data = old_conv.weight.data[:, :input_channels, :, :]

        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze last layers
        layer_list = [self.model.layer4, self.model.layer3, self.model.layer2, self.model.layer1]
        for i in range(trainable_layers):
            for param in layer_list[i].parameters():
                param.requires_grad = True

        # Replace classifier
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# ----------------------------
# 2️⃣ Load trained model
# ----------------------------
@torch.no_grad()
def load_model(model_path =r"D:\Machine learning\deep learning\pneumonia app\best_pneumonia_resnet2.pth" , device=None):
    """
    Load trained ResNet model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResNetClassifier(
        num_classes=2,
        resnet_version=50,
        dropout_rate=0.532,
        trainable_layers=2,
        input_channels=1
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# ----------------------------
# 3️⃣ Predict function
# ----------------------------
@torch.no_grad()
def predict_image(model, device, image, class_names=["Normal", "Pneumonia"]):
    """
    Predict class and confidence for a PIL image.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    img_tensor = transform(image).unsqueeze(0).to(device)
    output = model(img_tensor)
    probs = torch.softmax(output, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_idx].item()
    pred_class = class_names[pred_idx]

    return pred_class, confidence
