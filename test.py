import torch
from torchvision import transforms
from PIL import Image
import os
# Importing the NeuralNetwork class from your training script
from main import NeuralNetwork

def load_image(image_path):
    # Convert the image to grayscale, resize it to 28x28 and convert it to a tensor
    transformation = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path)
    image_tensor = transformation(image)
    return image_tensor.unsqueeze(0)  # Add a batch dimension


def predict(model, image_path, device):
    image_tensor = load_image(image_path).to(device)
    with torch.no_grad():
        output = model(image_tensor)
    return output.argmax(1).item()  # Get the predicted class

# Load the trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNetwork().to(device)
if os.path.exists("model.pth"):
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()
    print("Model loaded successfully.")
else:
    print("Model file not found!")

# Test with custom images
image_paths = ["1.png", "7.png"]
for image_path in image_paths:
    if os.path.exists(image_path):
        prediction = predict(model, image_path, device)
        print(f"Predicted digit for {image_path}: {prediction}")
    else:
        print(f"Image file {image_path} not found!")
