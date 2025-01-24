from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import transforms

from lenet.config import get_config
from lenet.model import LeNet
from lenet.utils import get_model_checkpoint, get_device, get_project_root_dir

"""
Perform inference on an input image using the trained model. 
Returns the predicted class and probability output.
"""
def inference(image_input_tensor, device):
    device = torch.device(device)
    model = LeNet().to(device)
    # Load the checkpoint
    model_checkpoint = get_model_checkpoint(get_config(), 4)
    checkpoint = torch.load(model_checkpoint, map_location=device, weights_only=True)
    # Load the model state_dict
    model.load_state_dict(checkpoint["model_state_dict"])

    # Set model to evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():  # No gradients needed during inference
        output = model(image_input_tensor)
        softmax = torch.nn.Softmax(dim=1)
        probability_output = softmax(output)
        predicted_class = torch.argmax(output, dim=1).item()

    return probability_output, predicted_class

"""
Before inference, load an image (png or jpg) from the provided path, and transform it to input tensor by resizing it to the trained model input size (28, 28), 
converting it to grayscale, and normalizing it.
"""
def preprocess_image(image_path, device):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),  # Resize to MNIST size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    return input_tensor


if __name__ == "__main__":
    device = get_device()
    image_path = Path(get_project_root_dir()).joinpath("data/4.png")
    image_input_tensor = preprocess_image(image_path, device)
    output, predicted_class = inference(image_input_tensor, device)
    print(f"probability_output: {output}")
    print(f"Predicted class: {predicted_class}")