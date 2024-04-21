import os
import torch
from PIL import Image
from torchvision import transforms
from visualDetection import load_model, predict, make_decision
from utils import calculate_mean_std
import argparse

def load_and_transform_images(directory, transform):
    """Load images directly from a directory and apply transformations."""
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            image = Image.open(img_path).convert('RGB')  # Ensure it's RGB
            images.append((transform(image), filename))
    return images


if __name__ == '__main__':
    print("Script started")  # This should print when you run the script
    parser = argparse.ArgumentParser(description='Evaluate a trained model on image data')
    parser.add_argument('model_path', type=str, help='Path to the trained model file')
    args = parser.parse_args()

    print("Model path:", args.model_path)  # This should print the path to the model

    base_dir = os.getcwd() + "/augmented_data"
    eval_dir = os.path.join(os.getcwd(), 'eval')

    calc_mean, calc_std = calculate_mean_std(base_dir)
    
    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((80, 80)),
        transforms.ToTensor(),
        transforms.Normalize(mean=calc_mean, std=calc_std)
    ])
    
    # Load the images
    eval_images = load_and_transform_images(eval_dir, transform)
    
    # Load the model
    model = load_model(args.model_path)  # Use the path from command line argument
    
    # Predict the class of each image
    for img_tensor, img_name in eval_images:
        prediction = predict(model, img_tensor)
        decision = make_decision(prediction, threshold=0.5)
        print(f"{img_name} : {prediction:.4f} -> {decision}")
