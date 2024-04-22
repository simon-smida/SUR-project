# File        : cnn-eval.py
# Date        : 17.4. 2024
# Description : Evaluate a trained model on image data and output predictions

import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from imageUtils import calculate_mean_std, load_model, predict


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
    
    # Argument parser
    parser = argparse.ArgumentParser(description='Evaluate a trained model on image data')
    parser.add_argument('model_path', type=str, help='Path to the trained model file')
    args = parser.parse_args()

    base_dir = os.getcwd() + "/augmented_data"
    eval_dir = os.path.join(os.getcwd(), 'eval')

    # Calculate mean and std on augmented data + original data (both train + dev)
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
    model = load_model(args.model_path)
    
    # Predict the class of each image
    for img_tensor, img_name in eval_images:
        img_name = img_name.split('.')[0]
        prediction = predict(model, img_tensor)
        # Decision threshold of 0.5
        decision = '1' if prediction >= 0.5 else '0' 
        print(f"{img_name} {prediction:.4f} {decision}")