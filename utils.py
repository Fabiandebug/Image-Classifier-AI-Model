# utils.py
import torch
from torch import nn  # Add this import statement
from torchvision import models
from PIL import Image
import json


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    
    classifier = create_classifier(checkpoint['input_size'], 
                                   checkpoint['output_size'], 
                                   checkpoint['hidden_layers'],
                                   checkpoint['drop_p'])
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


def process_image(image):
    size = 256
    image.thumbnail((size, size))
    
    crop_size = 224
    left = (size - crop_size) / 2
    top = (size - crop_size) / 2
    right = (size + crop_size) / 2
    bottom = (size + crop_size) / 2
    image = image.crop((left, top, right, bottom))
    
    np_image = (np.array(image) / 255.0)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image


def create_classifier(input_size, output_size, hidden_layers, drop_p):
    layers = []
    layer_sizes = [input_size] + hidden_layers + [output_size]
    
    for i in range(len(layer_sizes)-1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(drop_p))
    
    layers.pop()  # Remove the last dropout layer
    
    return nn.Sequential(*layers)


def load_category_names(file_path):
    with open(file_path, 'r') as f:
        category_names = json.load(f)
    return category_names
