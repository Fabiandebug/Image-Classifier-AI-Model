# predict.py
import argparse
import torch
from torchvision import models
from PIL import Image
import numpy as np
from utils import load_checkpoint, process_image, load_category_names

def predict(image_path, checkpoint, top_k, category_names, gpu):
    model = load_checkpoint(checkpoint)
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    image = Image.open(image_path)
    processed_image = process_image(image)
    
    image_tensor = torch.from_numpy(processed_image).type(torch.FloatTensor)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
    
    probabilities, indices = torch.topk(torch.exp(output), top_k)
    probabilities = probabilities[0].cpu().numpy()
    indices = indices[0].cpu().numpy()
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in indices]
    
    if category_names:
        class_names = load_category_names(category_names)
        flower_names = [class_names[class_] for class_ in classes]
    else:
        flower_names = classes
    
    return probabilities, flower_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict flower name from an image using a trained network.")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("checkpoint", help="Path to the checkpoint file")
    parser.add_argument("--top_k", type=int, default=1, help="Return top K most likely classes")
    parser.add_argument("--category_names", help="Path to JSON file mapping categories to real names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")

    args = parser.parse_args()

    probabilities, flower_names = predict(args.image_path, args.checkpoint, args.top_k, args.category_names, args.gpu)

    for prob, name in zip(probabilities, flower_names):
        print(f"Class: {name}, Probability: {prob*100:.2f}%")
