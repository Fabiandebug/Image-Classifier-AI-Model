# train.py
import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from utils import create_classifier, load_category_names, load_checkpoint

def train_model(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    # Define transforms
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)

    # Load the pre-trained model
    model = getattr(models, arch)(pretrained=True)

    # Freeze pre-trained parameters
    for param in model.parameters():
        param.requires_grad = False

    # Create new classifier
    classifier = create_classifier(25088, 102, hidden_units, 0.5)
    model.classifier = classifier

    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Move model to GPU if available
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
            model.eval()
            valid_loss = 0
            accuracy = 0

            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    valid_loss += criterion(outputs, labels).item()

                    ps = torch.exp(outputs)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            model.train()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/len(train_loader):.3f}.. "
                  f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(valid_loader):.3f}")

    # Save the checkpoint
    checkpoint = {
        'input_size': 25088,
        'output_size': 102,
        'hidden_layers': hidden_units,
        'drop_p': 0.5,
        'learnrate': learning_rate,
        'epochs': epochs,
        'arch': arch,
        'state_dict': model.state_dict(),
        'class_to_idx': train_dataset.class_to_idx
    }

    torch.save(checkpoint, save_dir + '/checkpoint.pth')
    print(f"Model saved to {save_dir}/checkpoint.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a new network on a dataset and save the model as a checkpoint.")
    parser.add_argument("data_dir", help="Path to the data directory")
    parser.add_argument("--save_dir", default=".", help="Directory to save checkpoints")
    parser.add_argument("--arch", default="vgg16", help="Choose architecture (e.g., vgg16)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Set learning rate")
    parser.add_argument("--hidden_units", type=int, nargs='+', default=[4096], help="Set hidden layer units")
    parser.add_argument("--epochs", type=int, default=5, help="Set number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")

    args = parser.parse_args()

    train_model(args.data_dir, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)
