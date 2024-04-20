"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
import wandb
import utils
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

#Import the different models
from model import Model
from unet2 import Model as Unet_plus_model
from Efficient_model import Model as Eff_Model
from Small_model import Model as Small_Model
from Own_Unet import Model as Own_Unet
from model_smaller import Model as Model_Smaller

from decimal import *
from torchvision.datasets import Cityscapes
from argparse import ArgumentParser
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets

from argparse import ArgumentParser
from torch.optim.lr_scheduler import StepLR
import torch.optim.lr_scheduler as lr_scheduler
from utils import map_id_to_train_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # You don't have a gpu, so use cpu


wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": "Adaptive",
    "architecture": "U-net",
    "dataset": "CityScapes",
    "epochs": 120,
    }
)


def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    parser.add_argument('--resize_height', type=int, default=512, help="The height for the resized image")
    parser.add_argument('--resize_width', type=int, default=512, help="The width for the resized image")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="The learning rate")
    parser.add_argument('--batch_size', type=int, default=8, help="The batch size")
    parser.add_argument('--weight_decay', type=float, default=2e-4, help="The weight decay")
    parser.add_argument('--epochs', type=int, default=120, help="The number of epochs")
    parser.add_argument('--patience', type=int, default=4, help="Number of epochs with no validation error improvement before stopping")
    parser.add_argument('--step_size', type=int, default=20, help="Step size for the learning rate schedular")
    parser.add_argument('--gamma', type=float, default = 0.1, help="Factor for learning rate schedular")

    return parser


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""
    
    transform = transforms.Compose([transforms.Resize((args.resize_height, args.resize_width)), #Change the size: 1024,2048
                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                            transforms.GaussianBlur(kernel_size = 3),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                            ])
    target_transforms = transforms.Compose([transforms.Resize((args.resize_height, args.resize_width)),
                            transforms.ToTensor(),])
    dataset_train = datasets.Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic', transform=transform, target_transform=target_transforms)

    split = 0.85
    boundary = round(split*round(len(dataset_train)))
    train_dataset = torch.utils.data.Subset(dataset_train, range(boundary))
    val_dataset = torch.utils.data.Subset(dataset_train, range(boundary, len(dataset_train)))

    # visualize example images
    img, mask = train_dataset[1]
    img_np = img.permute(1, 2, 0).numpy()
    mask_np = mask.squeeze().numpy()*255
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    colored_mask_np = np.zeros_like(img_np, dtype=np.uint8)
    for label in utils.LABELS:
        colored_mask_np[mask_np == label.id] = label.color
    target = Image.fromarray(colored_mask_np, mode='RGB')
    ax[0].imshow(img_np)
    ax[1].imshow(target)
    plt.show()
    
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False, drop_last=True)
    validationloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False, drop_last=True)
    pipe = DataLoader(train_dataset, batch_size=len(train_dataset))
    
    # define model
    model = Model_Smaller().to(device)

    # Print the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    
    # define optimizer and loss function (don't forget to ignore class index 255)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=255) 

    # training/validation loop
    train_model(args, model,trainloader, validationloader, criterion, optimizer)

    wandb.finish()

    # save model
    torch.save(model.state_dict(), '/gpfs/home6/scur0781/FinalAssignment/submit/stored_models/model_smaller.pth')


def train_model(args, model, train_loader, val_loader, criterion, optimizer):
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    num_epochs = args.epochs
    counter = 0
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            labels = (labels*255).squeeze().long()
            labels = utils.map_id_to_train_id(labels).to(device)
            optimizer.zero_grad()
            
            #inputs = inputs.view(3, 1024*2048)
            outputs = model(inputs.to(device))
            #print(outputs.shape, labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        v = epoch + 1   
        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        wandb.log({'Trainloss': epoch_loss})
        wandb.log({'Epoch': float(v)})
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        val_loss = val_model(model, val_loader, criterion)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
        if counter >= args.patience:
            break

        #Possible option to save the model for every number of epochs
        if epoch % 10 == 0:
            checkpoint = {
                'epochs' : epoch,
                'state_dict' : model.state_dict()
            }
            #torch.save(checkpoint, '/gpfs/home6/scur0781/FinalAssignment/submit/stored_models/Eff_model_{}_{}.pth'.format(epoch, loss))

def val_model(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            labels = (labels*255).squeeze().long()
            labels = utils.map_id_to_train_id(labels).to(device)
            outputs = model(inputs.to(device))
            #print(outputs.shape, labels.shape)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0) #Because of batchsize

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    average_loss = total_loss / total_samples
    accuracy = total_correct / (total_samples*256*128)

    wandb.log({'Validationloss': average_loss})
    wandb.log({'Accuracy': accuracy})
    print(f'Validationloss: {average_loss}')
    return average_loss

if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)



