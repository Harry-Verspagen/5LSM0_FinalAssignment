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
import os
from model import Model
from torchvision.datasets import Cityscapes
from argparse import ArgumentParser
from gluoncv.data import CitySegmentation
from gluoncv.utils.viz import get_color_pallete
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
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)


def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""

    # data loading
    #dataset = Cityscapes("\Data", split='train', mode='fine', target_type='semantic')
    #trainloader = torch.utils.data.DataLoader(dataset)
    transform = transforms.Compose([transforms.Resize((256, 256)),
                            transforms.ToTensor(),])
    target_transforms = transforms.Compose([transforms.Resize((256,256)),
                            transforms.ToTensor(),])
    train_dataset = datasets.Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic', transform=transform, target_transform=transform)
    # 'C:/Users/20191891/Documents/Temp/Final/data'
    #train_dataset = CitySegmentation(split='train')
    #trainloader = torch.utils.data.DataLoader(train_dataset)

    
    # visualize example images
    img, mask = train_dataset[1]
    print(img.dtype, mask.dtype)
    img_np = img.permute(1, 2, 0).numpy()
    mask_np = mask.squeeze().numpy()*255
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    #mask_colored = utils.map_id_to_train_id(mask_np)
    colored_mask_np = np.zeros_like(img_np, dtype=np.uint8)
    for label in utils.LABELS:
        colored_mask_np[mask_np == label.id] = label.color
    target = Image.fromarray(colored_mask_np, mode='RGB')
    ax[0].imshow(img_np)
    ax[1].imshow(target)
    trainloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=False)
    #validationloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=False)
    plt.show()
    
    # define model
    model = Model().to(device)

    # define optimizer and loss function (don't forget to ignore class index 255)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # training/validation loop
    #train_model_segmentation(model,trainloader, num_epochs=5,lr=0.001)
    train_model(model,trainloader, criterion, optimizer, num_epochs=2, lr=0.01)
    # save model


    # visualize some results
    wandb.finish()

    
#class MyCitySpace(torchvision.datasets.Cityspaces):
#    def __getitem__(self, item):
#        inputs, target = super().__getitem__(item)
#        return inputs, np.array(target)

def train_model(model, train_loader, criterion, optimizer, num_epochs=2, lr=0.01):
 #   criterion = nn.CrossEntropyLoss()
  #  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            labels = (labels*255).squeeze().long().to(device)
            optimizer.zero_grad()
            
            #inputs = inputs.view(3, 1024*2048)
            outputs = model(inputs.to(device))
            print(outputs.shape, labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        wandb.log({'loss': epoch_loss})
        wandb.log(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')



def train_model_segmentation(model, train_loader, num_epochs=5, lr=0.001, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    #epoch_data = collections.defaultdict(list)
    count = 0
    loss_list = []
    iteration_list = []
    scheduler = StepLR(optimizer, step_size=40, gamma=0.1)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), (data[1]*255).to(device).long()
            labels = map_id_to_train_id(labels).to(device) 
            #labels = labels.argmax(dim=1)
            labels=labels.squeeze(1)
            optimizer.zero_grad()

            #print(labels.unique())
            #print(labels.shape)
            optimizer.zero_grad()

            outputs = model(inputs)
            labels = labels.squeeze(1)

            loss = criterion(outputs, labels)
            v=epoch + 1
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            wandb.log({'loss': running_loss/(i+1)})
            wandb.log({'train_Iteration': i})
            wandb.log({'Train_Epoch':float(v)})
            
            print(f'Epoch {epoch + 1}, Iteration [{i}/{len(train_loader)}], Loss: {running_loss/(i+1)}')
        
       # clear_output()
        print(f'Finished epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')
        #if (epoch < 85):
        scheduler.step()


if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)

