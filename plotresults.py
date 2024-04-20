"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
import utils
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

from model_smaller import Model as Model_smaller
from argparse import ArgumentParser
from gluoncv.data import CitySegmentation
from gluoncv.utils.viz import get_color_pallete
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from torchvision import datasets
from model import Model
from model2 import Model as Model2
from model3 import Model as Model3
from Efficient_model import Model as Eff_Model
from Small_model import Model as Small_Model
from Own_Unet import Model as Own_Model
from argparse import ArgumentParser
from utils import map_id_to_train_id
from utils import get_class_weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # You don't have a gpu, so use cpu



def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""
    
    
    torch.manual_seed(42)
    transform = transforms.Compose([transforms.Resize((640, 1280)), 
                            ])
    target_transforms = transforms.Compose([transforms.Resize((640, 1280)),
                            transforms.ToTensor(),])
    #Change the location depending where the data is.
    dataset_train = datasets.Cityscapes('C:/Users/20191891/Documents/Temp/Final/data', split='train', mode='fine', target_type='semantic', transform=transform, target_transform=target_transforms)

    split = 0.85
    boundary = round(split*round(len(dataset_train)))
    train_dataset = torch.utils.data.Subset(dataset_train, range(boundary))
    val_dataset = torch.utils.data.Subset(dataset_train, range(boundary, len(dataset_train)))
    
    model = Model_smaller()
    model.load_state_dict(torch.load('model_smaller.pth', map_location=device))
    model.to(device)
    model.eval()

    transform1 = transforms.Compose([
                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                            transforms.GaussianBlur(kernel_size = 3),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                            ])
    transform2 = transforms.Compose([ 
                            #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                            #transforms.GaussianBlur(kernel_size = 5),
                            transforms.ToTensor(),
                            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                            ])
    plt.figure(figsize=(18,18))
    for j in range(1,4):
        img, mask = val_dataset[j]
        img1 = transform1(img)
        img1 = img1.unsqueeze(0)
        mask = mask.unsqueeze(0)
        img_np = transform2(img)
        img_np = img_np.squeeze()
        mask_np = mask.squeeze()
        img_np = img_np.permute(1,2,0).numpy()
        mask_np = mask_np.squeeze().numpy()*255
        
        colored_mask_np = np.zeros_like(img_np, dtype=np.uint8)
        for label in utils.LABELS:
            colored_mask_np[mask_np == label.id] = label.color
        groundt = Image.fromarray(colored_mask_np, mode='RGB')

        with torch.no_grad():
            output = model(img1.to(device))
        
        _, indices = output.softmax(dim=1).max(dim=1)
        indices = indices.squeeze()
        
        target = torch.zeros((3, indices.shape[0], indices.shape[1]),
                                dtype=torch.uint8, device=indices.device, requires_grad=False)

        for i, lbl in enumerate(utils.LABELS):
            eq = indices.eq(lbl.trainId)

            target[0][eq] = lbl.color[0]
            target[1][eq] = lbl.color[1]
            target[2][eq] = lbl.color[2]

        output_target = TF.to_pil_image(target.cpu(), 'RGB')
        

        

        plt.subplot(3,3,j)
        plt.imshow(img_np)
        plt.title('Input Image')
        plt.axis('off')

        plt.subplot(3,3,j+3)
        plt.imshow(groundt)
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(3,3,j+6)
        plt.imshow(output_target)
        plt.title('Prediction')
        plt.axis('off')

    plt.show()

if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)

