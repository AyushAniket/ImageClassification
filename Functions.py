# Imports modules
import argparse

import torch
from torchvision import transforms,datasets,models

from PIL import Image

import numpy as np


 
def get_input_args_train():

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type = str, default = 'flowers',
                        help='dataset directory')
    parser.add_argument('--save_dir', type = str, default = '/home/workspace/ImageClassifier/', 
                        help = 'path to the folder for saving checkpoints')
    parser.add_argument('--arch',type = str, default = 'densenet',
                        help = 'NN Model Architecture vgg or densenet. default = densenet')
    parser.add_argument('--learning_rate',type = float, default = 0.001,
                        help = 'value of learning rate')
    parser.add_argument('--hidden_units',type = int, default = 512,
                        help = 'number of hidden units')
    parser.add_argument('--epochs',type = int, default = 10,
                        help = 'number of iterations for training network')
    parser.add_argument('--gpu', type = bool, default = 'False',
                        help='device to run your model : gpu or cpu. Default = False i.e cpu')
    
    return parser.parse_args()

def get_input_args_predict():

    parser = argparse.ArgumentParser()
                        
    parser.add_argument('--image_path', type = str, default = '/home/workspace/ImageClassifier/flowers/test/1/image_06743.jpg', 
                        help = 'path to image')
    parser.add_argument('--checkpoint',type = str, default = 'checkpoint.pth',
                        help = 'trained model checkpoint')
    parser.add_argument('--top_k',type = int, default = 3,
                        help = 'number of classes with highest prob.')
    parser.add_argument('--category_names', default = 'cat_to_name.json',
                        help = 'mapping of categories to real names file')
    parser.add_argument('--gpu', type = bool, default = 'False',
                        help='device to run your model : gpu or cpu.Default = False i.e cpu')               
    
    return parser.parse_args()


def process_data(train_dir, test_dir, valid_dir):
                        
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    trainsets = datasets.ImageFolder(train_dir, transform = train_transforms)
    testsets = datasets.ImageFolder(test_dir, transform = test_transforms)
    validsets = datasets.ImageFolder(valid_dir, transform = test_transforms)
                        
    trainloader = torch.utils.data.DataLoader(trainsets, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testsets, batch_size=64)
    validloader = torch.utils.data.DataLoader(validsets, batch_size=64)
                        
    return trainloader, testloader, validloader, trainsets


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image)
    
    if image.size[0] > image.size[1]:
        aspect = image.size[1] / 256
        new_size = (image.size[0] / aspect, 256)
    else:
        aspect = image.size[0] / 256
        new_size = (256, image.size[1] / aspect)
        image.thumbnail(new_size, Image.ANTIALIAS)
        
    # crop out center of image
    width, height = image.size # Get dimensions
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    image = image.crop((left, top, right, bottom))
       
    np_image = np.array(image)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = np_image / 255.0
    np_image = (np_image - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
   

    return np_image                        
                
