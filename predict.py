from Functions import get_input_args_predict,process_image
import numpy as np
import torch
from torch import nn,optim
from torchvision import models
import json

def main():
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    in_arg = get_input_args_predict()
    image_path = in_arg.image_path
    is_gpu = in_arg.gpu
    top_k = in_arg.top_k
    checkpoint = in_arg.checkpoint
    
    with open(in_arg.category_names, 'r') as json_file:
        cat_to_name = json.load(json_file)
    

    device = torch.device("cuda" if is_gpu and torch.cuda.is_available() else "cpu")

    model = load_checkpoint(checkpoint)
    model.eval()
    model.to(device)
    
    images = process_image(image_path)
    images = torch.from_numpy(images).type(torch.FloatTensor)
    images = images.unsqueeze_(0)
    images = images.to(device)
    
    with torch.no_grad():
        log_ps = model.forward(images)
        ps = torch.exp(log_ps)
        
    probs, classes = ps.topk(top_k,dim=1)
    
    
    class_to_idx = model.class_to_idx
    idx_to_class = {v:k for k, v in model.class_to_idx.items()}
    
    top_p = probs.tolist()[0]
    top_class = [idx_to_class[i.item()] for i in classes[0].data]
    
    names = [cat_to_name[c] for c in top_class]

    print('Top  {} predictions are : {} , classes : {} with probabilities as {}'.format(top_k,names,top_class,top_p))     


def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    
    densenet121 = models.densenet121(pretrained=True)
    vgg19 = models.vgg19(pretrained=True)
    
    models_ = {'densenet': densenet121, 'vgg': vgg19}
    
    model = models_[checkpoint['arch']]
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    optimizer = optim.Adam(model.classifier.parameters(), checkpoint['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
   
    return model

if __name__ == '__main__':
    
    main()
