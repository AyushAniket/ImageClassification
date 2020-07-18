from Functions import get_input_args_train,process_data
from torch import nn, optim
import torch
from torchvision import models
from workspace_utils import active_session

def main():

    in_arg = get_input_args_train()
    is_gpu = in_arg.gpu
    hidden_units = in_arg.hidden_units
    arch = in_arg.arch
    lr = in_arg.learning_rate
    save_directory = in_arg.save_dir
    data_dir = in_arg.data_dir
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    trainloader, testloader, validloader, trainsets = process_data(train_dir,test_dir,valid_dir)

    print('Building network with architecture {}'.format(arch))
    device = torch.device("cuda" if is_gpu and torch.cuda.is_available() else "cpu")

    densenet121 = models.densenet121(pretrained=True);
    vgg19 = models.vgg19(pretrained=True);
    
    models_ = {'densenet': densenet121, 'vgg': vgg19}
    
    model = models_[arch];

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    if arch == 'densenet':
        
        model.classifier = nn.Sequential(nn.Linear(1024,hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units, 102),
                                 nn.LogSoftmax(dim=1))
    else:

        model.classifier = nn.Sequential(nn.Linear(25088,4096),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),
                                 nn.Linear(4096,hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),
                                 nn.Linear(hidden_units, 102),
                                 nn.LogSoftmax(dim=1))

        

    criterion = nn.NLLLoss()
    
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    model.to(device);



    print('Training network')
    
    with active_session():
        epochs = in_arg.epochs
        steps = 0
        running_loss = 0
        print_every = 10
    
        for epoch in range(epochs):
        
            for images, labels in trainloader:
            
                steps += 1
            
                # Move input and label tensors to the default device
                images, labels = images.to(device), labels.to(device)
        
                optimizer.zero_grad()
            
                #feed forward
                log_ps = model.forward(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
        
                if steps % print_every == 0:
                
                    valid_loss = 0
                    accuracy = 0
                    model.eval()
                
                    with torch.no_grad():
                    
                        for images, labels in validloader:
                        
                            images, labels = images.to(device), labels.to(device)
                        
                            log_ps = model.forward(images)
                            batch_loss = criterion(log_ps, labels)
                    
                            valid_loss += batch_loss.item()
                    
                            # Calculating accuracy
                            ps = torch.exp(log_ps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    print(f"Epoch {epoch+1}/{epochs}.. "
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"Test loss: {valid_loss/len(validloader):.3f}.. "
                        f"Test accuracy: {accuracy/len(validloader):.3f}")
                
                    running_loss = 0
                    model.train()

    checkpoint = {'classifier':model.classifier,
              'state_dict': model.state_dict(),
              'optimizer_state':optimizer.state_dict(),
              'class_to_idx':trainsets.class_to_idx,
              'arch':arch,
              'learning_rate':lr}

    print('Saving checkpoint')
    torch.save(checkpoint, save_directory +'checkpoint.pth')


if __name__ == '__main__':
    
    main()
