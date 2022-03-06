import torch
from dataloader import get_cifar10, get_cifar100
from model.wrn import WideResNet
import torch.nn as nn
from utils import accuracy

def test_cifar10(args, device, testdataset, filepath = "./path/to/model.pth.tar"):
    '''
    args: 
        testdataset : (torch.utils.data.Dataset)
        filepath    : (str) The path to the model file that is saved
    returns : (torch.Tensor) logits of the testdataset with shape 
                [num_samples, 10]. Apply softmax to the logits
    
    Description:
        This function loads the model given in the filepath and returns the 
        logits of the testdataset which is a torch.utils.data.Dataset. You can
        save the arguments needed to load the models (e.g. width, depth etc) 
        with the model file. Assume testdataset is like CIFAR-10. Test this
        function with the testdataset returned by get_cifar10()
    '''
    # TODO: SUPPLY the code for this function
    model = WideResNet(args.model_depth, 
                                args.num_classes, widen_factor=args.model_width)
    model = model.to(device)
    _, model = load_checkpoint(filepath, model)
    criterion = nn.CrossEntropyLoss()
    evaluate_model(model, testdataset, criterion, device)
    # raise NotImplementedError

def test_cifar100(args, device, testdataset, filepath="./path/to/model.pth.tar"):
    '''
    args: 
        testdataset : (torch.utils.data.Dataset)
        filepath    : (str) The path to the model file that is saved
    returns : (torch.Tensor) logits of the testdataset with shape 
                [num_samples, 100]. Apply softmax to the logits
    
    Description:
        This function loads the model given in the filepath and returns the 
        logits of the testdataset which is a torch.utils.data.Dataset. You can
        save the arguments needed to load the models (e.g. width, depth etc) 
        with the model file. Assume testdataset is like CIFAR-100. Test this
        function with the testdataset returned by get_cifar100()
    '''
    # TODO: SUPPLY the code for this function
    model = WideResNet(args.model_depth, 
                                args.num_classes, widen_factor=args.model_width)
    model = model.to(device)
    _, model = load_checkpoint(filepath, model)
    criterion = nn.CrossEntropyLoss()
    evaluate_model(model, testdataset, criterion, device)
    # raise NotImplementedError

def load_checkpoint(ckpt_path, model):
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])
    return checkpoint['validation_loss'], model

def evaluate_model(model, test_loader, criterion, device):
    with torch.no_grad():
        model.eval()
        test_loss = 0.0
        correct = 0.0
        for j, (x_t, y_t) in enumerate(test_loader):
            x_t, y_t = x_t.to(device), y_t.to(device)
            y_op_test = model(x_t)
            loss = criterion(y_op_test, y_t)

            test_loss += loss.item()
            _, y_pred_test = y_op_test.max(1)
            correct += y_pred_test.eq(y_t).sum()

        test_accuracy = 100 * correct.float() / len(test_loader.dataset)
        test_loss = test_loss / j

        print("Test Accuracy: {:.3f}, Test Loss: {:.3f}".format( 
            test_accuracy, 
            test_loss
        ))