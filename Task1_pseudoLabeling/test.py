import logging
import os

import torch
from dataloader import get_cifar10, get_cifar100
from model.wrn import WideResNet
import torch.nn as nn
from utils import accuracy

curr_path = os.path.dirname(os.path.abspath(__file__))

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
    logits = evaluate_model(model, testdataset, criterion, device)
    top1, topk = find_model_accuracy(model, testdataset, device)
    return logits
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
    logits = evaluate_model(model, testdataset, criterion, device)
    top1, topk = find_model_accuracy(model, testdataset, device)
    return logits
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
        y_logits = []
        for j, (x_t, y_t) in enumerate(test_loader):
            x_t, y_t = x_t.to(device), y_t.to(device)
            y_op_test = model(x_t)
            loss = criterion(y_op_test, y_t)

            test_loss += loss.item()
            _, y_pred_test = y_op_test.max(1)
            correct += y_pred_test.eq(y_t).sum()
            y_logits.append(y_op_test)

        test_accuracy = 100 * correct.float() / len(test_loader.dataset)
        test_loss = test_loss / j

        logging.info("Test Accuracy: %.3f, Test Loss: %.3f", 
            test_accuracy, 
            test_loss
        )
        print("Logits= ",y_logits)
        return y_logits

def save_checkpoint(checkpoint, best_path):
        logging.info('Saving model of epoch %s with validation accuracy = %.3f and loss = %.3f',
                     checkpoint['epoch'], checkpoint['validation_accuracy'], checkpoint['validation_loss'])
        torch.save(checkpoint, best_path)

def find_model_accuracy(model, test_loader, device):
    # _, model = load_checkpoint(path, model)
    with torch.no_grad():
        model.eval()

        test_accuracy = torch.empty((0,2))
        for j, (x_v, y_v) in enumerate(test_loader):
            x_v, y_v = x_v.to(device), y_v.to(device)
            y_op_val = model(x_v)
            res = accuracy(y_op_val, y_v, (1,5))
            res = torch.FloatTensor(res).reshape(1,2)
            test_accuracy = torch.cat((test_accuracy, res),0)
            # loss = criterion(y_op_val, y_v)
        top1, topk = enumerate(torch.div(torch.sum(test_accuracy, dim=0),test_accuracy.shape[0]))
        print('Top 1 Accuracy = {}; Top 5 Accuracy = {}'.format(top1[1].item(), topk[1].item()))
        logging.info('Top 1 Accuracy = %s; Top 5 Accuracy = %s', top1[1].item(), topk[1].item())
        return top1[1].item(), topk[1].item()