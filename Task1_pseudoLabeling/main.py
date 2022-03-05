import argparse
import math

from dataloader import get_cifar10, get_cifar100
from utils      import accuracy

from model.wrn  import WideResNet

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data   import DataLoader


def main(args):
    if args.dataset == "cifar10":
        args.num_classes = 10
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar10(args, 
                                                                args.datapath)
    if args.dataset == "cifar100":
        args.num_classes = 100
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar100(args, 
                                                                args.datapath)
    args.epoch = math.ceil(args.total_iter / args.iter_per_epoch)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    labeled_loader      = iter(DataLoader(labeled_dataset, 
                                    batch_size = args.train_batch, 
                                    shuffle = True, 
                                    num_workers=args.num_workers))
    unlabeled_loader    = iter(DataLoader(unlabeled_dataset, 
                                    batch_size=args.train_batch,
                                    shuffle = True, 
                                    num_workers=args.num_workers))
    test_loader         = iter(DataLoader(test_dataset,
                                    batch_size = args.test_batch,
                                    shuffle = False, 
                                    num_workers=args.num_workers))
    
    model       = WideResNet(args.model_depth, 
                                args.num_classes, widen_factor=args.model_width)
    model       = model.to(device)

    ############################################################################
    # TODO: SUPPLY your code
    ############################################################################

    def save_checkpoint(model, ckpt_path):
        raw_model = model.module if hasattr(model, "module") else model
        print('Saving model')
        torch.save(raw_model.state_dict(), ckpt_path)

    ckpt_path='/Task1_pseudoLabeling/cifar10_model.pt'
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params = model.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.wd)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    threshold = args.threshold

    best_loss = float('inf')
    
    for epoch in range(args.epoch):
        model.train()
        x_pseudo_set = []
        y_pseudo_set = []
        correct = 0
        total = 0
        for i in range(args.iter_per_epoch):
            try:
                #labeled data
                x_l, y_l    = next(labeled_loader)
            except StopIteration:
                labeled_loader      = iter(DataLoader(labeled_dataset, 
                                            batch_size = args.train_batch, 
                                            shuffle = True, 
                                            num_workers=args.num_workers))
                x_l, y_l    = next(labeled_loader)
            
            try:
                #unlabeled data
                x_ul, _     = next(unlabeled_loader)
            except StopIteration:
                unlabeled_loader    = iter(DataLoader(unlabeled_dataset, 
                                            batch_size=args.train_batch,
                                            shuffle = True, 
                                            num_workers=args.num_workers))
                x_ul, _     = next(unlabeled_loader)
            
            x_l, y_l    = x_l.to(device), y_l.to(device)
            x_ul        = x_ul.to(device)
            ####################################################################
            # TODO: SUPPLY your code
            ####################################################################

            #concatenate labeled and unlabeled
            if x_pseudo_set:          
                x_pseudo_tensor = torch.stack(x_pseudo_set).to(device)
                x_l = torch.cat((x_l, x_pseudo_tensor))
                y_l = torch.cat((y_l, torch.tensor(y_pseudo_set).to(device)))

            #train model
            y_pred_l = model(x_l)

            #compute loss
            correct += (torch.argmax(y_pred_l, axis=1) == y_l).float().sum()
            total += float(x_l.size(dim=0))

            loss = criterion(y_pred_l, y_l)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #predict unlabeled
            y_pseudo_pred = model(x_ul)

            #add to subset if probability is greater than threshold
            y_pseudo_label_prob, y_pseudo_label_class = torch.max(y_pseudo_pred, axis=1)

            x_pseudo_set = []
            y_pseudo_set = []
            for k, row in enumerate(y_pseudo_label_prob):
                if row > threshold:
                    x_pseudo_set.append(x_ul[k,:,:,:])
                    y_pseudo_set.append(y_pseudo_label_class[k])
            ## End of batch

        accuracy = 100 * correct / total

        with torch.no_grad():
            model.eval()
            try:
                #labeled data
                x_t, y_t    = next(test_loader)
            except StopIteration:
                test_loader      = iter(DataLoader(test_dataset, 
                                            batch_size = args.test_batch, 
                                            shuffle = False, 
                                            num_workers=args.num_workers))
                x_t, y_t    = next(test_loader)
            x_t, y_t = x_t.to(device), y_t.to(device)
            y_pred_test = model(x_t)
            accuracy_test = 100 * torch.mean((torch.argmax(y_pred_test, axis=1) == y_t).float())
            test_loss = criterion(y_pred_test, y_t)
            print("Epoch {}/{}, Train Accuracy: {:.3f}, Test Accuracy: {:.3f}, Test Loss: {:.3f}".format(epoch+1, args.epoch, accuracy.item(), accuracy_test.item(), test_loss.item()))
            if test_loss < best_loss:
                best_loss = test_loss
                save_checkpoint(model, ckpt_path)
        # scheduler.step()
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo labeling \
                                        of CIFAR10/100 with pytorch")
    parser.add_argument("--dataset", default="cifar10", 
                        type=str, choices=["cifar10", "cifar100"])
    parser.add_argument("--datapath", default="./data/", 
                        type=str, help="Path to the CIFAR-10/100 dataset")
    parser.add_argument('--num-labeled', type=int, 
                        default=4000, help='Total number of labeled samples')
    # parser.add_argument("--lr", default=0.03, type=float, 
    #                     help="The initial learning rate") 
    parser.add_argument("--lr", default=0.1, type=float, 
                        help="The initial learning rate")                     
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="Optimizer momentum")
    # parser.add_argument("--wd", default=0.00005, type=float,
    #                     help="Weight decay")
    parser.add_argument("--wd", default=0.0001, type=float,
                        help="Weight decay")
    parser.add_argument("--expand-labels", action="store_true", 
                        help="expand labels to fit eval steps")
    parser.add_argument('--train-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--test-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--total-iter', default=1024*512, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=1024, type=int,
                        help="Number of iterations to run per epoch")
    parser.add_argument('--num-workers', default=1, type=int,
                        help="Number of workers to launch during training")
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Confidence Threshold for pseudo labeling')
    parser.add_argument("--dataout", type=str, default="./path/to/output/",
                        help="Path to save log files")
    parser.add_argument("--model-depth", type=int, default=28,
                        help="model depth for wide resnet") 
    parser.add_argument("--model-width", type=int, default=2,
                        help="model width for wide resnet")
    
    # Add more arguments if you need them
    # Describe them in help
    # You can (and should) change the default values of the arguments
    
    args = parser.parse_args()

    main(args)