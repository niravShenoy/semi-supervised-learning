import argparse
import math
import logging
import os

from dataloader import get_cifar10, get_cifar100
from vat        import VATLoss
from utils      import accuracy
from model.wrn  import WideResNet

import torch
import torch.optim as optim
from torch.utils.data   import DataLoader
import torch.nn as nn
from utils import accuracy

curr_path = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S', filename=os.path.join(curr_path, 'out.task1.log'))

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
    test_loader         = DataLoader(test_dataset,
                                    batch_size = args.test_batch,
                                    shuffle = False, 
                                    num_workers=args.num_workers)
    
    model       = WideResNet(args.model_depth, 
                                args.num_classes, widen_factor=args.model_width)
    model       = model.to(device)

    ############################################################################
    # TODO: SUPPLY your code
    ############################################################################
    #logging.info('%s; Num Labeled = %s; Epochs = %s; LR = %s; VAT xi = %s; VAT eps = %s; alpha = %s',
                 #args.dataset, args.num_labeled, args.epoch, args.lr, args.vat_xi, args.vat_eps, args.alpha)

    #init_path = os.path.join(curr_path, 'init_model.pt')
    #torch.save(model.state_dict(), init_path)


    #def save_checkpoint(checkpoint, best_path):
     #   logging.info('Saving model of epoch %s with validation accuracy = %.3f and loss = %.3f',
      #               checkpoint['epoch'], checkpoint['validation_accuracy'], checkpoint['validation_loss'])
       # torch.save(checkpoint, best_path)

    
    loss_fn = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    

    vatLoss = VATLoss(args)
    
    args.epoch = 5
    args.iter_per_epoch = 20
    steps = 0
    print_every = 40
    
    for epoch in range(args.epoch):
        model.train()
        loss_list = []
        correct = 0
        total = 0
        running_loss = 0.0
        
        for i in range(args.iter_per_epoch):
            steps += 1

            try:
                x_l, y_l    = next(labeled_loader)
            except StopIteration:
                labeled_loader      = iter(DataLoader(labeled_dataset, 
                                            batch_size = args.train_batch, 
                                            shuffle = True, 
                                            num_workers=args.num_workers))
                x_l, y_l    = next(labeled_loader)
            
            try:
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
            # TODO: SUPPLY you code
            ####################################################################

            optimizer.zero_grad()

            
            vaLoss = vatLoss(model, x_ul)
            predictions = model(x_l)
            
            
            correct += (torch.argmax(predictions, axis=1)
                            == y_l).float().sum()
            total += float(x_l.size(dim=0))

            classifcationLoss = loss_fn(predictions, y_l)
            loss = classifcationLoss + args.alpha*vaLoss
         
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        train_accuracy = 100 * correct / total
        running_loss /= args.iter_per_epoch
        loss_list.append(running_loss)

        with torch.no_grad():
          model.eval()
          test_loss = 0.0
          correct = 0.0
          for j, (x_v, y_v) in enumerate(test_loader):
            x_v, y_v = x_v.to(device), y_v.to(device)
            y_op_val = model(x_v)
            loss = loss_fn(y_op_val, y_v)
            
            test_loss += loss.item()
            _, y_pred_test = y_op_val.max(1)
            correct += y_pred_test.eq(y_v).sum()

          test_accuracy = 100 * correct.float() / len(test_loader.dataset)
          test_loss = test_loss / j

          print("Epoch {}/{}, Train Accuracy: {:.3f}, Test Accuracy: {:.3f}, Training Loss: {:.3f}, Test Loss: {:.3f}".format(
                epoch+1,
                args.epoch,
                train_accuracy.item(),
                test_accuracy,
                running_loss,
                test_loss
            ))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virtual adversarial training \
                                        of CIFAR10/100 using with pytorch")
    parser.add_argument("--dataset", default="cifar10", 
                        type=str, choices=["cifar10", "cifar100"])
    parser.add_argument("--datapath", default="./data/", 
                        type=str, help="Path to the CIFAR-10/100 dataset")
    parser.add_argument('--num-labeled', type=int, 
                        default=4000, help='Total number of labeled samples')
    parser.add_argument("--lr", default=0.03, type=float, 
                        help="The initial learning rate") 
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="Optimizer momentum")
    parser.add_argument("--wd", default=0.00005, type=float,
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
    parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                        help='regularization coefficient (default: 0.01)')
    parser.add_argument("--dataout", type=str, default="./path/to/output/",
                        help="Path to save log files")
    parser.add_argument("--model-depth", type=int, default=28,
                        help="model depth for wide resnet") 
    parser.add_argument("--model-width", type=int, default=2,
                        help="model width for wide resnet")
    parser.add_argument("--vat-xi", default=10.0, type=float, 
                        help="VAT xi parameter")
    parser.add_argument("--vat-eps", default=1.0, type=float, 
                        help="VAT epsilon parameter") 
    parser.add_argument("--vat-iter", default=1, type=int, 
                        help="VAT iteration parameter") 
    # Add more arguments if you need them
    # Describe them in help
    # You can (and should) change the default values of the arguments
    
    args = parser.parse_args()

    main(args)