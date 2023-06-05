import os
import argparse
from tkinter.tix import Tree
import torch
from torch import nn
from torch.backends import cudnn
# import dataset as dataset
import numpy as np
import torch.nn.functional as F
from wideresnet import WideResNet
from lenet import LeNet
from resnet import resnet
# import logging
import copy
import re

parser = argparse.ArgumentParser(description='FREDIS for Unreliable Partial Label Learning with consistency regularization.')
parser.add_argument('--rounds', default=250, type=int, metavar='N', help='number of total rounds to run')
parser.add_argument('--epochs', default=250, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--bs', default=256, type=int)
parser.add_argument('--lam', default=1.0, type=float)
parser.add_argument('--alpha', default=1.0, type=float)
parser.add_argument('--beta', default=1.0, type=float)

parser.add_argument('--T', default=1.0, type=float)
parser.add_argument('--decay', default=0.0, type=float)

parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'fmnist', 'kmnist'], default='cifar10')

parser.add_argument('--model', type=str, choices=['lenet', 'resnet'], default='resnet')
parser.add_argument('--load_weight', type=int, choices=[0, 1], default=1)
parser.add_argument('--weight_path', type=str, default="")
parser.add_argument('--warm_up', help='warm up step', type=int, default=250)

parser.add_argument("--theta", help="initial threshold", type=float, default=1e-6)
parser.add_argument("--inc", help="increment of threshold", type=float, default=1e-6)
parser.add_argument("--delta", help="initial threshold", type=float, default=1.0)
parser.add_argument("--dec", help="increment of threshold", type=float, default=1e-6)
parser.add_argument("--times", type=float, default=2.0)
parser.add_argument("--change_size", type=int, default=500)
parser.add_argument("--update_interval", type=int, default=20)


parser.add_argument('--lr', default=0.05, type=float)
parser.add_argument('--wd', default=0.001, type=float)
parser.add_argument('--partial_rate', default=0.5, type=float)
parser.add_argument('--noisy_rate', default=0.1, type=float)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--used_seed', default=2, type=int)
parser.add_argument('--lo', type=str, default='fredis')


def load_args_from_file(args, file_path=""):
    def str_to_typefun(type_str):
        if type_str == "int":
            return int
        if type_str == "float":
            return float
        if type_str == "str":
            return str

    with open(file_path, "r") as f:
        args_str = f.readlines()[0]

    args_str = re.findall(r"[(](.*?)[)]", args_str)[0]
    key_values = args_str.split(", ")
    for key_value in key_values:
        key, value = key_value.split("=", maxsplit=1)
        value = eval(value)
        typefun = str_to_typefun(type(getattr(args, key, None)).__name__)
        setattr(args, key, typefun(value))
    return args

args = parser.parse_args()
file_path = "hyper_parameters/{}/n={}_p={}/suggest.log".format(args.dataset, args.noisy_rate, args.partial_rate)
args = load_args_from_file(args, file_path=file_path)
print(args)

num_classes = 10 if args.dataset != 'cifar100' else 100
device = torch.device("cuda")

import random
def set_seed(seed):
    torch.manual_seed(seed) # 为CPU设置随机种子
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.	
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def accuracy_check(loader, model, device):
    with torch.no_grad():
        total, num_samples = 0, 0
        for images, labels in loader:
            labels, images = labels.to(device), images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # _, y = torch.max(labels.data, 1)
            # print(predicted, labels)
            total += (predicted == labels).sum().item()
            num_samples += labels.size(0)

    return 100*(total/num_samples)


def DPLL_train(args, train_loader, model, optimizer, epoch, consistency_criterion, confidence, posterior):
    """
        Run one train epoch
    """
    model.train()
    for i, (x_aug0, x_aug1, x_aug2, part_y, y, index) in enumerate(train_loader):
        part_y = part_y.float().cuda()
        # original samples with pre-processing
        x_aug0 = x_aug0.cuda()
        y_pred_aug0 = model(x_aug0)
        # augmentation1
        x_aug1 = x_aug1.cuda()
        y_pred_aug1 = model(x_aug1)
        # augmentation2
        x_aug2 = x_aug2.cuda()
        y_pred_aug2 = model(x_aug2)

        y_pred_aug0_probas_log = torch.log_softmax(y_pred_aug0 / args.T, dim=-1)
        y_pred_aug1_probas_log = torch.log_softmax(y_pred_aug1 / args.T, dim=-1)
        y_pred_aug2_probas_log = torch.log_softmax(y_pred_aug2 / args.T, dim=-1)

        y_pred_aug0_probas = torch.softmax(y_pred_aug0 / args.T, dim=-1)
        y_pred_aug1_probas = torch.softmax(y_pred_aug1 / args.T, dim=-1)
        y_pred_aug2_probas = torch.softmax(y_pred_aug2 / args.T, dim=-1)

        # consist loss
        consist_loss0 = consistency_criterion(y_pred_aug0_probas_log, torch.tensor(confidence[index]).float().cuda())
        consist_loss1 = consistency_criterion(y_pred_aug1_probas_log, torch.tensor(confidence[index]).float().cuda())
        consist_loss2 = consistency_criterion(y_pred_aug2_probas_log, torch.tensor(confidence[index]).float().cuda())
        # supervised loss
        super_loss = - torch.mean(torch.sum(torch.log(1.0000001 - F.softmax(y_pred_aug0 / args.T, dim=1)) * (1 - part_y), dim=1) + \
                                    torch.sum(torch.log(1.0000001 - F.softmax(y_pred_aug1 / args.T, dim=1)) * (1 - part_y), dim=1) + \
                                        torch.sum(torch.log(1.0000001 - F.softmax(y_pred_aug2 / args.T, dim=1)) * (1 - part_y), dim=1))
        lam = args.lam

        # Unified loss
        final_loss = lam * (consist_loss0 + consist_loss1 + consist_loss2) + args.alpha * super_loss #+ args.beta * super_loss_2

        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        confidence_update(confidence, y_pred_aug0_probas, y_pred_aug1_probas, y_pred_aug2_probas, part_y, index)
        posterior_update(posterior, y_pred_aug0, index, T=args.T)

    return


def confidence_update(confidence, y_pred_aug0_probas, y_pred_aug1_probas, y_pred_aug2_probas, part_y, index):
    y_pred_aug0_probas = y_pred_aug0_probas.detach()
    y_pred_aug1_probas = y_pred_aug1_probas.detach()
    y_pred_aug2_probas = y_pred_aug2_probas.detach()

    revisedY0 = part_y.clone()

    revisedY0 = revisedY0 * torch.pow(y_pred_aug0_probas, 1 / (2 + 1)) \
                * torch.pow(y_pred_aug1_probas, 1 / (2 + 1)) \
                * torch.pow(y_pred_aug2_probas, 1 / (2 + 1))
    revisedY0 = revisedY0 / revisedY0.sum(dim=1).repeat(num_classes, 1).transpose(0, 1)

    confidence[index, :] = revisedY0.cpu().numpy()


def posterior_update(posterior, y_pred_aug0, index, T=1):
    y_pred_aug0_probas = torch.softmax(y_pred_aug0 / T, dim=-1)
    y_pred_aug0_probas = y_pred_aug0_probas.detach()

    posterior[index, :] = y_pred_aug0_probas.cpu()


def initialize_confidence_posterior(train_loader, model, confidence, posterior):
    model.eval()
    for i, (x_aug0, x_aug1, x_aug2, part_y, y, index) in enumerate(train_loader):

        part_y = part_y.float().cuda()
        # original samples with pre-processing
        x_aug0 = x_aug0.cuda()
        y_pred_aug0 = model(x_aug0)
        # augmentation1
        x_aug1 = x_aug1.cuda()
        y_pred_aug1 = model(x_aug1)
        # augmentation2
        x_aug2 = x_aug2.cuda()
        y_pred_aug2 = model(x_aug2)

        y_pred_aug0_probas = torch.softmax(y_pred_aug0, dim=-1)
        y_pred_aug1_probas = torch.softmax(y_pred_aug1, dim=-1)
        y_pred_aug2_probas = torch.softmax(y_pred_aug2, dim=-1)

        # update confidence
        confidence_update(confidence, y_pred_aug0_probas, y_pred_aug1_probas, y_pred_aug2_probas, part_y, index)
        posterior_update(posterior, y_pred_aug0, index)
    return 




def DPLL():
    global args
    set_seed(args.used_seed)
    if args.dataset == "cifar10":
        from utils.cifar10 import load_cifar10
        train_loader, valid_loader, test_loader, dim, K = load_cifar10(args.partial_rate, args.noisy_rate, batch_size=args.bs)
        channel = 3
    elif args.dataset == 'cifar100':
        from utils.cifar100 import load_cifar100
        train_loader, valid_loader, test_loader, dim, K = load_cifar100(args.partial_rate, args.noisy_rate, batch_size=args.bs)
        channel = 3
    elif args.dataset == 'fmnist':
        from utils.fmnist import load_fmnist
        train_loader, valid_loader, test_loader, dim, K = load_fmnist(args.partial_rate, args.noisy_rate, batch_size=args.bs)
        channel = 1
    elif args.dataset == 'kmnist':
        from utils.kmnist import load_kmnist
        train_loader, valid_loader, test_loader, dim, K = load_kmnist(args.partial_rate, args.noisy_rate, batch_size=args.bs)
        channel = 1
    else:
        assert "Unknown dataset"

    set_seed(args.seed)
    # load model
    if args.model == 'widenet':
        model = WideResNet(34, num_classes, widen_factor=10, dropRate=0.0)
    elif args.model == 'lenet':
        model = LeNet(out_dim=num_classes, in_channel=1, img_sz=28)
    elif args.model == 'resnet':
        model = resnet(depth=32, num_classes=num_classes)
    else:
        assert "Unknown model"
    model = model.cuda()
    initial_model = copy.deepcopy(model)
    # moving average
    from ema import EMA
    decay = args.decay
    ema = EMA(model, decay)
    ema.register()

    # criterion
    criterion = nn.CrossEntropyLoss().cuda()
    consistency_criterion = nn.KLDivLoss(reduction='batchmean').cuda()
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    # scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=-1)

    cudnn.benchmark = True
    # init confidence
    partial_Y = copy.deepcopy(train_loader.dataset.given_label_matrix)
    true_labels = copy.deepcopy(train_loader.dataset.true_labels).long()
    
    true_Y = torch.zeros_like(partial_Y)
    true_Y[torch.arange(len(true_labels)), true_labels] = 1.0

    prop = (partial_Y * true_Y).sum().item() / len(true_Y) * 100
    avg  = (partial_Y * (1 - true_Y)).sum().item() / len(true_Y)
    


    partial_Y_o = copy.deepcopy(train_loader.dataset.given_label_matrix)
    posterior = torch.ones_like(partial_Y)
    posterior = posterior / posterior.sum(dim=1, keepdim=True)

    confidence = copy.deepcopy(train_loader.dataset.given_label_matrix.numpy())
    confidence = confidence / confidence.sum(axis=1)[:, None]

    # load weight
    
    if args.load_weight:
        model.load_state_dict(torch.load(args.weight_path, map_location=device))
        initialize_confidence_posterior(train_loader, model, confidence, posterior)

    # earlystopping
    save_path = "checkpoints/{}/n={}_p={}/{}/".format(args.dataset, args.noisy_rate, args.partial_rate, args.lo)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    from earlystopping import EarlyStopping
    early = EarlyStopping(patience=50, path=os.path.join(save_path, "{}_n={}_p={}_lo={}_seed={}.pt".format(
        args.dataset, args.noisy_rate, args.partial_rate, args.lo, args.seed)))

    
    pre_correction_label_matrix = torch.zeros_like(partial_Y)
    correction_label_matrix     = copy.deepcopy(partial_Y)

    theta = args.theta
    inc = args.inc
    delta = args.delta
    dec = args.dec
    change_size = int(args.change_size) 
    times = int(args.times)
    update_interval = int(args.update_interval)

    valid_acc = accuracy_check(loader=valid_loader, model=model, device=device)
    test_acc  = accuracy_check(loader=test_loader,  model=model, device=device)
    print("Epoch {:>3d}, valid acc: {:.2f}, test acc: {:.2f}. ".format(0, valid_acc, test_acc))
    valid_acc_o = valid_acc
    best_val, best_test, best_epoch = valid_acc, test_acc, -1
    # Train loop
    
    for epoch in range(0, args.epochs):
        model.train()
        DPLL_train(args, train_loader, model, optimizer, epoch, consistency_criterion, confidence, posterior)
        # lr_step
        scheduler.step()
        model.eval()
        valid_acc = accuracy_check(loader=valid_loader, model=model, device=device)
        test_acc  = accuracy_check(loader=test_loader,  model=model, device=device)
        print("Epoch {:>3d}, valid acc: {:.2f}, test acc: {:.2f}. ".format(epoch+1, valid_acc, test_acc))
        early(valid_acc, model, epoch)
        if early.early_stop:
            break
        if valid_acc > best_val:
            best_val = valid_acc
            best_epoch = epoch
            best_test = test_acc

        if ((epoch >= args.warm_up) or args.load_weight) and epoch % update_interval == 0:
            ema.update()
            ema.apply_shadow()
            pre_correction_label_matrix = correction_label_matrix.clone()

            pred, _  = torch.max(posterior, dim=1, keepdim=True)
            tmp_diff = pred - posterior

            torch.set_printoptions(profile="full")
            
            pre_correction_label_matrix = copy.deepcopy(correction_label_matrix)
            non_change_matrix = copy.deepcopy(pre_correction_label_matrix)
            non_change_matrix[tmp_diff < theta] = 1
            non_change = torch.sum(torch.not_equal(pre_correction_label_matrix, non_change_matrix))
            if non_change > change_size:
                row, col = torch.where(tmp_diff < theta)
                idx_list = [ i for i in range(0, len(row))]
                random.shuffle(idx_list)
                non_row, non_col = row[idx_list[0:change_size]], col[idx_list[0:change_size]]
                non_change = change_size
                correction_label_matrix[non_row, non_col] = 1
            else:
                correction_label_matrix[tmp_diff < theta] = 1

            can_change_matrix = copy.deepcopy(pre_correction_label_matrix)
            can_change_matrix[tmp_diff > delta] = 0
            can_change = torch.sum(torch.not_equal(pre_correction_label_matrix, can_change_matrix))
            while can_change < non_change * times:
                # delta *= (1 - dec)
                delta = delta - dec
                can_change_matrix = copy.deepcopy(pre_correction_label_matrix)
                can_change_matrix[tmp_diff > delta] = 0
                can_change = torch.sum(torch.not_equal(pre_correction_label_matrix, can_change_matrix))
            # print('\t\tupdate the threshold delta : ' + str(delta))
            if can_change > change_size * times:
                row, col = torch.where(tmp_diff > delta)
                idx_list = [ i for i in range(0, len(row))]
                random.shuffle(idx_list)
                can_row, can_col = row[idx_list[0:change_size * times]], col[idx_list[0:change_size * times]]
                can_change = change_size * times
                correction_label_matrix[can_row, can_col] = 0
            else:
                correction_label_matrix[tmp_diff > delta] = 0

           

            for i in range(len(correction_label_matrix)):
                if correction_label_matrix[i].sum == 0:
                    correction_label_matrix[i] = copy.deepcopy(pre_correction_label_matrix[i])

            tmp_label_matrix = posterior * correction_label_matrix
            confidence = (tmp_label_matrix / tmp_label_matrix.sum(dim=1).repeat(tmp_label_matrix.size(1), 1).transpose(0, 1)).numpy()


            partial_Y = copy.deepcopy(correction_label_matrix)
            train_loader.dataset.given_label_matrix = copy.deepcopy(partial_Y)
            prop = (partial_Y * true_Y).sum().item() / len(true_Y) * 100
            avg  = (partial_Y * (1 - true_Y)).sum().item() / len(true_Y)
        

            change = torch.sum(torch.not_equal(pre_correction_label_matrix, correction_label_matrix))  

            if theta < 0.9 and delta > 0.1:
                if change < change_size:
                    theta += inc
                    delta -= dec

    print("Best Epoch {:>3d}, Best valid acc: {:.2f}, test acc: {:.2f}. ".format(best_epoch, best_val, best_test))


if __name__ == '__main__':
    DPLL()
