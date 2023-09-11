import os
from pathlib import Path
from argparse import ArgumentParser
from tqdm.auto import tqdm
from functools import partial

import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms
import yaml

import models, models.classifier, data
from utils import losses, clone_state_dict

parser = ArgumentParser('Train Student Classifier')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--teacher', type=str.lower, default='resnet18')
parser.add_argument('--student', type=str.lower, default='resnet18')
parser.add_argument('--dataset', type=str.lower, default='cifar100')
parser.add_argument('--learning-rate', '-lr', type=float, default=1.0E-3)
parser.add_argument('--temperatures', '-T', type=float, nargs='+', default=[2.0, 3.0, 4.0, 5.0, 6.0])
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--tag', type=str, required=True)
CFG = parser.parse_args()

torch.cuda.set_device(CFG.gpu)

# Logging
LOG_DIR = Path('./log')
EXP_DIR = LOG_DIR.joinpath(CFG.tag)
if EXP_DIR.exists():
    answer = None
    while answer not in {'y', 'n'}:
        answer = input('Overwrite? [Y/n] ').strip()
        if len(answer) == 0:
            answer = 'y'
    
    if answer[0].lower() == 'y':
        os.system(f'rm -rf "{EXP_DIR}"')
    else:
        exit(0)
EXP_DIR.mkdir(parents=True)

CFG_FILENAME = EXP_DIR.joinpath('config.yaml')
CKPT_FILENAME = EXP_DIR.joinpath('ckpt.pt')

os.system(f'cp "{__file__}" "{EXP_DIR}"')
with open(CFG_FILENAME, 'w') as stream:
    yaml.dump(vars(CFG), stream=stream, indent=2)

writer = SummaryWriter(log_dir=LOG_DIR)

# Data & Loader
transform = transforms.Compose([
    transforms.Normalize(*data.get_mean_std(CFG.dataset)),
    transforms.Resize(models.classifier.get_input_size(CFG.model)),
    transforms.ToTensor(),
])
trainset = data.get_dataset(CFG.dataset, train=True, transform=transform)
validset = data.get_dataset(CFG.dataset, train=False, transform=transform)
classes = data.get_classes(CFG.dataset)
num_classes = len(classes)

loader_kwargs = dict(
    batch_size=CFG.batch_size,
    num_workers=CFG.num_workers,
)
train_loader = DataLoader(trainset, shuffle=True, **loader_kwargs)
valid_loader = DataLoader(validset, shuffle=False, **loader_kwargs)

# Model & Optimizer
net_teacher = models.classifier.get_teacher(CFG.teacher, CFG.dataset).cuda()
net_student = models.classifier.get_student(CFG.student, CFG.dataset).cuda()
optimizer = optim.Adam(net_student.parameters(), lr=CFG.learning_rate)
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1, CFG.epoch, eta_min=CFG.learning_rate * 0.1)

loss_kwargs = dict(
    # knowledge_distillation_loss
    temperatures=CFG.temperatures, 
    alpha=0.9,
    
    # multi_distillation_loss
    temperature=CFG.temperatures[0], 
    num_classes=num_classes, 
)
loss_fn = partial(losses.multi_distillation_loss, **loss_kwargs)

# Training Loop
with tqdm(range(1, CFG.epoch + 1), desc='EPOCH', position=1, leave=False) as epoch_bar:
    best_top1, best_epoch, best_state_dict = -1, -1, None
    lr_list, loss_list, top1_list = [], [], []
    
    for epoch in epoch_bar:
        with tqdm(train_loader, desc='TRAIN', position=2, leave=False) as train_bar:
            train_loss, train_correct, train_total = 0, 0, 0
            train_mean_loss, train_top1 = None, None
            
            for inputs, targets in train_loader:
                batch_size = inputs.size(0)
                inputs, targets = inputs.cuda(), targets.cuda()
                
                with torch.no_grad():
                    outs_teacher = net_teacher(inputs)
                outs_student = net_teacher(inputs)
                
                batch_loss = loss_fn(outs_student, outs_teacher)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    preds = torch.argmax(outs_student, dim=1)
                    train_loss += batch_loss.item() * batch_size
                    train_correct += (preds == targets).sum().item()
                    train_total += batch_size
                    
                train_mean_loss = train_loss / train_total
                train_top1 = train_correct / train_total * 100
                train_bar.set_postfix_str(f'loss={train_loss:.2f} | top1={train_top1:.2f}%')
            
            writer.add_scalars('train', {
                'loss': train_mean_loss,
                'top1': train_top1,
            }, global_step=epoch)
            
        with tqdm(valid_loader, desc='VALID', position=2, leave=False) as valid_bar, torch.no_grad():
            valid_loss, valid_correct, valid_total = 0, 0, 0
            valid_mean_loss, valid_top1 = None, None
            
            for inputs, targets in valid_loader:
                batch_size = inputs.size(0)
                inputs, targets = inputs.cuda(), targets.cuda()
                outs_teacher = net_teacher(inputs)
                outs_student = net_teacher(inputs)
                preds = torch.argmax(outs_student, dim=1)
                
                batch_loss = loss_fn(outs_student, outs_teacher)
                valid_loss += batch_loss.item() * batch_size
                valid_correct += (preds == targets).sum().item()
                valid_total += batch_size
                
                valid_mean_loss = valid_loss / valid_total
                valid_top1 = valid_correct / valid_total * 100
                valid_bar.set_postfix_str(f'loss={valid_loss:.2f} | top1={valid_top1:.2f}%')

            writer.add_scalars('valid', {
                'loss': valid_mean_loss,
                'top1': valid_top1,
            }, global_step=epoch)
        
        lr_list.append(optimizer.param_groups[0]['lr'])
        loss_list.append(train_mean_loss)
        top1_list.append(valid_top1)
        if valid_top1 >= best_top1:
            best_top1 = valid_top1
            best_epoch = epoch
            best_state_dict = clone_state_dict(net_student)
        
        writer.add_scalars('valid', {
            'best_loss': loss_list[best_epoch - 1],
            'best_top1': best_top1,
        }, global_step=epoch)
        
        torch.save({
            'cfg': CFG,
            'last_epoch': epoch,
            'best_epoch': best_epoch,
            'lr_list': lr_list,
            'loss_list': loss_list,
            'top1_list': top1_list,
            'best_state_dict': best_state_dict,
            'last_state_dict': clone_state_dict(net_student),
        }, CKPT_FILENAME)
        
        writer.add_scalars('epoch', {
            'epoch': epoch,
            'learning rate': lr_list[-1],
        }, global_step=epoch)

best_loss = loss_list[best_epoch - 1]
print(f'Best: loss={best_loss:.2f}, top1={best_top1:.2f}% @ {best_epoch}-th epoch.')
