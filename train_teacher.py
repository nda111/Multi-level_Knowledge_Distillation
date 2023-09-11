import os
from argparse import ArgumentParser
from tqdm.auto import tqdm

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from torchvision import transforms

import models, models.classifier, data
from utils import clone_state_dict

parser = ArgumentParser('Train Teacher Classifier')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--model', type=str.lower, default='resnet18')
parser.add_argument('--dataset', type=str.lower, default='cifar100')
parser.add_argument('--learning-rate', '-lr', type=float, default=1.0E-3)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--num-workers', type=int, default=4)
CFG = parser.parse_args()

torch.cuda.set_device(CFG.gpu)
CKPT_DIR, CKPT_FILENAME = models.get_checkpoint_filename(CFG.model, CFG.dataset)
CKPT_FILENAME = os.path.join(CKPT_DIR, CKPT_FILENAME)
os.system(f'mkdir -p "{CKPT_DIR}"')

# Data & Loader
transform = transforms.Compose([
    transforms.Normalize(*data.get_mean_std(CFG.dataset)),
    transforms.Resize(models.classifier.get_input_size(CFG.model)),
    transforms.ToTensor(),
])
trainset = data.get_dataset(CFG.dataset, train=True, transform=transform)
validset = data.get_dataset(CFG.dataset, train=False, transform=transform)

loader_kwargs = dict(
    batch_size=CFG.batch_size,
    num_workers=CFG.num_workers,
)
train_loader = DataLoader(trainset, shuffle=True, **loader_kwargs)
valid_loader = DataLoader(validset, shuffle=False, **loader_kwargs)

# Model & Optimizer
net_teacher = models.classifier.get_student(CFG.model, CFG.dataset, pretrained=True).cuda()
optimizer = optim.Adam(net_teacher.get_classifier().parameters(), lr=CFG.learning_rate)
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1, CFG.epoch, eta_min=CFG.learning_rate * 0.1)
ce_loss = nn.CrossEntropyLoss().cuda()

# Training Loop
with tqdm(range(1, CFG.epoch + 1), desc='EPOCH', position=1, leave=False) as epoch_bar:
    best_top1, best_epoch, best_state_dict = -1, -1, None
    lr_list, loss_list, top1_list = [], [], []
    
    for epoch in epoch_bar:
        with tqdm(train_loader, desc='TRAIN', position=2, leave=False) as train_bar:
            train_loss, train_correct, train_total = 0, 0, 0
            train_mean_loss, train_top1 = None, None
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outs = net_teacher(inputs)
                
                batch_loss = ce_loss(outs, targets)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    batch_size = inputs.size(0)
                    preds = torch.argmax(outs, dim=1)
                    train_loss += batch_loss.item() * batch_size
                    train_correct += (preds == targets).sum().item()
                    train_total += batch_size
                    
                train_mean_loss = train_loss / train_total
                train_top1 = train_correct / train_total * 100
                train_bar.set_postfix_str(f'loss={train_loss:.2f} | top1={train_top1:.2f}%')
            
        with tqdm(valid_loader, desc='VALID', position=2, leave=False) as valid_bar, torch.no_grad():
            valid_loss, valid_correct, valid_total = 0, 0, 0
            valid_mean_loss, valid_top1 = None, None
            
            for inputs, targets in valid_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outs = net_teacher(inputs)
                preds = torch.argmax(outs, dim=1)
                batch_loss = ce_loss(outs, targets)
                
                batch_size = inputs.size(0)
                valid_loss += batch_loss.item() * batch_size
                valid_correct += (preds == targets).sum().item()
                valid_total += batch_size
                
                valid_mean_loss = valid_loss / valid_total
                valid_top1 = valid_correct / valid_total * 100
                valid_bar.set_postfix_str(f'loss={valid_loss:.2f} | top1={valid_top1:.2f}%')
        
        lr_list.append(optimizer.param_groups[0]['lr'])
        loss_list.append(train_mean_loss)
        top1_list.append(valid_top1)
        if valid_top1 >= best_top1:
            best_top1 = valid_top1
            best_epoch = epoch
            best_state_dict = clone_state_dict(net_teacher.get_classifier())
        
        torch.save({
            'cfg': CFG,
            'last_epoch': epoch,
            'best_epoch': best_epoch,
            'lr_list': lr_list,
            'loss_list': loss_list,
            'top1_list': top1_list,
            'best_state_dict': best_state_dict,
            'last_state_dict': clone_state_dict(net_teacher.get_classifier()),
        }, CKPT_FILENAME)

best_loss = loss_list[best_epoch - 1]
print(f'Best: loss={best_loss:.2f}, top1={best_top1:.2f}% @ {best_epoch}-th epoch.')
