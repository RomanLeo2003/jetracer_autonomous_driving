import torch
import torchvision
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from prepare_dataset import DonkeyDataset
import random
import numpy as np
import os
import tqdm

from collections import namedtuple
from typing import NamedTuple, List

model = torchvision.models.vit_l_16(weights=torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
model.heads = torch.nn.Linear(1024, 2)  # можно заморозить некоторые слои, надо экспериментировать


# model = torchvision.models.swin_v2_b(weights='IMAGENET1K_V1') # трансформер полегче в 3 раза
# model.head = torch.nn.Linear(1024, 2)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision='bf16')
batch_size = 16
to_resize = 224

train_transforms = torchvision.transform.Compose([])
test_transforms = torchvision.transform.Compose([])

train_dataset = DonkeyDataset()
test_dataset = DonkeyDataset()

train_dataloader = DataLoader()
test_dataloader = DataLoader()


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim,
    criterion: torch.nn.modules.loss._Loss,
    loader: torch.utils.data.DataLoader,
    device: torch.device
):
    acc_loss = 0
    total = len(loader.dataset)
    # model.to(device)
    model.train()
    for data, target in tqdm.tqdm(loader):
      with accelerator.accumulate(model): # для имитации большого размера батча (полезно для трансформеров)
        # data = data.to(device)
        # target = target.to(device)
        pred = torch.nn.functional.softmax(model(data), dim=-1)
        loss = criterion(pred, target)
        # scaler.scale(loss).backward()
        # scaler.unscale_(optimizer)
        # scaler.step(optimizer)
        # loss.backward()
        accelerator.backward(loss) # вместо loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        acc_loss += loss.item()

    return acc_loss / total

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


EvalOut = namedtuple("EvalOut", ['loss', 'MSE'])

def eval_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    loader: torch.utils.data.DataLoader,
    device: torch.device
):
    mses = []
    total = len(loader.dataset)
    acc_loss = 0
    model.eval()
    # model.to(device)
    with torch.inference_mode():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            pred = model(data)
            loss = criterion(pred, target)
            acc_loss += loss.item()
            mses.append((pred - target) ** 2)

    return EvalOut(loss=(acc_loss / total), MSE=(sum(mses) / total))


class TrainOut(NamedTuple):
    train_loss: List[float]
    eval_loss: List[float]
    eval_accuracy: List[float]


def train(
    model: torch.nn.Module,
    optimizer: torch.optim,
    criterion: torch.nn.modules.loss._Loss,
    sheduler: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 10
):
    train_loss = []
    eval_loss = []
    eval_MSE = []
    model.to(device)
    for i in range(epochs):
        print(f"Epoch - {i}:")
        if (train_loader != None):
            print("Train...\n")
            train_loss.append(train_epoch(model, optimizer, criterion, train_loader, device))
        print("Validation...\n")
        eval_out = eval_epoch(model, criterion, val_loader, device)
        eval_loss.append(eval_out.loss)
        eval_MSE.append(eval_out.MSE)
        print(f'Validation MSE: {eval_out.MSE}')
        sheduler.step()
        print('lr: ', get_lr(optimizer))
        if i > 1 and eval_MSE[i] == min(eval_MSE):
          unwrapped_model = accelerator.unwrap_model(model)
          accelerator.save({
                "model": model.state_dict(),
                "optimizer": optimizer.optimizer.state_dict() # optimizer is an AcceleratedOptimizer object
            }, "bundle.pth")
          torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoint_{i}.pth')

    return TrainOut(train_loss = train_loss,
                    eval_loss = eval_loss,
                    eval_accuracy = eval_MSE)



criterion = torch.nn.RMSE()
optimizer = torch.optim.AdamW(model.parameters(), lr=15e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
epochs = 100
model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, scheduler
)

tr_tuple = train(model, optimizer, criterion, scheduler, train_dataloader, test_dataloader, device, epochs)