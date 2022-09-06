import transformers
from transformers import ViTModel, ViTConfig, ViTFeatureExtractor, ViTModel, ViTForImageClassification
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models import utils, ViT_encoder
from datasets import coco

from configuration import *
from PIL import Image, ImageOps
import os
from os.path import join
import numpy as np
import math
import sys
import tqdm

import torch.nn.functional as F


def criteria(loss_fun, output, context, dev, weight=(0.9, 0.69, 0.49)):
    losses = 0
    for i, key in enumerate(output):
        #print(output[key], context[key])
        #print(output[key].shape, context[key].shape)
        losses += weight[i] / sum(weight) * loss_fun(output[key], context[key].to(dev))
    return losses


def get_accuracy(pred, label):
    label['where'] = (F.one_hot(label['where'], num_classes=3)).long()
    label['when'] = (F.one_hot(label['when'], num_classes=3)).long()
    label['whom'] = (F.one_hot(label['whom'], num_classes=3)).long()

    pred_where = pred['where'].detach().max(dim=1)[1].cpu().numpy()
    pred_when = pred['when'].detach().max(dim=1)[1].cpu().numpy()
    pred_whom = pred['whom'].detach().max(dim=1)[1].cpu().numpy()
    gt_where = label['where'].long().max(dim=1)[1].cpu().numpy()
    gt_when = label['when'].long().max(dim=1)[1].cpu().numpy()
    gt_whom = label['whom'].long().max(dim=1)[1].cpu().numpy()

    acc0 = np.sum(pred_where == gt_where).astype(float)/pred_where.shape[0]
    acc1 = np.sum(pred_when == gt_when).astype(float)/pred_when.shape[0]
    acc2 = np.sum(pred_whom == gt_whom).astype(float)/pred_whom.shape[0]

    return {'where': acc0, 'when': acc1, 'whom': acc2}


def train_an_epoch(config, model, loss_func, data_loader,
                    optimizer, device, epoch, max_norm):
    model.train()
    #loss_func.train()
    epoch_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for i, tuples in enumerate(data_loader):
            inputs, contexts = tuples[6], tuples[7]
            '''
            contexts['where'] = (F.one_hot(contexts['where'], num_classes=3)).long()
            contexts['when'] = (F.one_hot(contexts['when'], num_classes=3)).long()
            contexts['whom'] = (F.one_hot(contexts['whom'], num_classes=3)).long()
            '''

            inputs['pixel_values'] = inputs['pixel_values'].squeeze(1).to(device)

            outputs = model(inputs['pixel_values'])
            #print(outputs, contexts)
            loss = criteria(loss_func, outputs, contexts, device, config.vit_weights)
            loss_value = loss.item()
            epoch_loss += loss_value

            if not math.isfinite(loss_value):
                print(f'Loss is {loss_value}, stopping training')
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update(1)

    return epoch_loss / total


@torch.no_grad()
def evaluate(config, model, loss_func, data_loader, device):
    model.eval()
    loss_func.eval()

    validation_loss = 0.0
    total = len(data_loader)

    count = 0
    acc_where = 0.0
    acc_when = 0.0
    acc_whom = 0.0
    with tqdm.tqdm(total=total) as pbar:
        for i, tuples in enumerate(data_loader):
            inputs, contexts = tuples[6], tuples[7]

            inputs['pixel_values'] = inputs['pixel_values'].squeeze(1).to(device)

            outputs = model(inputs['pixel_values'])

            loss = criteria(loss_func, outputs, contexts, device)
            loss_value = loss.item()
            validation_loss += loss_value

            acc = get_accuracy(outputs, contexts)
            acc_where += acc['where']
            acc_when += acc['when']
            acc_whom += acc['whom']
            count += 1

            pbar.update(1)

    print(f'Accuracy (where, when, whom): {acc_where / count}, {acc_when / count}, {acc_whom / count}')

    return validation_loss / total, acc_where / count, acc_when / count, acc_whom / count


def main(config):
    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    for idx, dev in enumerate(available_gpus):
        print("Available GPU-{} name: {}".format(idx, dev))
    device = torch.device(config.device)
    print(f'Initializing Device: {device}')

    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ViT_encoder.build_ViTEncoder(config)
    model.to(device)

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")

    param_dicts = [
        {"params": [p for n, p in model.named_parameters(
        ) if "body" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "body" in n and p.requires_grad],
            "lr": config.vit_body_lr,
        },
    ]
    '''
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)
    exit()
    '''
    optimizer = torch.optim.AdamW(param_dicts, lr=config.vit_lr, weight_decay=config.vit_weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20)

    # Weighted Loss Func
    class_weights = torch.tensor([0.4, 0.4, 0.2]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Dataset
    if config.modality == 'ego':
        dataset_train = coco.build_dataset_egocap(config, mode='training')
        dataset_val = coco.build_dataset_egocap(config, mode='validation')
    else:
        raise TypeError("Input Modality not supported!")
    print(f"Train: {len(dataset_train)}")
    print(f"Valid: {len(dataset_val)}")

    # Sampler


    # DataLoader
    data_loader_train = DataLoader(dataset_train, config.batch_size,
                                   drop_last=False, num_workers=config.num_workers)
    data_loader_val = DataLoader(dataset_val, config.batch_size,
                                 drop_last=False, num_workers=config.num_workers)

    # Free GPU memory n allow growth
    torch.cuda.empty_cache()

    save_dir = '/mnt/datasets/COCO/vit_checks'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    best_acc = 0.0

    print("Start Training..")
    for epoch in range(config.start_epoch, config.epochs):
        print(f"Epoch: {epoch}")
        epoch_loss = train_an_epoch(config, model, criterion, data_loader_train,
                                    optimizer, device, epoch, config.clip_max_norm)
        lr_scheduler.step()
        print(f"Training Loss: {epoch_loss}")

        validation_loss, acc_where, acc_when, acc_whom = evaluate(config, model, criterion, data_loader_val, device)
        print(f"Validation Loss: {validation_loss}")

        # magic numbers?
        avg_acc = (0.24 * acc_where + 0.38 * acc_when + 0.38 * acc_whom)
        if best_acc < avg_acc:
            best_acc = avg_acc
            print('Saving model ...')
            model_name = 'ctx_vit-accwhere{}_accwhen{}_accwhom{}.pth'.format(
                round(acc_where * 100),
                round(acc_when * 100),
                round(acc_whom * 100)
            )  # 'ViT-epoch{}_loss{}.pth'.format(epoch, round(validation_loss * 100))
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'best_where': acc_where,
                'best_when': acc_when,
                'best_whom': acc_whom,
            }, join(save_dir, model_name))
        print()


if __name__ == "__main__":
    config = ConfigEgo()
    main(config)
