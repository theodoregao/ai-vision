from __future__ import print_function, division

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torchvision import datasets, transforms

from net import SimpleConv3

writer = SummaryWriter('logs')


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_accs = 0.0
            number_batch = 0

            for data in dataloaders[phase]:
                inputs, labels = data
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data.item()
                running_accs += torch.sum(preds == labels).item()
                number_batch += 1

            epoch_loss = running_loss / number_batch
            epoch_accs = running_accs / dataset_sizes[phase]

            if phase == 'train':
                writer.add_scalar('data/train-loss', epoch_loss, epoch)
                writer.add_scalar('data/train-acc', epoch_accs, epoch)
            else:
                writer.add_scalar('data/val-loss', epoch_loss, epoch)
                writer.add_scalar('data/val-acc', epoch_accs, epoch)

            print('{} Loss: {:.4f}, Acc: {:.4f}'.format(phase, epoch_loss, epoch_accs))
    writer.close()
    return model


if __name__ == '__main__':
    image_size = 64
    crop_size = 48
    nclasses = 4
    model = SimpleConv3(nclasses)
    data_dir = 'data'

    if not os.path.exists('models'):
        os.mkdir('models')

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    print(model)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'val': transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4) for x
                   in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    step_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)

    model = train_model(model=model, criterion=criterion, optimizer=optimizer_ft, scheduler=step_lr_scheduler,
                        num_epochs=300)

    torch.save(model.state_dict(), 'models/model.pt')
