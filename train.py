import torch
import numpy as np
import argparse
import datetime
import random
import time
import sys
import os

from torchvision import datasets, transforms
from models import SimpleCNN, MyCNN, MyResNet
from utils import AverageMeter, Tee, Plotter, subdivide_dataset

# Config


def parse():
    parser = argparse.ArgumentParser()

    # --------------- General options ---------------
    parser.add_argument('-x', '--expID', type=str,
                        default='', help='experiment ID')
    parser.add_argument('--output-dir', type=str, default='./output/',
                        help='output directory (default: ./output/)')
    parser.add_argument('--seed', type=int, default=321,
                        help='random seed (default: 321)')
    # --------------- GPU options ---------------
    parser.add_argument('-j', '--workers', type=int, default=4,
                        help='number of workers (default: 4)')
    parser.add_argument(
        '--disable-cuda', action='store_true', help='disable CUDA')
    # --------------- Data options ---------------
    parser.add_argument('--dataset-dir', type=str, default='./datasets/',
                        help='dataset directory (default: ./datasets/)')
    parser.add_argument('--data-augmentation', type=bool,
                        default=False, help='data augmentation (default: False)')
    parser.add_argument('--val-size', type=float, default=0.1,
                        help='validation split size (default: 0.1)')
    # --------------- Model options ---------------
    parser.add_argument('--model', type=str, default='SimpleCNN',
                        help='network model (default: SimpleCNN)')
    # --------------- Training options ---------------
    parser.add_argument('-b', '--batch-size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('-e', '--epochs', type=int, default=40,
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--lr', '--learning-rate', type=float,
                        default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--wd', '--weight-decay', type=float, default=1e-6,
                        help='Weight decay regularization (default: 1e-6)')
    parser.add_argument('--scheduler-gamma', type=float, default=0.95,
                        help='learning rate decay factor (default: 0.95)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='number of batches before logging (default: 100)')

    return parser.parse_args()


def main(args):
    # DATASET
    print('\nLOADING DATASET.')

    train_data = datasets.FashionMNIST(
        args.dataset_dir, train=False, transform=args.transform, download=True)
    train_sampler, val_sampler = subdivide_dataset(
        train_data, val_size=args.val_size, shuffle=True)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, sampler=train_sampler, **args.kwargs)
    val_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, sampler=val_sampler, **args.kwargs)

    print('Dataset: {:d} samples (T: {:d} / V: {:d})'.format(
        len(train_data), len(train_sampler), len(val_sampler)))

    # MODEL
    print('\nLOADING MODEL.')

    model = eval(args.model)()
    model.to(args.device)

    print('Model: {} ({:.2f}M params)'.format(model._get_name(),
                                              sum(p.numel() for p in model.parameters()) / 1e6))

    # TRAINING
    print('\nTRAINING')
    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(filter(
        lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optim, gamma=args.scheduler_gamma)

    train_loss, val_loss, val_acc = [], [], []
    best_acc, i = 0, 0

    start_time = time.time()
    for epoch in range(args.epochs):

        # Train loop
        model.train()
        for i, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(args.device), labels.to(args.device)

            # Forward pass
            optim.zero_grad()
            output = model(images)
            loss = criterion(output, labels)

            # Backward pass
            loss.backward()
            optim.step()

            # Log
            train_loss.append(float(loss))
            if args.log_interval and i % args.log_interval == 0:
                mean_loss = np.mean(train_loss[-args.log_interval:])
                print("[Epoch {:02d} / Iter: {:04d}]: TRAIN loss: {:.3f}".format(
                    epoch+1, epoch*len(train_loader)+i, mean_loss))

        # Validation loop
        model.eval()
        val_loss_meter = AverageMeter()
        val_acc_meter = AverageMeter()
        for images, labels in val_loader:
            images, labels = images.to(args.device), labels.to(args.device)

            with torch.no_grad():
                output = model(images)
                val_loss_meter.update(
                    float(criterion(output, labels)), n=labels.size(0))
                pred = torch.max(output, 1)[1]
                val_acc_meter.update(
                    torch.sum(pred == labels).item() / labels.size(0), n=labels.size(0))

        val_loss.append(val_loss_meter.avg)
        val_acc.append(val_acc_meter.avg)
        scheduler.step()

        # Epoch logging
        if args.log_interval:
            print("[Epoch {:02d} / Iter: {:04d}]: EVAL. loss: {:.3f}".format(
                epoch+1, epoch*len(train_loader)+i, val_loss_meter.avg))
            print("[Epoch {:02d} / Iter: {:04d}]: EVAL. acc.: {:.3f}".format(
                epoch+1, epoch*len(train_loader)+i, val_acc_meter.avg))

            eta = int((time.time() - start_time) *
                      (args.epochs - (epoch + 1)) / ((epoch + 1)))
            print('ETA: {:s}'.format(str(datetime.timedelta(seconds=eta))))

        # Early Stopping
        is_best = val_acc_meter.avg > best_acc
        if is_best:
            best_acc = val_acc_meter.avg

            print('Saving at checkpoint...')
            state = {"train_loss": train_loss,
                     "val_loss": val_loss,
                     "val_acc": val_acc,
                     "best_acc": best_acc,
                     "epoch": epoch,
                     "model": model.state_dict(),
                     "args": args.__dict__}
            torch.save(state, os.path.join(
                args.output_dir, args.expID + "_checkpoint.pth"))

    # Plot training curves
    plotter = Plotter(os.path.join(args.output_dir, args.expID))
    plotter.plot_training(train_loss, val_loss, val_acc)


if __name__ == '__main__':
    args = parse()
    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(
        args.output_dir, args.expID + "_stdout.txt"), "w")

    print('SETUP.')
    print(args.__dict__)

    # Seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Cuda
    args.device = torch.device('cpu')
    args.kwargs = {}
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
        args.kwargs = {'num_workers': 4, 'pin_memory': True}
        print("CUDA is on!")

    # Data augmentation
    args.transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        # transforms.RandomRotation(18),
        # transforms.RandomCrop((28, 28), (2, 2)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]) if args.data_augmentation else transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    main(args)
