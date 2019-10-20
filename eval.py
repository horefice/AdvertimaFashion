import torch
import numpy as np
import argparse

from torchvision import datasets, transforms
from models import SimpleCNN, MyCNN, MyResNet
from utils import AverageMeter, Plotter, accuracy


# Config
def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, default='',
                        help='load checkpoint from path')
    parser.add_argument('--dataset-dir', type=str, default='./datasets/',
                        help='dataset directory (default: ./datasets/)')

    return parser.parse_args()


def main(args):
    # CHECKPOINT
    print('\nLOADING CHECKPOINT.')
    try:
        checkpoint = torch.load(
            args.checkpoint, map_location=torch.device('cpu'))
    except Exception as e:
        print('No checkpoint found. Exiting.')
        return
    else:
        print('Success!')

    # DATASET
    print('\nLOADING DATASET.')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_data = datasets.FashionMNIST(
        args.dataset_dir, train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=256, shuffle=False)
    print('Dataset: {:d} samples'.format(len(test_data)))

    # MODEL
    print('\nLOADING MODEL.')
    model = eval(checkpoint['args']['model'])()
    model.load_state_dict(checkpoint['model'])
    model.to(args.device)
    model.eval()
    print('Model: {} ({:.2f}M params)'.format(model._get_name(),
                                              sum(p.numel() for p in model.parameters()) / 1e6))

    # Evaluation
    print('\nEVALUATION')
    top1, top5 = AverageMeter(), AverageMeter()
    for images, labels in test_loader:
        images, labels = images.to(args.device), labels.to(args.device)

        with torch.no_grad():
            output = model(images)
            topk = accuracy(output, labels, (1, 5))
            top1.update(topk[0], n=labels.size(0))
            top5.update(topk[1], n=labels.size(0))

    print('TEST top-1 acc.: {:.2f}%'.format(top1.avg))
    print('TEST top-5 acc.: {:.2f}%'.format(top5.avg))


if __name__ == '__main__':
    args = parse()
    args.device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    main(args)
