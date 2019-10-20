import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import os


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum / self.count)


class Tee(object):
    def __init__(self, name="stdout.txt", mode="a"):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()


class Plotter(object):
    def __init__(self, output_name="./"):
        self.output_name = output_name

    def plot_training(self, train_loss=[], val_loss=[], val_acc=[]):
        fig, ax1 = plt.subplots()

        ratio = int(len(train_loss) / len(val_loss))
        x = np.arange(1, len(val_acc) + 1)

        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss', color='tab:red')
        ax1.set_yscale('log')
        ax1.plot(x, train_loss[ratio-1::ratio],
                 color='tab:red', label="train loss", marker=",")
        ax1.scatter(x, val_loss, color='tab:orange',
                    label="eval. loss", marker='x')
        # ax1.tick_params(axis='y', color='tab:red')

        ax2 = ax1.twinx()

        ax2.set_ylabel('accuracy', color='tab:blue')
        ax2.plot(x, val_acc, color='tab:blue', label="eval. acc.", marker=".")
        # ax2.tick_params(axis='y', color='tab:blue')

        fig.legend(loc='lower left')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.savefig(os.path.join(self.output_name+"_training.png"))
        # plt.show()


def subdivide_dataset(dataset, val_size=0.2, shuffle=True):
    indices = list(range(len(dataset)))
    if shuffle:
        np.random.shuffle(indices)

    i = int(len(dataset)*val_size)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[i:])
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:i])

    return train_sampler, val_sampler


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    """Source: https://github.com/bearpaw/pytorch-classification/blob/cc9106d598ff1fe375cc030873ceacfea0499d77/utils/eval.py """
    maxk = max(topk)
    batch_size = target.size(0)

    pred = output.topk(maxk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
