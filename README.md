# AdvertimaFashion

This repo contains the code for the Advertima task on the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).

### Models
Three models were trained and evaluated with the provided scripts (train.py and eval.py) using PyTorch. The proposed models are CNNs with different architectures and were coded in a single, separate file (models.py).

The SimpleCNN is the simplest provided model and serves as a baseline. The MyCNN is a considerable improvement over our baseline and was trained with and without data augmentation. At last, the MyResNet imports blocks of layers from the ResNet18 for a deeper network.

* SimpleCNN: conv - relu - maxpool - conv - relu - maxpool - fc
* MyCNN: conv - bn - relu - maxpool - conv - bn - relu - maxpool - dropout - fc - relu - fc
* MyResNet: conv - bn - relu - block1 (resnet18) - block2 (resnet18) - block3 (resnet18) - adaptiveavgpool - fc

### Training pipeline
The training data was split into training (90%) and validation (10%). The loss function we're optimizing for was the Cross-Entropy Loss. In addition, ADAM was selected as optimizer, with a learning rate scheduler with exponential decay.  Moreover, early stop was used, saving the model with best validation accuracy.

The following data augmentation techniques were implemented:
* RandomHorizontalFlip: 50% chance of horizontal flip
* RandomColorJitter: +- 5% changes on contrast and brightness

The training script outputs the model checkpoint (.pth), the stdout into a .txt file and the training curves plot with loss and accuracy, all files named after the experimentID.

### Training
```
usage: train.py [-h] [-x EXPID] [--output-dir OUTPUT_DIR] [--seed SEED]
                [-j WORKERS] [--disable-cuda] [--dataset-dir DATASET_DIR]
                [--data-augmentation DATA_AUGMENTATION] [--val-size VAL_SIZE]
                [--model MODEL] [-b BATCH_SIZE] [-e EPOCHS] [--lr LR]
                [--wd WD] [--scheduler-gamma SCHEDULER_GAMMA]
                [--log-interval LOG_INTERVAL]

optional arguments:
  -h, --help            show this help message and exit
  -x EXPID, --expID EXPID
                        experiment ID
  --output-dir OUTPUT_DIR
                        output directory (default: ./output/)
  --seed SEED           random seed (default: 321)
  -j WORKERS, --workers WORKERS
                        number of workers (default: 4)
  --disable-cuda        disable CUDA
  --dataset-dir DATASET_DIR
                        dataset directory (default: ./datasets/)
  --data-augmentation DATA_AUGMENTATION
                        data augmentation (default: False)
  --val-size VAL_SIZE   validation split size (default: 0.1)
  --model MODEL         network model (default: SimpleCNN)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        input batch size for training (default: 128)
  -e EPOCHS, --epochs EPOCHS
                        number of epochs to train (default: 40)
  --lr LR, --learning-rate LR
                        learning rate (default: 1e-3)
  --wd WD, --weight-decay WD
                        Weight decay regularization (default: 1e-6)
  --scheduler-gamma SCHEDULER_GAMMA
                        learning rate decay factor (default: 0.95)
  --log-interval LOG_INTERVAL
                        number of batches before logging (default: 100)
```
### Evaluation
```
usage: eval.py [-h] [--checkpoint CHECKPOINT] [--dataset-dir DATASET_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
                        load checkpoint from path
  --dataset-dir DATASET_DIR
                        dataset directory (default: ./datasets/)
```

### Results
| | SimpleCNN | MyCNN | MyCNN + data aug. | MyResNet |
| --- | --- | --- | --- | --- |
| Parameters (M) | **0.02**  | 0.42 | 0.42 | 2.76 |
| Top-1 Accuracy (%) | 92.15 | 93.11 | 93.17 | **93.58** |
| Top-5 Accuracy (%) | 99.91 | 99.91 | **99.96** | 99.71 |

### License
MIT
