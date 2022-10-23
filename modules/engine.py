# modified version of my earlier work: https://github.com/Jollokim/Alexnet-edgemaps-vs-RGB/blob/main/engine.py
import torch
from torch.nn import functional as F


from torch.utils.data import DataLoader
from torch.optim import Optimizer

from modules.dataset import TinyImageNetDataset

from tqdm import tqdm


def train_one_epoch(model: torch.nn.Module, dataloader: DataLoader, optimizer: Optimizer,
                    epoch: int, device: torch.device):

    # setting the model in training mode
    model.train(True)

    n_batches = len(dataloader)
    batch_count = 1
    loss_over_epoch = 0

    print('Epoch', epoch)

    pbar = tqdm(dataloader)
    for batch in pbar:
        # Putting images and targets on given device
        X = batch['X'].to(device)
        y = batch['y'].to(device)

        # zeroing gradients before next pass through
        model.zero_grad()

        # passing images in batch through model
        outputs = model(X)

        # print(outputs.shape)
        # print(y.shape)

        # calculating loss and backpropagation the loss through the network
        loss = F.cross_entropy(outputs, y)
        
        loss.backward()

        # adjusting weight according to backpropagation
        optimizer.step()

        batch_count += 1

        # accumulating loss over complete epoch
        loss_over_epoch += loss.item()

        pbar.set_description(f'loss: {loss}')

    # mean loss for the epoch
    mean_loss = loss_over_epoch / n_batches

    return mean_loss


@torch.no_grad()
def test_accuracy(model: torch.nn.Module, dataset: TinyImageNetDataset, device: torch.device):
    # set model in evaluation mode. turns of dropout layers and other layers which only are used for training. same
    # as .train(False)
    model.eval()

    # how many correct classified images
    cnt = 0

    for i in tqdm(range(len(dataset))):
        # gets image an corresponding target
        X = dataset.__getitem__(i)['X'] 
        y = dataset.__getitem__(i)['y']

        # puts tensors onto devices
        X = X.to(device)
        y = y.to(device)

        # reshapes image to (1, C, H, W), model will only take images in batches, so here batch of one
        X = X.view(-1, 3, 64, 64)

        # pass image and get output vector
        output = model(X)

        # check argmax is same as target
        if torch.argmax(output) == torch.argmax(y):
            cnt += 1

    # number of correct predicted / total number of samples
    accuracy = cnt / len(dataset)

    return accuracy

@torch.no_grad()
def f1Score(model: torch.nn.Module, dataset: TinyImageNetDataset, device: torch.device):
    model.eval()

    truePositive = 0
    falsePositive = 0
    falseNegative = 0

    for i in tqdm(range(len(dataset))):
        # gets image an corresponding target
        X = dataset.__getitem__(i)['X']
        y = dataset.__getitem__(i)['y']

        # puts tensors onto devices
        X = X.to(device)
        y = y.to(device)

        # reshapes image to (1, C, H, W), model will only take images in batches, so here batch of one
        X = X.view(-1, 3, 64, 64)

        # pass image and get output vector
        output = model(X)

        # check argmax is same as target
        if torch.argmax(output) == torch.argmax(y):
            truePositive += 1
        elif torch.argmax(output) < torch.argmax(y):
            falseNegative += 1
        elif torch.argmax(output) > torch.argmax(y):
            falsePositive += 1

    precision = truePositive / (truePositive + falsePositive)
    recall = truePositive / (truePositive + falseNegative)

    return 2 * (precision * recall) / (precision + recall)
