import pandas as pd
from PIL import Image
import platform

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchvision.models as models


class ImageDataset(Dataset):
    """
    A custom implementation of a torch.utils.data.Dataset meant to load images. This implementation is pulled from the PyTorch quickstart
    tutorial at https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
    """

    def __init__(self, annotations_file, root_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels.iloc[idx, 1] - 1
        img_path = self.img_labels.iloc[idx, 0]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class CustomEfficientNet(nn.Module):
    """
    A modified implementation of EfficientNetV2 Small (from the EfficientNetV2 paper, located at
    https://arxiv.org/abs/2104.00298)

    The only difference between this net and EfficientNetV2 Small is the classifier.
    There is a third layer in the classifier: a nn.Linear with 1000 inputs and class_num outputs.
    """

    def __init__(self, class_num):
        """
        Initializes a CustomEfficientNet, based on the EfficientNetV2 architecture.
        :param class_num: the number of classes in the set of possible predictions.
        """
        super(CustomEfficientNet, self).__init__()
        self.model = models.efficientnet_v2_s(weights='DEFAULT')
        self.model.classifier = nn.Sequential(
            self.model.classifier[0],
            self.model.classifier[1],
            nn.Linear(in_features=self.model.classifier[1].out_features, out_features=class_num)
        )

    def forward(self, x):
        """
        Returns the prediction of this CustomEfficientNet.
        :param x: a torch.Tensor
        :return: a torch.Tensor of length class_num
        """
        x = self.model(x)
        return x


class MeanCELoss(nn.CrossEntropyLoss):
    """
    Custom mean categorical cross-entropy loss with masking.

    :param int num_classes: The number of unique classes in the target labels.

    This class inherits from `torch.nn.CrossEntropyLoss` and modifies the forward pass to calculate
    the categorical cross-entropy loss for each sample, then masks the loss based on the true class labels.
    The masked losses are averaged across all classes to produce the final loss value.

    The masking works by creating a mask for each class and calculating the contribution of that class to the loss.
    The final loss is a weighted average of the individual class losses, where the weight is the proportion of the samples belonging to that class.
    """

    def __init__(self, num_classes):
        super(MeanCELoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        """
        Computes the mean categorical cross-entropy loss with masking.

        :param torch.Tensor y_pred: The predicted outputs from the model (logits) with shape (batch_size, num_classes).
        :param torch.Tensor y_true: The ground truth labels, one-hot encoded or class indices with shape (batch_size, num_classes) or (batch_size,).

        :returns: The computed masked mean categorical cross-entropy loss.
        :rtype: torch.Tensor

        This function calculates the categorical cross-entropy loss for each sample, then masks the loss based on the true class labels.
        The masked losses are averaged across all classes to produce the final loss value.

        The masking works by creating a mask for each class and calculating the contribution of that class to the loss.
        The final loss is a weighted average of the individual class losses, where the weight is the proportion of the samples belonging to that class.
        """
        c_ce = F.cross_entropy(y_pred, y_true, reduction='none')
        loss = 0.0

        for i in range(self.num_classes):
            inter_mask = (torch.argmax(y_true) == i).float()
            loss += (1.0 / self.num_classes) * torch.sum(inter_mask * c_ce) / (torch.sum(inter_mask) + 1e-15)

        return loss


def _masked_categorical_acc(y_true, y_pred, mask_value):
    """
    Computes the masked categorical accuracy.

    :param torch.Tensor y_true: The ground truth labels, one-hot encoded or class indices with shape (batch_size, num_classes) or (batch_size,).
    :param torch.Tensor y_pred: The predicted outputs from the model (logits) with shape (batch_size, num_classes).
    :param int mask_value: The number of classes to include in the accuracy calculation.

    :returns: The computed masked categorical accuracy.
    :rtype: torch.Tensor

    This function calculates the categorical accuracy for each sample, masking the accuracy calculations based on the provided mask value.
    The accuracy is calculated by determining the proportion of correct predictions out of the total predictions for the specified classes.

    The masking works by iterating over each class up to the mask value, creating a mask for each class, and calculating the contribution of that class to the overall accuracy.
    The final accuracy is a weighted average of the individual class accuracies.
    """
    sum_cols = torch.clamp(torch.sum(y_true, dim=0), 0.0, 1.0)
    num_in = torch.sum(sum_cols[:mask_value]) + 1e-7

    y_true = torch.argmax(y_true, dim=-1).float()
    y_pred = torch.argmax(y_pred, dim=-1).float()

    num_correct = 0
    for i in range(mask_value):
        mask_iter = (y_true == i).float()
        num_correct += (1.0 / num_in) * torch.sum(mask_iter * (y_true == y_pred).float()) / (
                    torch.sum(mask_iter) + 1e-7)

    return num_correct


def _l1_regularization(model, l1_lambda):
    """
    Calculates the sum of the absolute value of a model's parameters.
    :param torch.nn.Module model: a torch model
    :param float l1_lambda: the regularization coefficient
    :return: the l1 regularization of the model
    """
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return l1_lambda * l1_norm


def _l2_regularization(model, l2_lambda):
    """
    Calculates the sum of the squares of a model's parameters.
    :param torch.nn.Module model: a torch model
    :param float l2_lambda: the regularization coefficient
    :return: the l2 regularization of the model
    """
    l2_norm = sum(p.square().sum() for p in model.parameters())
    return l2_lambda * l2_norm


def _train_full_pass(train_dataloader, val_dataloader, model, loss_fn, optimizer, device, num_classes, l1_lam=None,
                     l2_lam=None):
    """
    Trains a PyTorch neural network.

    :param train_dataloader: the dataloader containing the training data
    :param val_dataloader: the dataloader containing the validation data
    :param nn.Module model: the neural network to be trained
    :param nn.Module loss_fn: the loss metric
    :param optimizer: the optimization algorithm
    :param device: the device to compute on
    :param int num_classes: the number of classes in the prediction set
    :param float l1_lam: the optional L1 regularization coefficient
    :param float l2_lam: the optional L2 regularization coefficient
    :return: 1 if model is done training, 0 otherwise
    """
    size = len(train_dataloader.dataset)
    model.train()
    model.to(device)
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        # compute error
        pred = model(X)
        if l1_lam and l2_lam:
            reg = _l1_regularization(model, l1_lam) + _l2_regularization(model, l2_lam)
        elif l1_lam:
            reg = _l1_regularization(model, l1_lam)
        elif l2_lam:
            reg = _l2_regularization(model, l2_lam)
        else:
            reg = 0
        loss = loss_fn(pred, y) + reg

        # Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    val_result = _validate(val_dataloader, model, loss_fn, device)
    if val_result == 1:
        return 1
    else:
        return 0


def _train(train_dataloader, val_dataloader, model, loss_fn, optimizer, device, num_classes, l1_lam=None, l2_lam=None):
    """
    Trains a PyTorch neural network.

    :param train_dataloader: the dataloader containing the training data
    :param val_dataloader: the dataloader containing the validation data
    :param nn.Module model: the neural network to be trained
    :param nn.Module loss_fn: the loss metric
    :param optimizer: the optimization algorithm
    :param device: the device to compute on
    :param int num_classes: the number of classes in the prediction set
    :param float l1_lam: the optional L1 regularization coefficient
    :param float l2_lam: the optional L2 regularization coefficient
    :return: 1 if model is done training, 0 otherwise
    """
    size = len(train_dataloader.dataset)
    model.train()
    model.to(device)
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        # compute error
        pred = model(X)
        if l1_lam and l2_lam:
            reg = _l1_regularization(model, l1_lam) + _l2_regularization(model, l2_lam)
        elif l1_lam:
            reg = _l1_regularization(model, l1_lam)
        elif l2_lam:
            reg = _l2_regularization(model, l2_lam)
        else:
            reg = 0
        loss = loss_fn(pred, y) + reg

        # Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

        if batch % 500 == 0:
            val_result = _validate(val_dataloader, model, loss_fn, device)
            if val_result == 1:
                return 1

    val_result = _validate(val_dataloader, model, loss_fn, device)
    if val_result == 1:
        return 1
    else:
        return 0


def _validate(dataloader, model, loss_fn, device):
    """
    Validates a given model on a validation set.
    :param dataloader: the dataloader containing the validation set
    :param model: the model to be validated
    :param loss_fn: the loss metric to validate with
    :param device: the device to compute on
    :return: 1 if the model has 100% accuracy during validation, 0 otherwise
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct, total, masked_acc_sum = 0, 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            # Calculate standard accuracy
            correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()

            # Calculate masked categorical accuracy
            one_hot_y = torch.nn.functional.one_hot(y, num_classes=pred.size(1)).float()
            masked_acc_sum += _masked_categorical_acc(one_hot_y, pred, pred.size(1)).item()

            total += len(y)
    test_loss /= num_batches
    accuracy = correct / total
    masked_accuracy = masked_acc_sum / num_batches

    print(f"Validation Error: \n Accuracy: {(100*accuracy):>0.1f}%, Masked Accuracy: {(100*masked_accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return 1 if accuracy == 1.0 or masked_accuracy == 1.0 else 0


def find_device():
    """Returns the most suitable computation device."""
    return "cuda" if torch.cuda.is_available() else "mps" if platform.system() == "Darwin" and torch.backends.mps.is_available() else "cpu"


def train_model(num_classes):
    """
    Training loop for the classification model. It automatically exits once the model has been fully trained
    :param num_classes: the number of classes to be predicted
    """
    model = CustomEfficientNet(num_classes)
    model.to(find_device())
    loss_fn = MeanCELoss(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=3.25e-5, weight_decay=1e-4)

    train_dataloader = DataLoader(ImageDataset("train_labels.csv", "data", models.EfficientNet_V2_S_Weights.DEFAULT.
                                               transforms()))
    validation_dataloader = DataLoader(ImageDataset('validation_labels.csv', 'data', models.EfficientNet_V2_M_Weights.
                                                    DEFAULT.transforms()))
    l1_reg = 2e-5

    epochs = 50
    _train_full_pass(train_dataloader, validation_dataloader, model, loss_fn, optimizer, device=find_device(),
           num_classes=num_classes, l1_lam=l1_reg)
    for t in range(epochs):
        if _train(train_dataloader, validation_dataloader, model, loss_fn, optimizer, device=find_device(),
                  num_classes=num_classes, l1_lam=l1_reg) == 1:
            torch.save(model.state_dict(), 'model.pth')
            break
