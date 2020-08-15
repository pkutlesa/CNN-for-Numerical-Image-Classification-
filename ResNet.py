import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from tqdm.autonotebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
import time
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from skimage.transform import resize


# -------------------------------------------------- ResNet ---------------------------------------------------------- #
class MnistResNet(ResNet):
    def __init__(self):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)

    def forward(self, x):
        return torch.softmax(super(MnistResNet, self).forward(x), dim=-1)

    
def per_image_normalization(X, constant=10.0, copy=True):
    if copy:
        X_res = X.copy()
    else:
        X_res = X

    means = np.mean(X, axis=1)
    variances = np.var(X, axis=1) + constant
    X_res = (X_res.T - means).T
    X_res = (X_res.T / np.sqrt(variances)).T
    return X_res

# ------------------------------------------------ Load Data --------------------------------------------------------- #
def get_data_loaders(train_batch_size, val_batch_size):
    images = pd.read_pickle('train_max_x')
    train = pd.read_csv('train_max_y.csv')

    # reshape images and normalize
    images = images.astype('float32')
    images /= 255
    images = images.reshape(images.shape[0], 1, 128, 128)

    # store features and targets as numpy arrays
    features_numpy = np.array(images)
    targets_numpy = train['Label'].values
    
    # split training and validation data
    features_train, features_test, targets_train, targets_test = train_test_split(
        features_numpy, targets_numpy, test_size=0.2, random_state=42)

    # create feature and targets tensor for train set.
    featuresTrain = torch.from_numpy(features_train)
    targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)

    # create feature and targets tensor for validation set.
    featuresTest = torch.from_numpy(features_test)
    targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)

    # Pytorch train, validation and test sets
    train = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
    val = torch.utils.data.TensorDataset(featuresTest, targetsTest)

    # data loader
    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val, batch_size=val_batch_size, shuffle=False)

    return train_loader, val_loader


# --------------------------------------------- Helper Functions ----------------------------------------------------- #
def calculate_metric(metric_fn, true_y, pred_y):
    if "average" in inspect.getfullargspec(metric_fn).args:
        return metric_fn(true_y, pred_y, average="macro")
    else:
        return metric_fn(true_y, pred_y)


def print_scores(p, r, f1, a, batch_size):
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores) / batch_size:.4f}")


# ----------------------------------------------- Train Model -------------------------------------------------------- #
def train_model(model, epochs):
    start_ts = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    train_loader, val_loader = get_data_loaders(128, 128)


    losses = []
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters())

    batches = len(train_loader)
    val_batches = len(val_loader)

    # training loop + eval loop
    for epoch in range(epochs):
        total_loss = 0
        progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)
        model.train()

        for i, data in progress:
            X, y = data[0].to(device), data[1].to(device)

            model.zero_grad()
            outputs = model(X)
            loss = loss_function(outputs, y)

            loss.backward()
            optimizer.step()
            current_loss = loss.item()
            total_loss += current_loss
            progress.set_description("Loss: {:.4f}".format(total_loss / (i + 1)))

        torch.cuda.empty_cache()

        val_losses = 0
        precision, recall, f1, accuracy = [], [], [], []

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                X, y = data[0].to(device), data[1].to(device)
                outputs = model(X)
                val_losses += loss_function(outputs, y)

                predicted_classes = torch.max(outputs, 1)[1]

                for acc, metric in zip((precision, recall, f1, accuracy),
                                       (precision_score, recall_score, f1_score, accuracy_score)):
                    acc.append(calculate_metric(metric, y.cpu(), predicted_classes.cpu()))

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"training loss: {total_loss / batches}, "
              f"validation loss: {val_losses / val_batches}")
        print_scores(precision, recall, f1, accuracy, val_batches)
        losses.append(total_loss / batches)
    print(losses)
    print(f"Training time: {time.time() - start_ts}s")

# ----------------------------------------------- Test Set Loader ------------------------------------------------------ #
def get_test_loader():
    test_images = pd.read_pickle('test_max_x')

    # reshape images and normalize
    test_images = test_images.astype('float32')
    test_images /= 255
    test_images = test_images.reshape(test_images.shape[0], 1, 128, 128)

    # store features as numpy arrays
    features_testset = np.array(test_images)

    # create feature tensor for test set.
    features_testset = torch.from_numpy(features_testset)

    # Pytorch test set
    test = torch.utils.data.TensorDataset(features_testset)

    # data loader
    test_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)

    return test_loader

# ----------------------------------------------- Main Function ------------------------------------------------------ #
model = MnistResNet()
train_model(model, 20)


# Make Predictions
test_loader = get_test_loader()
test_batches = len(test_loader)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
preds = []
model.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader):
        X = data[0].to(device)
        outputs = model(X)
        predicted_classes = torch.max(outputs, 1)[1]
        preds.append(predicted_classes.cpu().numpy()[0])

pd.DataFrame(preds).to_csv("./predictions.csv")
