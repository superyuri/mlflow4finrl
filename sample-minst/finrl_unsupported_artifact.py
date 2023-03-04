#
# Trains an MNIST digit recognizer using PyTorch, and uses tensorboardX to log training metrics
# and weights in TensorBoard event format to the MLflow run's artifact directory. This stores the
# TensorBoard events in MLflow for later access using the TensorBoard command line tool.
#
# NOTE: This example requires you to first install PyTorch (using the instructions at pytorch.org)
#       and tensorboardX (using pip install tensorboardX).
#
# Code based on https://github.com/lanpa/tensorboard-pytorch-examples/blob/master/mnist/main.py.
#
import argparse
import os
import mlflow
import mlflow.pytorch
import pickle
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import tensorboardX
from tensorboardX import SummaryWriter
import mlflow.pyfunc
import cloudpickle
from sys import version_info

# mlflow.set_tracking_uri("http://mlflow:5000")
# mlflow.set_experiment("finrl_u")

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                              minor=version_info.minor,
                                              micro=version_info.micro)

# Command-line arguments
parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    metavar="N",
    help="input batch size for training (default: 64)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=1000,
    metavar="N",
    help="input batch size for testing (default: 1000)",
)
parser.add_argument(
    "--epochs", type=int, default=10, metavar="N", help="number of epochs to train (default: 10)"
)
parser.add_argument(
    "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
)
parser.add_argument(
    "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
)
parser.add_argument(
    "--enable-cuda",
    type=str,
    choices=["True", "False"],
    default="True",
    help="enables or disables CUDA training",
)
parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    metavar="N",
    help="how many batches to wait before logging training status",
)
args = parser.parse_args()

enable_cuda_flag = True if args.enable_cuda == "True" else False

args.cuda = enable_cuda_flag and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs,
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=args.test_batch_size,
    shuffle=True,
    **kwargs,
)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=0)

    def log_weights(self, writer, step):
        writer.add_histogram("weights/conv1/weight", self.conv1.weight.data, step)
        writer.add_histogram("weights/conv1/bias", self.conv1.bias.data, step)
        writer.add_histogram("weights/conv2/weight", self.conv2.weight.data, step)
        writer.add_histogram("weights/conv2/bias", self.conv2.bias.data, step)
        writer.add_histogram("weights/fc1/weight", self.fc1.weight.data, step)
        writer.add_histogram("weights/fc1/bias", self.fc1.bias.data, step)
        writer.add_histogram("weights/fc2/weight", self.fc2.weight.data, step)
        writer.add_histogram("weights/fc2/bias", self.fc2.bias.data, step)


    def log_scalar(self, writer, name, value, step):
        """Log a scalar value to both MLflow and TensorBoard"""
        writer.add_scalar(name, value, step)
        mlflow.log_metric(name, value)


class MnistModel(mlflow.pyfunc.PythonModel):

    def __init__(self):
        super().__init__()
        self.model = Net()
        if args.cuda:
            self.model.cuda()

        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum)

        # Create a SummaryWriter to write TensorBoard events locally
        output_dir = dirpath = tempfile.mkdtemp()
        self.writer = SummaryWriter(output_dir)
        print("Writing TensorBoard events locally to %s\n" % output_dir)

    def train(self, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % args.log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.data.item(),
                    )
                )
                step = epoch * len(train_loader) + batch_idx
                self.model.log_scalar(self.writer, "train_loss", loss.data.item(), step)
                self.model.log_weights(self.writer, step)


    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = self.model(data)
                test_loss += F.nll_loss(
                    output, target, reduction="sum"
                ).data.item()  # sum up batch loss
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data).cpu().sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuracy = 100.0 * correct / len(test_loader.dataset)
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss, correct, len(test_loader.dataset), test_accuracy
            )
        )
        step = (epoch + 1) * len(train_loader)
        self.model.log_scalar(self.writer, "test_loss", test_loss, step)
        self.model.log_scalar(self.writer, "test_accuracy", test_accuracy, step)

    # preprocess the input with prediction from the mnist model
    def _score(self, data):
        self.model.eval()
        output = self.model(data)
        pred = output.data.max(1)[1]
        return pred

    def predict(self, context, model_input):
        # Apply the preprocess function from the vader model to score
        # model_output = model_input.apply_(lambda col: self._score(col))
        model_output = self._score(model_input)
        return model_output

model_path = "model"
# reg_model_name = "PyFuncMnist"
mnist_model = MnistModel()

# Set the tracking URI to use local SQLAlchemy db file and start the run
# Log MLflow entities and save the model
# mlflow.set_tracking_uri("sqlite:///mlruns.db")

# Log our parameters into mlflow
for key, value in vars(args).items():
    mlflow.log_param(key, value)

# Perform the training
for epoch in range(1, args.epochs + 1):
    mnist_model.train(epoch)
    mnist_model.test(epoch)

# Edited by Edward 
# SummaryWriter is not supported by mlflow so remove before log_model
mnist_model.writer = None

# Edited by Edward 
# save_model is commented out
# mlflow.pyfunc.save_model(path=model_path, python_model=mnist_model, conda_env=conda_env)

# Use the saved model path to log and register into the model registry
model_info = mlflow.pyfunc.log_model(artifact_path=model_path,
                        python_model=mnist_model,
                        # registered_model_name=reg_model_name,
                        # conda_env=conda_env
                        )

# Load the model from the model registry and score
# model_uri = f"models:/{reg_model_name}/1"
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

# Extract a few examples from the test dataset to evaluate on
eval_data, eval_labels = next(iter(test_loader))
if args.cuda:
    eval_data, eval_labels = eval_data.cuda(), eval_labels.cuda() # add this line
# Make a few predictions
predictions = loaded_model.predict(eval_data)
template = 'Sample {} : Ground truth is "{}", model prediction is "{}"'
print("\nSample predictions")
for index in range(5):
    print(template.format(index, eval_labels[index], predictions[index]))