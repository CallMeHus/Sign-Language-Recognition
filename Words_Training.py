import sys
import os
import numpy as np
import numpy as np
# from sklearn.preprocessing import MultiLabelBinarizer
import glob
import os
import torch.nn as nn
from torch.nn.modules.utils import _triple
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time
import math
import json

# Defining the convolutional layers for Spatio Temporal Neural Network
class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input
    planes with distinct spatial and time axes, by performing a 2D convolution over the
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]
        spatial_stride = [1, stride[1], stride[2]]
        spatial_padding = [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride = [stride[0], 1, 1]
        temporal_padding = [padding[0], 0, 0]

        # compute the number of intermediary channels (M) using formula
        # from the paper section 3.5
        intermed_channels = int(
            math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels) / \
                       (kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        # the spatial conv is effectively a 2D conv due to the
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                      stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()

        # the temporal conv is effectively a 1D conv, but has batch norm
        # and ReLU added inside the model constructor, not here. This is an
        # intentional design choice, to allow this module to externally act
        # identical to a standard Conv3D, so it can be reused easily in any
        # other codebase
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size,
                                       stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x

# Defining the Residual Block for Spatio Temporal Neural Network
class SpatioTemporalResBlock(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv in
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the block.
            kernel_size (int or tuple): Size of the convolving kernels.
            downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()

        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.downsample = downsample

        # to allow for SAME padding
        padding = kernel_size // 2

        if self.downsample:
            # downsample with stride =2 the input x
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)

            # downsample with stride = 2when producing the residual
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()

        # standard conv->batchnorm->ReLU
        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.outrelu(x + res)

# Defining the Residual layers for Spatio Temporal Neural Network
class SpatioTemporalResLayer(nn.Module):
    r"""Forms a single layer of the ResNet network, with a number of repeating
    blocks of same output size stacked on top of each other

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the layer.
            kernel_size (int or tuple): Size of the convolving kernels.
            layer_size (int): Number of blocks to be stacked to form the layer
            block_type (Module, optional): Type of block that is to be used to form the layer. Default: SpatioTemporalResBlock.
            downsample (bool, optional): If ``True``, the first block in layer will implement downsampling. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=SpatioTemporalResBlock,
                 downsample=False):

        super(SpatioTemporalResLayer, self).__init__()

        # implement the first block
        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x

# Defining the full network architecture for the model
class R2Plus1DNet(nn.Module):
    r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in
    each layer set by layer_sizes, and by performing a global average pool at the end producing a
    512-dimensional vector for each element in the batch.

        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
        """

    def __init__(self, layer_sizes, block_type=SpatioTemporalResBlock):
        super(R2Plus1DNet, self).__init__()

        # first conv, with stride 1x2x2 and kernel size 3x7x7
        self.conv1 = SpatioTemporalConv(3, 64, [3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3])
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[0], block_type=block_type)
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True)
        self.conv4 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True)
        self.conv5 = SpatioTemporalResLayer(256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True)

        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool(x)

        return x.view(-1, 512)

# Defining the output layer
class R2Plus1DClassifier(nn.Module):
    r"""Forms a complete ResNet classifier producing vectors of size num_classes, by initializng 5 layers,
    with the number of blocks in each layer set by layer_sizes, and by performing a global average pool
    at the end producing a 512-dimensional vector for each element in the batch,
    and passing them through a Linear layer.

        Args:
            num_classes(int): Number of classes in the data
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
        """

    def __init__(self, num_classes, layer_sizes, block_type=SpatioTemporalResBlock):
        super(R2Plus1DClassifier, self).__init__()

        self.res2plus1d = R2Plus1DNet(layer_sizes, block_type)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.res2plus1d(x)
        x = self.linear(x)

        return x

# Defining the classes and label map to int - line 238
label_map = dict()
all_argv = sys.argv

actions = os.listdir("Training")
for i, j in enumerate(actions):
    label_map[j] = i
print(label_map)
with open("label_to_int.json", "w") as f:
    json.dump(label_map, f)
sequence_length = int(all_argv[-1])
batch = int(all_argv[1])

# Defining a function for preparing the dataset for training and validation
def prepare_data_sample(np_array, action, sequence_length, DATA_PATH, sequences, labels):
    for sequence in np_array:
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# Preparing the dataset for training and validation - line 276
sequences, labels = [], []
X_train = []
Y_train = []
X_val = []
Y_val = []
for action in actions:
    all_training_folder = os.listdir(os.path.join("Training", action))
    all_validation_folder = os.listdir(os.path.join("Validation", action))
    prepare_data_sample(all_training_folder, action, sequence_length, "Training", X_train, Y_train)
    prepare_data_sample(all_validation_folder, action, sequence_length, "Validation", X_val, Y_val)

X_train = np.array(X_train).transpose((0, 4, 1, 2, 3))
Y_train = np.array(Y_train)
X_val = np.array(X_val).transpose((0, 4, 1, 2, 3))
Y_val = np.array(Y_val)

print(X_train.shape)
print(X_val.shape)
print(Y_train.shape)
print(Y_val.shape)
np.save("X_train.npy", X_train)
np.save("Y_train.npy", Y_train)
np.save("X_val.npy", X_val)
np.save("Y_val.npy", Y_val)

# Creating the dataloader
train_dataloader = DataLoader(TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train)), batch_size=batch,
                              shuffle=True)
val_dataloader = DataLoader(TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val)), batch_size=batch)

dataloaders = {'train': train_dataloader, 'val': val_dataloader}
#
dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
print(dataset_sizes)

# Defining the model saving path, number classes and device
save = True
model_path = f'class10_{batch}_{sequence_length}_model_data.pt'

# saves the time the process was started, to compute total time at the end
num_classes = 10
start = time.time()

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

# initalize the ResNet 18 version of this model
# model = R2Plus1DClassifier(num_classes=num_classes, layer_sizes=[2, 2, 2, 2]).to(device)
model = R2Plus1DClassifier(num_classes=num_classes, layer_sizes=[2, 2, 2, 2]).to(device) # Creating the model

criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # hyperparameters as given in paper sec 4.1
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5,
                                      gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

# Below code is for training and validation
epoch_resume = 0
num_epochs = 100
max_val_score = 0
training_record = dict()
for epoch in tqdm(range(epoch_resume, num_epochs), unit="epochs", initial=epoch_resume, total=num_epochs):
    training_record[epoch] = dict()
    # each epoch has a training and validation step, in that order
    for phase in ['train', 'val']:
        # reset the running loss and corrects
        running_loss = 0.0
        running_corrects = 0

        # set model to train() or eval() mode depending on whether it is trained
        # or being validated. Primarily affects layers such as BatchNorm or Dropout.
        if phase == 'train':
            # scheduler.step() is to be called once every epoch during training
            scheduler.step()
            model.train()
        else:
            model.eval()

        for inputs, labels in dataloaders[phase]:
            # move inputs and labels to the device the training is taking place on
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor).to(device)
            optimizer.zero_grad()

            # keep intermediate states iff backpropagation will be performed. If false,
            # then all intermediate states will be thrown away during evaluation, to use
            # the least amount of memory possible.
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                # we're interested in the indices on the max values, not the values themselves
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Backpropagate and optimize iff in training mode, else there's no intermediate
                # values to backpropagate with and will throw an error.
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).cpu().numpy().tolist()

        epoch_loss = running_loss / dataset_sizes[phase]
        print(running_corrects)
        print(dataset_sizes[phase])
        #        print("****************************")
        epoch_acc = running_corrects / dataset_sizes[phase]

        print(f"{phase} Loss: {epoch_loss} Acc: {epoch_acc}")

        local_record = {"Loss:": epoch_loss, "Acc": epoch_acc}
        #       print(local_record)
        training_record[epoch][phase] = local_record
        if phase == "val":
            if epoch_acc > max_val_score:
                print("saving new model")
                max_val_score = epoch_acc
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': epoch_acc,
                    'opt_dict': optimizer.state_dict(),
                }, model_path)
# print the total time needed, HH:MM:SS format
time_elapsed = time.time() - start
json_path = f'class10_{batch}_{sequence_length}_model_data.json'
with open(json_path, "w") as outfile:
    json.dump(training_record, outfile)
print(f"Training complete in {time_elapsed // 3600}h {(time_elapsed % 3600) // 60}m {time_elapsed % 60 :.4}s")
