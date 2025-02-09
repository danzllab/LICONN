import gunpowder as gp
import math
import numpy as np
import torch
from funlib.learn.torch.models import UNet, ConvPass
import logging
import tifffile

#Label-free prediction U-Net architecture
class Net(torch.nn.Module):
    def __init__(self,
                 depth=4,
                 mult_chan=32,
                 in_channels=1,
                 out_channels=1,
    ):
        super().__init__()
        self.depth = depth
        self.mult_chan = mult_chan
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.net_recurse = _Net_recurse(n_in_channels=self.in_channels, mult_chan=self.mult_chan, depth=self.depth)
        self.conv_out = torch.nn.Conv3d(self.mult_chan, self.out_channels, kernel_size=3, padding=1)

    def forward(self, input):
        x_rec = self.net_recurse(input)
        return self.conv_out(x_rec)


class _Net_recurse(torch.nn.Module):
    def __init__(self, n_in_channels, mult_chan=2, depth=0):
        """Class for recursive definition of U-network.p
        Parameters:
        in_channels - (int) number of channels for input.
        mult_chan - (int) factor to determine number of output channels
        depth - (int) if 0, this subnet will only be convolutions that double the channel count.
        """
        super().__init__()
        self.depth = depth
        n_out_channels = n_in_channels*mult_chan
        self.sub_2conv_more = SubNet2Conv(n_in_channels, n_out_channels)
        
        if depth > 0:
            self.sub_2conv_less = SubNet2Conv(2*n_out_channels, n_out_channels)
            self.conv_down = torch.nn.Conv3d(n_out_channels, n_out_channels, 2, stride=2)
            self.bn0 = torch.nn.BatchNorm3d(n_out_channels)
            self.relu0 = torch.nn.ReLU()
            
            self.convt = torch.nn.ConvTranspose3d(2*n_out_channels, n_out_channels, kernel_size=2, stride=2)
            self.bn1 = torch.nn.BatchNorm3d(n_out_channels)
            self.relu1 = torch.nn.ReLU()
            self.sub_u = _Net_recurse(n_out_channels, mult_chan=2, depth=(depth - 1))
            
    def forward(self, input):
        if self.depth == 0:
            return self.sub_2conv_more(input)
        else:  # depth > 0
            x_2conv_more = self.sub_2conv_more(input)
            x_conv_down = self.conv_down(x_2conv_more)
            x_bn0 = self.bn0(x_conv_down)
            x_relu0 = self.relu0(x_bn0)
            x_sub_u = self.sub_u(x_relu0)
            x_convt = self.convt(x_sub_u)
            x_bn1 = self.bn1(x_convt)
            x_relu1 = self.relu1(x_bn1)
            x_cat = torch.cat((x_2conv_more, x_relu1), 1)  # concatenate
            x_2conv_less = self.sub_2conv_less(x_cat)
        return x_2conv_less

class SubNet2Conv(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(n_in,  n_out, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm3d(n_out)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm3d(n_out)
        self.relu2 = torch.nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

logging.basicConfig(level=logging.INFO)

torch.backends.cudnn.benchmark = True

liconn = gp.ArrayKey("LICONN")

gt_bassoon = gp.ArrayKey("GT_BASSOON")
prediction = gp.ArrayKey('PREDICTION')
mask = gp.ArrayKey("UNLABELED_MASK")

voxel_size = gp.Coordinate([300,150,150])
input_shape = gp.Coordinate([64,128,128])
output_shape = gp.Coordinate([64,128,128]) #?
input_size = input_shape * voxel_size
output_size = output_shape * voxel_size


#list of samples to be used for training. 
#If more data is used, "probabilities" varible should be odified accosringly. E.g if 4 samples are used, probabilities = [0.25,0.25,0.25,0.25]
samples = ["LICONN_test_dataset.zarr",]
probabilities = [1]

data_sources = tuple(
        gp.ZarrSource(
            s,
            datasets = {
                liconn: "volumes/liconn_data_raw",
                gt_bassoon: "volumes/bassoon_data", #to train the shank2 prediction model, we replace bassoon_data with shank2_data
                mask: "volumes/basson_mask"
            },
            array_specs = {
                liconn: gp.ArraySpec(interpolatable=True),
                gt_bassoon: gp.ArraySpec(interpolatable=True),
                mask: gp.ArraySpec(interpolatable=False),
            }
        ) +
        gp.Normalize(liconn) +
        gp.Normalize(gt_bassoon)+
        gp.RandomLocation(min_masked=0.001, mask=mask)
        for s in samples
    )

simple_augment = gp.SimpleAugment(transpose_only=[1, 2], mirror_only=[1, 2])
intensity_augmentation = gp.IntensityAugment(liconn, 0.9, 1.1, -0.1, 0.1)

snapshot = gp.Snapshot(
    dataset_names={
        liconn: "liconn",
        gt_bassoon: "gt_bassoon",
        prediction:'prediction'
        },
    output_filename="batch_train_bassoon_{iteration}.zarr",
    every = 5000
)

model = Net()
print(model)
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)#, betas = (0.5, 0.999))

cache = gp.PreCache(
    cache_size = 20,
    num_workers= 10
)

unsqueeze = gp.Unsqueeze([liconn, gt_bassoon])

stack = gp.Stack(8)

train = gp.torch.Train(
    model,
    loss,
    optimizer,
    inputs = {
        'input': liconn
    },
    loss_inputs = {
        0: prediction,
        1: gt_bassoon,
    },
    outputs = {
        0: prediction
    },
    save_every = 5000,
    log_dir = 'log',
    checkpoint_basename="train_bassoon"
)

squeeze = gp.Squeeze([liconn,gt_bassoon,prediction])
squeeze2 = gp.Squeeze([liconn,gt_bassoon,prediction])

pipeline = (
    data_sources
    + gp.RandomProvider(probabilities=probabilities) 
    + simple_augment
    + intensity_augmentation
    + unsqueeze 
    + stack 
    + cache
    + train
    + snapshot
)

request = gp.BatchRequest()
request.add(liconn, input_size)
request.add(mask, output_size)
request.add(gt_bassoon, output_size)
request.add(prediction, output_size)

with gp.build(pipeline):
  for i in range(10000):
        batch = pipeline.request_batch(request)
print('Pipeline:')
print(pipeline)