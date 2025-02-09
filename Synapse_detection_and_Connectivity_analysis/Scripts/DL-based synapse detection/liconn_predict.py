import gunpowder as gp
import logging
import math
import numpy as np
import os
import sys
import torch
import zarr
from funlib.learn.torch.models import UNet, ConvPass

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

def predict(
        file_name,
        dataset,
        out_file,
        out_ds):

    voxel_size = gp.Coordinate([300,150,150])

    liconn = gp.ArrayKey("LICONN")
    prediction = gp.ArrayKey('PREDICTION')

    input_shape = gp.Coordinate([64,128,128])
    output_shape = gp.Coordinate([64,128,128])

    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    context = (input_size - output_size) / 2
    
    model = Net()
    model.eval()

    scan_request = gp.BatchRequest()

    scan_request.add(liconn, input_size)
    scan_request.add(prediction, output_size)

    source = gp.ZarrSource(
                file_name,
            {
                liconn: dataset
            },
            {
                liconn: gp.ArraySpec(interpolatable=True)
            })

    with gp.build(source):
        total_input_roi = source.spec[liconn].roi.grow(context, context)
        total_output_roi = source.spec[liconn].roi

    f = zarr.open(out_file, 'a')
    ds = f.create_dataset(out_ds, shape=[1]+[i/j for i, j in zip(total_output_roi.get_shape(), voxel_size)])
    ds.attrs['resolution'] = voxel_size
    ds.attrs['offset'] = total_output_roi.get_offset()

    predict = gp.torch.Predict(
        model,
        checkpoint=f'train07_20K', 
        inputs = {
            'input': liconn
        },
        outputs = {
            0: prediction,
        })

    
    scan = gp.Scan(scan_request)

    write = gp.ZarrWrite(
            dataset_names={
                prediction: out_ds
            },
            output_filename=out_file)

    pipeline = (
            source +
            gp.Pad(liconn, context) +
            gp.Normalize(liconn) +
            gp.Unsqueeze([liconn]) +
            gp.Unsqueeze([liconn]) +
            predict +
            gp.Squeeze([prediction]) +
            write+
            scan)
    
    predict_request = gp.BatchRequest()

    predict_request[liconn] = total_input_roi
    predict_request[prediction] = total_output_roi

    with gp.build(pipeline):
        pipeline.request_batch(predict_request)
    print(pipeline)


if __name__ == "__main__":

    for name in [
        'LICONN_test_dataset.zarr'
    ]:
        file_name = name 
        dataset_name = 'volumes/liconn_data_raw'
        out_file =name + '_bassoon_lfp_train_liconn.zarr'
        out_ds = 'prediction'

        predict(
                file_name,
                dataset_name,
                out_file,
                out_ds)
            