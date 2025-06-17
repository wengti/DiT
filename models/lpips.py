from __future__ import absolute_import
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import torch.nn
import torchvision

# Taken from https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def spatial_average(in_tens, keepdim=True):
    """
    To reduce an input of B x C x H x W into B x C x 1 x 1 using mean
    
    Args:
        in_tens (float tensor): Expected shape of B x C x H x W \n
        keepdim (boolean): Keep the original dimensionality of the input or not \n
    
    Returns:
        (float tensor): B x C x 1 x 1 if keepdim else B x C
    """
    
    
    return in_tens.mean([2, 3], keepdim=keepdim)


class vgg16(torch.nn.Module):
    
    """
    A pretrained vgg16 model
    
    Args:
        requires_grad (boolean): Decide whether to freeze the model \n
        pretrained (boolean): Decide whether to load the pretrained weight \n
    """
    
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__() # This is in fact same as super().__init__() in python 3 and above
        # Load pretrained vgg model from torchvision
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        
        #4: (Conv + Actiavtion) x 2
        for x in range(4): 
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
            
        #5: Downsample + (Conv + Actiavtion) x 2
        for x in range(4, 9): 
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
            
        #7: Downsample + (Conv + Actiavtion) x 3
        for x in range(9, 16): 
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        
        #7: Downsample + (Conv + Actiavtion) x 3
        for x in range(16, 23): 
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        
        #7: Downsample + (Conv + Actiavtion) x 3
        for x in range(23, 30): 
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        
        # Freeze vgg model
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, X):
        
        """
        Forward propagation of the vgg16 model
        
        Args:
            X (float tensor): Expected in the shape of B x C x H x W, in the range of -1 to 1, on cuda device
            
        Returns:
            out (namedtuple): A tuple consists of output feature maps at 5 different level
        """
        
        
        # Return output of vgg features
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        
        # Refer to this link to learn how namedTuple works: https://www.geeksforgeeks.org/namedtuple-in-python/
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


# Learned perceptual metric
class LPIPS(nn.Module):
    
    """
    The Learned Perceptual Image Patch Similarity model
    
    Args:
        net (str): Name of the pretrained model to be used, will be used to find the directories to the pretrained weight for LPIPS \n
        version (str): Version of the pretrained model to be used, will be used to find the directories to the pretrained weight for LPIPS \n
        use_dropout (boolean): Decide whether to apply dropout before the reducing the output from B x C x H x W into B x 1 x H x W \n
    """
    def __init__(self, net='vgg', version='0.1', use_dropout=True):
        super(LPIPS, self).__init__()
        self.version = version
        # Imagenet normalization
        self.scaling_layer = ScalingLayer()
        ########################
        
        # Instantiate vgg model
        self.chns = [64, 128, 256, 512, 512]
        self.L = len(self.chns)
        self.net = vgg16(pretrained=True, requires_grad=False)
        
        # Add 1x1 convolutional Layers
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        self.lins = nn.ModuleList(self.lins)
        ########################
        
        # Load the weights of trained LPIPS model
        import inspect
        import os
        
        # The weight can be found in here: https://github.com/richzhang/PerceptualSimilarity/tree/master
        # os.path.abspath -> Make it into an absolute path, so no matter where is this file, the path can be located
        # inspect.getfile is to get the current directory of self.__init__, .. is to get the get cwd
        model_path = os.path.abspath( 
            os.path.join(inspect.getfile(self.__init__), '..', 'weights/v%s/%s.pth' % (version, net)))
        print('Loading model from: %s' % model_path)
        self.load_state_dict(torch.load(model_path, map_location=device), strict=False) #strict = False is to ignore mismatching keys
        ########################
        
        # Freeze all parameters
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        ########################
    
    def forward(self, in0, in1, normalize=False):
        
        """
        Forward propagation of LPIPS
        
        Args:
            in0 (float tensor): Input Image 1, expected in the shape of B x C x H x W, in the range of -1 to 1 / 0 to 1, on cuda device \n
            in1 (float tensor): Input Image 2, expected in the shape of B x C x H x W, in the range of -1 to 1 / 0 to 1, on cuda device \n
            normalize (boolean): Decide whether to normalize images from 0 to 1 into -1 to 1 \n
        
        Returns:
            val (float tensor): The final output of the metrics of LPIPS, expected in the shape of B x 1 x 1 x 1
        """
    
        # Scale the inputs to -1 to +1 range if needed
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1
        ########################
        
        # Normalize the inputs according to imagenet normalization
        in0_input, in1_input = self.scaling_layer(in0), self.scaling_layer(in1)
        ########################
        
        # Get VGG outputs for image0 and image1
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        ########################
        
        # Compute Square of Difference for each layer output
        for kk in range(self.L):
            feats0[kk], feats1[kk] = torch.nn.functional.normalize(outs0[kk], dim=1), torch.nn.functional.normalize(
                outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
        ########################
        
        # 1x1 convolution followed by spatial average on the square differences
        res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        val = 0
        
        # Aggregate the results of each layer
        for l in range(self.L):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    
    """
    A block that is responsible to normalize an input into the expected range for a VGG16.
    """
    
    def __init__(self):
        super(ScalingLayer, self).__init__()
        
        # Refer to this link for what is self.register_buffer
        # https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723
        # In short: register_buffer creates a parameter that will not be optimized / back_prop
        # register_parameter creates a paramter that will be optimized / back_prop
        
        # Imagnet normalization for (0-1)
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # The values below are basically x*2 - 1 to suit images in the range of -1 to 1
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])
        
    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """
    A single linear layer which does a 1x1 conv 
    
    Args:
        chn_in (int): Input channels to the convolutional layer \n
        chn_out (int): Output channels to the convolutional layer, defaulted to 1 \n
        use_dropout(boolean): Decide whether to apply dropout before the reducing the output from B x C x H x W into B x 1 x H x W \n
    """
    
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ] # Return same shapes, but only 1 channels
        self.model = nn.Sequential(*layers) #* is to unpackage
    
    def forward(self, x):
        out = self.model(x)
        return out
