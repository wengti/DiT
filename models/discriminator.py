import torch
import torch.nn as nn
import yaml

class discriminator(nn.Module):
    
    """
    A patch GAN discriminator
    
    Args:
        config (dict): A config with .yaml extension loaded by yaml.safe_load \n
    """
    
    def __init__(self, config):
        super().__init__()
        
        channels = [config['img_channels']] + [64, 128, 256, 1]
        kernel_size = [4, 4, 4, 4]
        stride = [2, 2, 2, 1]
        padding = [1, 1, 1, 1]
        bias = [True, False, False, False]
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels = channels[i],
                          out_channels = channels[i+1],
                          kernel_size = kernel_size[i],
                          stride = stride[i],
                          padding = padding[i],
                          bias = bias[i]),
                nn.BatchNorm2d(num_features = channels[i+1]) if i!=0 and i!=len(channels)-2 else nn.Identity(),
                nn.LeakyReLU(0.2) if i!=len(channels) - 2 else nn.Identity()
                ) for i in range(len(kernel_size))])
        
    def forward(self, x):
        """
        Forward Propagation of the Patch GAN discriminator
        
        Args:
            x (float tensor): A batch of input images, expected on cuda device, in the shape of (B, img_channels, img_size, img_size)
            
        Returns:
            out (float tensor): The predicted output, in the logits form (havent gone through sigmoid), in the shape of (B, 1, h, w)
        """

        out = x
        for block in self.blocks:
            out = block(out)
        return out


