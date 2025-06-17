import torch
import torch.nn as nn
from models.blocks import down_block, mid_block, up_block

class VAE(nn.Module):
    
    """
    A Variational Auto Encoder (VAE)
    
    Args:
        config (dict): A config file in .yaml format, loaded using yaml.safe_load. \n
    """
    
    def __init__(self, config):
        super().__init__()
        
        ##############
        # Attributes #
        ##############
        
        # General Attributes 
        img_channels = config['img_channels']
        norm_channels = config['norm_channels']
        z_channels = config['z_channels']
        codebook_size = config['codebook_size']
        
        # Attributes for down blocks
        down_channels = config['down_channels']
        down_sample_flag = config['down_sample']
        down_attn_flag = config['down_attn']
        
        # Attributes for encoder mid blocks
        enc_mid_channels = config['mid_channels']
        enc_mid_attn_flag = config['mid_attn']
        enc_num_heads = config['num_heads']
        
        # Attributes for decoder mid blocks
        dec_mid_channels = list(reversed(enc_mid_channels))
        dec_mid_attn_flag = list(reversed(enc_mid_attn_flag))
        dec_num_heads = config['num_heads']
        
        # Attributes for up blocks
        up_channels = list(reversed(down_channels))
        up_sample_flag = list(reversed(down_sample_flag))
        up_attn_flag = list(reversed(down_attn_flag))
        
        # Test
        assert down_channels[-1] == enc_mid_channels[0], "[WARNING] Last down channels do not match the first encoder mid channels."
        assert up_channels[0] == dec_mid_channels[-1], "[WARNING] Last decoder mid channels do not match the first up channels."
        
        
        
        #####################################################################################################
        # Input Conv - To change channels of input images to be same as the first channels in down channels #
        #####################################################################################################
        
        self.input_conv_block = nn.Conv2d(in_channels = img_channels,
                                          out_channels = down_channels[0],
                                          kernel_size = 3,
                                          stride = 1, 
                                          padding = 1)
        
        #########################
        # Encoder - Down Blocks #
        #########################
        
        self.down_blocks = nn.ModuleList([])
        for i in range(len(down_channels) - 1):
            self.down_blocks.append(down_block(config = config,
                                               in_channels = down_channels[i],
                                               out_channels = down_channels[i+1],
                                               time_embedding_dim = None,
                                               down_attn_flag = down_attn_flag[i],
                                               num_heads = None,
                                               context_flag = False,
                                               context_dim = None,
                                               down_sample_flag = down_sample_flag[i]))
        
        ########################
        # Encoder - Mid Blocks # 
        ########################
        
        self.enc_mid_blocks = nn.ModuleList([])
        for i in range(len(enc_mid_channels) - 1):
            self.enc_mid_blocks.append(mid_block(config = config,
                                                 in_channels = enc_mid_channels[i],
                                                 out_channels = enc_mid_channels[i+1],
                                                 time_embedding_dim = None,
                                                 mid_attn_flag = enc_mid_attn_flag[i],
                                                 num_heads = enc_num_heads,
                                                 context_flag = False,
                                                 context_dim = None))
        
        ########################################################################################
        # Encoder out block - To convert channels into 2 x z_channels (1 for mean, 1 for logvar) #
        ########################################################################################
        
        self.enc_out_block = nn.Sequential(nn.GroupNorm(num_groups = norm_channels,
                                                        num_channels = enc_mid_channels[-1]),
                                           nn.SiLU(),
                                           nn.Conv2d(in_channels = enc_mid_channels[-1],
                                                     out_channels = 2 * z_channels,
                                                     kernel_size = 3,
                                                     stride = 1, 
                                                     padding = 1))
        
        ###################
        # Pre quant block #
        ###################
        
        self.pre_quant_block = nn.Conv2d(in_channels = 2 * z_channels,
                                         out_channels = 2 * z_channels,
                                         kernel_size = 1,
                                         stride = 1, 
                                         padding = 0)
        
        ####################
        # Post quant block #
        ####################
        
        self.post_quant_block = nn.Conv2d(in_channels = z_channels,
                                          out_channels = z_channels,
                                          kernel_size = 1,
                                          stride = 1,
                                          padding = 0)
        
        
        #####################################################################################
        # Decoder in block - To convert channels into first channel of decoder mid channels #
        #####################################################################################
        
        self.dec_in_block = nn.Conv2d(in_channels = z_channels,
                                      out_channels = dec_mid_channels[0],
                                      kernel_size = 3,
                                      stride = 1,
                                      padding = 1)
        
         
        ########################
        # Decoder - Mid Blocks #
        ########################

        self.dec_mid_blocks = nn.ModuleList([])
        for i in range(len(dec_mid_channels) - 1):
            self.dec_mid_blocks.append(mid_block(config = config,
                                                 in_channels = dec_mid_channels[i],
                                                 out_channels = dec_mid_channels[i+1],
                                                 time_embedding_dim = None,
                                                 mid_attn_flag = dec_mid_attn_flag[i],
                                                 num_heads = dec_num_heads,
                                                 context_flag = False,
                                                 context_dim = None))
        
        #######################
        # Decoder - Up Blocks #
        #######################

        self.up_blocks = nn.ModuleList([])
        for i in range(len(up_channels) - 1):
            self.up_blocks.append(up_block(config = config,
                                           in_channels = up_channels[i],
                                           out_channels = up_channels[i+1],
                                           time_embedding_dim = None,
                                           up_attn_flag = up_attn_flag[i],
                                           num_heads = None,
                                           context_flag = False,
                                           context_dim = None,
                                           up_sample_flag = up_sample_flag[i]))
            
        ###############################################################
        # Decoder out block - To convert channels into image channels #
        ###############################################################
        
        self.dec_out_block = nn.Sequential(nn.GroupNorm(num_groups = norm_channels,
                                                        num_channels = up_channels[-1]),
                                           nn.SiLU(),
                                           nn.Conv2d(in_channels = up_channels[-1],
                                                     out_channels = img_channels,
                                                     kernel_size = 3,
                                                     stride = 1,
                                                     padding = 1))
        
    
    
    def encode(self, x):
        
        """
        Encoding of a VAE
        
        Args:
            x (float tensor): A batch of input images, (B, img_channels, img_size, img_size), on cuda device, in range of -1 to 1. \n
        
        Returns:
            sample (float tensor): A batch of sampled latent data, (B, z_channels, latent_size, latent_size), on cuda device. \n
            mean_log_var (float tensor): Mean and log var to sample the latent data, (B, 2 x z_channels, latent_size, latent_size), on cuda device. \n
        """
        
        out = x
        
        # Input conv - To convert channels to first down channel
        out = self.input_conv_block(out)
        
        # Go through down blocks
        for block in self.down_blocks:
            out = block(out)
        
        # Go through encoder mid blocks
        for block in self.enc_mid_blocks:
            out = block(out)
        
        # Encoder out blocks - To convert channels to 2 x z_channels, for mean and log_var respectively
        out = self.enc_out_block(out)
        
        # Go through pre quantization blocks
        mean_log_var = self.pre_quant_block(out)
        
        # Go through quantization process
        mean, log_var = torch.chunk(mean_log_var, chunks = 2, dim = 1)
        std = torch.exp(log_var) ** 0.5
        z = torch.randn_like(mean)
        sample = mean + std*z
        
        return sample, mean_log_var
    
    def decode(self, x):
        
        """
        Decoding of a VAE
        
        Args:
            x (float tensor): A batch of sampled latent data, (B, z_channels, latent_size, latent_size), on cuda device. \n
        
        Returns:
            out (float tensor): A batch of reconstructed images, (B, img_channels, img_size, img_size), on cuda device. \n
        """
        
        out = x
        
        # Go through post quantization blocks
        out = self.post_quant_block(out)
        
        # Decode in blocks - To convert channels into first decoder mid channels
        out = self.dec_in_block(out)
        
        # Go through decoder mid blocks
        for block in self.dec_mid_blocks:
            out = block(out)
        
        # Go through up blocks
        for block in self.up_blocks:
            out = block(out)
        
        # Decoder out blocks - To convert channels into img channels
        out = self.dec_out_block(out)
        
        return out
    
    def forward(self, x):
        
        """
        Forward propagation of VAE - Encoding and Decoding
        
        Args:
            x (float tensor): A batch of input images, (B, img_channels, img_size, img_size), on cuda device, in range of -1 to 1. \n
        
        Returns:
            recon (float tensor): A batch of reconstructed images, (B, img_channels, img_size, img_size), on cuda device. \n
            mean_log_var (float tensor): Mean and log var to sample the latent data, (B, 2 x z_channels, latent_size, latent_size), on cuda device. \n
        """
        
        sample, mean_log_var = self.encode(x)
        recon = self.decode(sample)
        return recon, mean_log_var
