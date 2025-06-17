import torch
import torch.nn as nn
from models.patch_embed import patch_embed
from models.transformer_block import transformer_block
from einops import rearrange

def get_sinusoidal_embedding(timestep, timestep_embedding_dim):
    
    """
    Get sinusoidal embedding for the given timestep.
    
    Args:
        timestep (tensor): A batch of time step, expected in the shape of (B,). \n
        timestep_embedding_dim (int): Timestep embedding dimension. \n
    
    Returns:
        time_embedding (float tensor): Time embedding, in the shape of (B, timestep_embedding_dim), on the same device as timestep. \n
    """
    
    ######################################
    # Expand the shape of input timestep #
    ######################################
    timestep = timestep[:, None].repeat(1, timestep_embedding_dim // 2) # Shape (B, timestep_embedding_dim // 2)
    
    ####################################
    # Factor = 10000 ** (2i / d_model) #
    ####################################
    factor = (torch.arange(0, timestep_embedding_dim // 2) * 2) / (timestep_embedding_dim)
    factor = 10000 ** factor   # Shape (timestep_embedding_dim // 2, )
    factor = factor.to(timestep.device.type)
    
    #########################################
    # Concatentate the sine and cosine part #
    #########################################
    time_embedding = torch.cat([ torch.sin(timestep / factor), torch.cos(timestep / factor)], dim =-1) # Shape (B, timestep_embedding_dim)
    
    return time_embedding


class DiT(nn.Module):
    
    """
    A Diffusion Transformer, used to predict noise added to the input images.
    
    Args:
        config (dict): A config file in .yaml format, loaded using yaml.safe_load. \n
        img_channels (int): The number of channels in input image / latent data. \n
        img_size (int): The spatial size of the input image / latent data. \n
    """
    
    def __init__(self, config, img_channels, img_size):
        super().__init__()
        self.config = config
        self.timestep_embedding_dim = config['timestep_embedding_dim']
        self.hidden_size = config['hidden_size']
        
        self.patch_size = config['patch_size']
        self.num_layers = config['num_layers']
        
        self.img_channels = img_channels
        self.img_size = img_size
        
        #########################
        # Patch Embedding Block #
        #########################
        self.patch_embed_block = patch_embed(config = self.config,
                                             img_channels = self.img_channels,
                                             img_size = self.img_size)
        
        ##################################################################################
        # Time Projection Blocks - To convert from timestep_embedding_dim to hidden_size #
        ##################################################################################
        self.t_proj_block = nn.Sequential(nn.Linear(in_features = self.timestep_embedding_dim,
                                                    out_features = self.hidden_size),
                                          nn.SiLU(),
                                          nn.Linear(in_features =  self.hidden_size,
                                                    out_features = self.hidden_size))
        
        ##############################
        # List of Transformer Blocks #
        ##############################
        self.transformer_blocks = nn.ModuleList([])
        for _ in range(self.num_layers):
            self.transformer_blocks.append(transformer_block(config = self.config))
        
       
        ##############################
        # Output Normalization Block #
        ##############################
        self.out_norm_block = nn.LayerNorm(normalized_shape = self.hidden_size,
                                           elementwise_affine = False,
                                           eps = 1E-6)
        
        ######################################################
        # Block to learn scale and shift from time_embedding #
        ######################################################
        self.adaptive_norm_block = nn.Sequential(nn.SiLU(),
                                                 nn.Linear(in_features = self.hidden_size,
                                                           out_features = 2 * self.hidden_size ))
        
        #####################################################################################################
        # Output Projection Block - To convert from hidden_size to (patch_size x patch_size x img_channels) #
        #####################################################################################################
        self.out_proj_block = nn.Linear(in_features = self.hidden_size,
                                        out_features = self.patch_size * self.patch_size * self.img_channels)
        
        ################################
        # Weight & Bias Initialization #
        ################################
        nn.init.normal_(self.t_proj_block[0].weight, std=0.02)
        nn.init.normal_(self.t_proj_block[-1].weight, std=0.02)
        nn.init.constant_(self.adaptive_norm_block[-1].weight, 0)
        nn.init.constant_(self.adaptive_norm_block[-1].bias, 0)
        nn.init.constant_(self.out_proj_block.weight, 0)
        nn.init.constant_(self.out_proj_block.bias, 0)
        
    def forward(self, x, timestep = None):
        
        """
        Forward propagation of Diffusion Transformer (DiT).
        
        Args:
            x (float tensor): A batch of input images modified with noise, (B, C, H, W). \n
            timestep (int tensor): A batch of timesteps input, (B,). \n
            
        Returns:
            out (float tensor): A batch of predicted added noise to the input (B, C, H , W). \n
        """
        
        if timestep == None:
            print("Enterring testing mode... ")
            timestep = torch.tensor([0])
        
        ################################################################
        # Sinusoidal Embedding - Result in (B, timestep_embedding_dim) #
        ################################################################
        timestep_embedding = get_sinusoidal_embedding(timestep = timestep,
                                                      timestep_embedding_dim = self.timestep_embedding_dim)
        
        ###############################################################
        # Send the timestep_embedding to the same device as the model #
        ###############################################################
        device = self.t_proj_block[0].weight.device.type 
        timestep_embedding = timestep_embedding.to(device)
        
        ##############################################################################
        # Project the timestep_embedding to hidden_size - Result in (B, hidden_size) #
        ##############################################################################
        timestep_embedding = self.t_proj_block(timestep_embedding)
        
        ###################################################################
        # Learn the scale and shift parameter from the timestep_embedding #
        ###################################################################
        scale_shift = self.adaptive_norm_block(timestep_embedding)
        pre_mlp_shift, pre_mlp_scale = torch.chunk(scale_shift, chunks = 2, dim = -1)
        
        ############################
        # Main forward propagation #
        ############################
        out = x
        
        #############################################################
        # Patch Embedding - Result in (B, num_patches, hidden_size) #
        #############################################################
        out = self.patch_embed_block(out)
        
        ##############################################################
        # Transformer Block - Result in (B, num_patches, hidden_size #
        ##############################################################
        for block in self.transformer_blocks:
            out = block(out, timestep_embedding)
        
        ###################################
        # Normalisation and scale & shift #
        ###################################
        out = self.out_norm_block(out)
        out = (out * (1 + pre_mlp_scale.unsqueeze(1))) + (pre_mlp_shift.unsqueeze(1))
        
        ########################################################################################################
        # Project to original patchified image shape - Result in (B, num_patches, patch_size x patch_size x C) #
        ########################################################################################################
        out = self.out_proj_block(out)
        
        ##########################################################
        # Rearrange into original shape - Result in (B, C, H, W) #
        ##########################################################
        out = rearrange(out, 'B (nh nw) (ph pw C) -> B C (nh ph) (nw pw)',
                        nh = int(self.img_size / self.patch_size),
                        nw = int(self.img_size / self.patch_size),
                        ph = self.patch_size,
                        pw = self.patch_size,
                        C = self.img_channels)
        
        return out
        
        
        
        
    

