import torch
import torch.nn as nn
from einops import rearrange


def get_2D_pos_embedding(img_size, patch_size, pos_embedding_dim, device):
    
    """
    Get positional embedding in 2-dimensions
    
    Args:
        img_size (int): Spatial size of the image. \n
        patch_size (int): Spatial size of a patch within the image. \n
        pos_embedding_dim (int): Positional embedding dimension. \n
        device (str): 'cpu' or 'cuda'
    
    Returns:
        out_2D_embedding (float tensor): A 2D position embedding, (img_size/patch_size x img_size/patch_size, pos_embedding_dim), on device
    """
        
    #####################################################
    # Calculate number of grid in the h and w direction #
    #####################################################
    num_grid_h = img_size / patch_size
    num_grid_w = img_size / patch_size
    
    ############################################################
    # Create the index for the grid in both h and w directions #
    ############################################################
    grid_h_idx = torch.arange(0, num_grid_h).to(device)
    grid_w_idx = torch.arange(0, num_grid_w).to(device)
    
    ############################################################
    # Form a grid and extract the 2D coordinates for each grid #
    ############################################################
    coordinate_h, coordinate_w = torch.meshgrid(grid_h_idx, grid_w_idx,
                                                indexing='ij')
    #############################
    # Flatten their coordinates #
    #############################
    coordinate_h = coordinate_h.reshape(-1)
    coordinate_w = coordinate_w.reshape(-1)
    
    ################################################################################
    # Expand them into shape of (num_grid_h x num_grid_w , pos_embedding_dim // 4) #
    ################################################################################
    coordinate_h = coordinate_h[:, None].repeat(1, pos_embedding_dim // 4) # (num_grid_h x num_grid_w , pos_embedding_dim // 4)
    coordinate_w = coordinate_w[:, None].repeat(1, pos_embedding_dim // 4) # (num_grid_h x num_grid_w , pos_embedding_dim // 4)
    
    
    
                                                ########################
                                                # Sinusoidal Embedding #
                                                ########################
                                                
    ####################################
    # Factor = 10000 ** (2i / d_model) #
    ####################################
    factor = (torch.arange(0, pos_embedding_dim//4) * 2) / (pos_embedding_dim/2)
    factor = 10000 ** factor # (pos_embedding_dim // 4, )
    factor = factor.to(device)
    
    h_embedding = torch.cat([ torch.sin(coordinate_h / factor), torch.cos(coordinate_h / factor) ], dim = 1) # (num_grid_h x num_grid_w , pos_embedding_dim // 2)
    w_embedding = torch.cat([ torch.sin(coordinate_w / factor), torch.cos(coordinate_w / factor) ], dim = 1) # (num_grid_h x num_grid_w , pos_embedding_dim // 2)
    
    out_2D_embedding = torch.cat([h_embedding, w_embedding], dim = 1) # (num_grid_h x num_grid_w , pos_embedding_dim)
    
    return out_2D_embedding




class patch_embed(nn.Module):
    
    """
    Convert a batch of images into a patch embedding in the shape of (B, num_patches, hidden_size)
    
    Args:
        config (dict): A config file in .yaml format, loaded using yaml.safe_load. \n
        img_channels (int): The number of channels in input image / latent data. \n
        img_size (int): The spatial size of the input image / latent data. \n
    """
    
    def __init__(self, config, img_channels, img_size):
        super().__init__()
        self.config = config
        self.patch_size = config['patch_size']
        self.hidden_size = config['hidden_size']
        
        self.img_channels = img_channels
        self.img_size = img_size
        
        #################################################################################################
        # Attention dimension projection block - Convert dimensions from (ph x ph x C) to (hidden_size) #
        #################################################################################################
        self.attn_dim_proj_block = nn.Linear(in_features = self.patch_size * self.patch_size * self.img_channels,
                                             out_features = self.hidden_size)
        
        nn.init.xavier_uniform_(self.attn_dim_proj_block.weight)
        nn.init.constant_(self.attn_dim_proj_block.bias, 0)
    
    def forward(self, x):
        
        """
        Forward propagation for the patch embed block.
        
        Args:
            x (float tensor): A batch of input images, (B, C, H, W)
        
        Returns:
            out (float tensor): A batch of patch embedding, (B, num_patches, hidden_size)
        """
        
        out = x
        
        ########################################################
        # Patchify - Result in shape (B, nh x nw, ph x pw x C) #
        ########################################################
        out = rearrange(out, 'B C (nh ph) (nw pw) -> B (nh nw) (ph pw C)',
                        ph = self.patch_size,
                        pw = self.patch_size)
        
        ##############################################################################################
        # Project the last dimensions to match the hidden size - Result in (B, nh x nw, hidden_size) #
        ##############################################################################################
        out = self.attn_dim_proj_block(out)
        
        #################################################################################
        # Add 2D positional embedding information - Result in (B, nh x nw, hidden_size) #
        #################################################################################
        out_2D_embedding = get_2D_pos_embedding(img_size = self.img_size,
                                                patch_size = self.patch_size,
                                                pos_embedding_dim = self.hidden_size,
                                                device = out.device.type)
        
        out = out + out_2D_embedding
        
        return out

