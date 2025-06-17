import torch
import torch.nn as nn
from einops import rearrange


class attention_block(nn.Module):
    
    """
    A self attention block.
    
    Args:
        config (dict): A config file in .yaml format, loaded using yaml.safe_load. \n
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_heads']
        self.head_dim = config['head_dim']
        
        ############################################################################
        # qkv block - A Linear Layer that projects input into query, value and key #
        ############################################################################
        
        self.qkv_block = nn.Linear(in_features = self.hidden_size,
                                   out_features = 3 * self.num_heads * self.head_dim)
        
        ############################################################################################
        # Hidden size projection block - A Linear Layer that projects output back into hidden size #
        ############################################################################################
    
        self.hidden_size_proj_block = nn.Linear(in_features = self.num_heads * self.head_dim,
                                               out_features = self.hidden_size)
        
        ##################################
        # Weight and Bias Initialization #
        ##################################
        nn.init.xavier_uniform_(self.qkv_block.weight)
        nn.init.constant_(self.qkv_block.bias, 0)
        nn.init.xavier_uniform_(self.hidden_size_proj_block.weight)
        nn.init.constant_(self.hidden_size_proj_block.bias, 0)
        
        
    
    def forward(self, x):
        
        """
        Forward propagation of the self-attention block.
        
        Args:
            x (float tensor): Input tensor, (B, num_patches, hidden_size)
        
        Returns:
            out (float tensor): Output tensor, (B, num_patches, hidden_size)
        """
        
        out = x
        
        ######################################################################################
        # Obtain Query, Key and Value - Result in (B, num_patches, 3 * num_heads * head_dim) #
        ######################################################################################    
        out = self.qkv_block(out)
        
        ######################################################################################
        # Split into Query, Key and Value - Result in (B, num_patches, num_heads * head_dim) #
        ######################################################################################
        query, key, value = torch.split(out,
                                        split_size_or_sections = self.num_heads * self.head_dim,
                                        dim = -1)
        
        ########################################################################################
        #               Rearrange Query, Key, Value to ease operation for each heads           #
        # From (B, num_patches, num_heads * head_dim) -> (B, num_heads, num_patches, head_dim) #
        ########################################################################################
        query = rearrange(query, 'B N (nh H) -> B nh N H', nh = self.num_heads, H = self.head_dim)
        key = rearrange(key, 'B N (nh H) -> B nh N H', nh = self.num_heads, H = self.head_dim)
        value = rearrange(value, 'B N (nh H) -> B nh N H', nh = self.num_heads, H = self.head_dim)
        
        #################################################################################
        # Compute attention weight - Result in (B, num_heads, num_patches, num_patches) #
        #################################################################################
        attn_weight = torch.matmul(query, key.transpose(-2,-1)) / (self.head_dim ** 0.5)
        attn_weight = torch.nn.functional.softmax(attn_weight, dim = -1)
        
        ############################################################
        # Compute output by multiplying attention weight and Value #
        # Result in (B, num_heads, num_patches, head_dim)          #
        ############################################################
        out = torch.matmul(attn_weight, value)
        
        ########################################################################################
        # Prepare to project to hidden size - Result in (B, num_patches, num_heads x head_dim) #
        ########################################################################################
        out = rearrange(out, 'B nh N H -> B N (nh H)')
        
        ############################################################
        # Project into hidden size - Result in (B, N, hidden_size) #
        ############################################################
        out = self.hidden_size_proj_block(out)
        
        return out
