import torch
import torch.nn as nn
from models.attention import attention_block

class transformer_block(nn.Module):
    
    """
    A transformer block. The scale and shift parameters are learnt based on time embedding instead of using built-in learnable parameter in normalisation blocks.
    
    Args:
        config (dict): A config file in .yaml format, loaded using yaml.safe_load. \n
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.hidden_size = config['hidden_size']
        self.ff_hidden_dim = 4 * self.hidden_size
        
        ###################################################
        # Normalisation block before attention block      #
        # Built-in learnable scale and shift are disabled #
        ###################################################
        self.attn_norm_block = nn.LayerNorm(normalized_shape = self.hidden_size,
                                            elementwise_affine = False,
                                            eps = 1E-6)
        
        #################################################################################################
        # Attention block - Take in (B, num_patches, hidden_size), output (B, num_patches, hidden_size) #
        #################################################################################################
        self.attn_block = attention_block(config = self.config)
        
        
        ###################################################
        # Normalisation block before final mlp block      #
        # Built-in learnable scale and shift are disabled #
        ################################################### 
        self.ff_norm_block = nn.LayerNorm(normalized_shape = self.hidden_size,
                                          elementwise_affine = False,
                                          eps = 1E-6)
        
        ###########################################################################################
        # MLP block - Take in (B, num_patches, hidden_size), output (B, num_patches, hidden_size) # 
        ###########################################################################################
        self.mlp_block = nn.Sequential(nn.Linear(in_features = self.hidden_size,
                                                 out_features = self.ff_hidden_dim),
                                       nn.GELU(approximate = 'tanh'),
                                       nn.Linear(in_features = self.ff_hidden_dim,
                                                 out_features = self.hidden_size))
        
        ######################################################################
        # Adaptive Norm Block - Result in (B, num_patches, 6 * hidden_size)  #
        # Responsible to generate the 6 different scale and shift parameters #
        ######################################################################
        self.adaptive_norm_block = nn.Sequential(nn.SiLU(),
                                                 nn.Linear(in_features = self.hidden_size,
                                                           out_features = 6 * self.hidden_size))
    
        ################################
        # Weight & Bias Initialization #
        ################################
        nn.init.xavier_uniform_(self.mlp_block[0].weight)
        nn.init.constant_(self.mlp_block[0].bias, 0)
        nn.init.xavier_uniform_(self.mlp_block[-1].weight)
        nn.init.constant_(self.mlp_block[-1].bias, 0)
        nn.init.constant_(self.adaptive_norm_block[1].weight, 0)
        nn.init.constant_(self.adaptive_norm_block[1].bias, 0)
                
    
    def forward(self, x, time_embedding):
        """
        Forward propagation of a transformer block.
        
        Args:
            x (float tensor): Input tensor, (B, num_patches, hidden_size). \n
            time_embedding (float tensor): Time embedding, (B, hidden_size). \n
        
        Returns:
            out (float tensor): Output tensor, (B, num_patches, hidden_size)
        """
        
        ###################################################
        # Get all the shift and scale parameters          #
        # Each parameter in the shape of (B, hidden_size) #
        ###################################################
        shift_scale = self.adaptive_norm_block(time_embedding)
        
        pre_attn_shift, pre_attn_scale, post_attn_scale, \
        pre_mlp_shift, pre_mlp_scale, post_mlp_scale = torch.chunk(shift_scale, chunks = 6, dim = -1)
        
        ############################################
        # Actual forward propagation of the layers #
        ############################################
        out = x
        
        ########################
        # Obtain the residuals #
        ########################
        attn_in = out
        
        ###############################################
        # Pass through normalisation before attention #
        # Result in (B, num_patches, hidden_size)     #
        ###############################################
        out = self.attn_norm_block(out)
        
        #######################################################################
        # Perform scale and shift                                             #
        # out in the shape of (B, num_patches, hidden_size)                   #
        # shift and scale parameters are unsqueezed into (B, 1, hidden_size)  #
        #######################################################################
        out = (out * (1 + pre_attn_scale.unsqueeze(1))) + (pre_attn_shift.unsqueeze(1))
        
        ###########################################
        # Pass through attention block            #
        # Result in (B, num_patches, hidden_size) #
        ###########################################
        out = self.attn_block(out)
        
        #############################################################
        # Perform scale                                             #
        # out in the shape of (B, num_patches, hidden_size)         #
        # scale parameters are unsqueezed into (B, 1, hidden_size)  #
        #############################################################
        out = out * (post_attn_scale.unsqueeze(1))
        
        ##########################
        # Sum with the residuals #
        ##########################
        out = out + attn_in
        
        
        ########################
        # Obtain the residuals #
        ########################
        mlp_in = out
        
        ###########################################
        # Pass through normalisation before mlp   #
        # Result in (B, num_patches, hidden_size) #
        ###########################################
        out = self.ff_norm_block(out)
        
        #######################################################################
        # Perform scale and shift                                             #
        # out in the shape of (B, num_patches, hidden_size)                   #
        # shift and scale parameters are unsqueezed into (B, 1, hidden_size)  #
        #######################################################################
        out = (out * (1 + pre_mlp_scale.unsqueeze(1))) + (pre_mlp_shift.unsqueeze(1))
        
        ###########################################
        # Pass through mlp block                  #
        # Result in (B, num_patches, hidden_size) #
        ###########################################
        out = self.mlp_block(out)
        
        #############################################################
        # Perform scale                                             #
        # out in the shape of (B, num_patches, hidden_size)         #
        # scale parameters are unsqueezed into (B, 1, hidden_size)  #
        #############################################################
        out = out * (post_mlp_scale.unsqueeze(1))
        
        ##########################
        # Sum with the residuals #
        ##########################
        out = out + mlp_in
        
        return out
        
        

