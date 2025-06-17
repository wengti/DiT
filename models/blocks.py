import torch
import torch.nn as nn


#######################
# D O W N - B L O C K #
#######################

class down_block(nn.Module):
    
    """
    A down block, can be used to create an encoder-decoder or U-Net.
    
    Args: 
        config (dict): A config file in .yaml format, loaded using yaml.safe_load. \n
        in_channels (int): Input channels. \n
        out_channels (int): Output channels. \n
        time_embedding_dim (int): Time embedding dimension. Also decides if time embedding block is included in the architecture. (Default: None) \n
        down_attn_flag (boolean): Down attention flag. Decides if self-attention block is included in the architecture. (Default: False) \n
        num_heads (int): Number of heads in self-attention and cross-attention blocks. (Default: None) \n
        context_flag (boolean): Context flag. Decides if cross-attention blocks are included in the architecture. (Default: False) \n
        context_dim (int): Context dimension. (Default: None) \n
        down_sample_flag (boolean): Down sample flag. Decides if down sample blocks are included in the architecture. (Default: True) \n
    """
    
    def __init__(self, config, in_channels, out_channels, time_embedding_dim = None, down_attn_flag = False, num_heads = None,
                 context_flag = False, context_dim = None, down_sample_flag = True):
        super().__init__()
        
        self.norm_channels = config['norm_channels']
        self.num_down_layers = config['num_down_layers']
        self.down_attn_flag = down_attn_flag
        self.context_flag = context_flag
        
        #################
        # ResNet Blocks # 
        #################
        
        self.first_resnet_block = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups = self.norm_channels,
                             num_channels = in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels = in_channels if i == 0 else out_channels,
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1)
                ) for i in range(self.num_down_layers)])
        
        self.second_resnet_block = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups = self.norm_channels,
                             num_channels = out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels = out_channels,
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1)
                ) for _ in range(self.num_down_layers)])
        
        self.residual_conv_block = nn.ModuleList([
            nn.Conv2d(in_channels = in_channels if i == 0 else out_channels,
                      out_channels = out_channels,
                      kernel_size = 1,
                      stride = 1,
                      padding = 0)
            for i in range(self.num_down_layers)])
        
        ######################
        # Down sample blocks #
        ######################
        
        self.downsample_block = nn.Conv2d(in_channels = out_channels,
                                          out_channels = out_channels,
                                          kernel_size = 4,
                                          stride = 2,
                                          padding = 1) if down_sample_flag else nn.Identity()
        
        ###################################################################
        # Activate time embedding block if time embedding dim is not None # 
        ###################################################################
        
        if time_embedding_dim is not None:
            self.time_embedding_block = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(in_features = time_embedding_dim,
                              out_features = out_channels)
                    ) for _ in range(self.num_down_layers)])
        
        ############################################################
        # Activate attention blocks if down attention flag is True # 
        ############################################################
        
        if self.down_attn_flag:
            assert num_heads is not None, "[WARNING] Attention block is added. Hence, number of heads cannot be None."
            
            self.attention_norm_block = nn.ModuleList([
                nn.GroupNorm(num_groups = self.norm_channels,
                             num_channels = out_channels)
                for _ in range(self.num_down_layers)])
            
            self.attention_block = nn.ModuleList([
                nn.MultiheadAttention(embed_dim = out_channels,
                                      num_heads = num_heads,
                                      batch_first = True)
                for _ in range(self.num_down_layers)])
        
        ##########################################################
        # Activate cross attention block is context flag is True # 
        ##########################################################
        
        if self.context_flag:
            
            assert context_dim is not None, "[WARNING] Context is involved. Cross Attention block is added. Hence, context dimension cannot be None."
            assert num_heads is not None, "[WARNING] Context is involved. Cross Attention block is added. Hence, number of heads cannot be None."
            
            self.context_projection_block = nn.ModuleList([
                nn.Linear(in_features = context_dim,
                          out_features = out_channels)
                for _ in range(self.num_down_layers)])
            
            self.cross_attention_norm_block = nn.ModuleList([
                nn.GroupNorm(num_groups = self.norm_channels,
                             num_channels = out_channels)
                for _ in range(self.num_down_layers)])
            
            self.cross_attention_block = nn.ModuleList([
                nn.MultiheadAttention(embed_dim = out_channels,
                                      num_heads = num_heads,
                                      batch_first = True)
                for _ in range(self.num_down_layers)])
    
    def forward(self, x, time_embedding = None, context = None):
        
        """
        Forward propgation of a down block.
        
        Args:
            x (float tensor): Input tensor, (B, in_C, H, W), on cuda device. \n
            time_embedding (float tensor): Time embedding tensor, (B, time_embedding_dim), on cuda device. \n
            context (float tensor): Context tensor, to be used as Key and Value in Cross Attention blocks, (B, token_length, context_dimension), on cuda device. \n
        
        Returns:
            out (float tensor): Output tensor, (B, out_C, H, W) or (B, out_C, H/2, W/2) if down sample flag is True, on cuda device
        """
        
        out = x
        for i in range(self.num_down_layers):
            
            res_in = out
            out = self.first_resnet_block[i](out)
            if time_embedding is not None:
                out = out + self.time_embedding_block[i](time_embedding)[:, None, None]
            out = self.second_resnet_block[i](out)
            out = out + self.residual_conv_block[i](res_in)
            
            if self.down_attn_flag:
                
                attn_in = out
                
                B, C, H, W = out.shape
                out = out.reshape(B, C, H*W)
                out = self.attention_norm_block[i](out)
                out = torch.transpose(out, 1, 2)
                out, _ = self.attention_block[i](out, out, out)
                out = torch.transpose(out, 1, 2)
                out = out.reshape(B, C, H, W)
                
                out = attn_in + out
            
            if self.context_flag:
                
                assert context is not None, "[WARNING] Cross Attention block is added. Hence, context cannot be None."
                
                cross_attn_in = out
                
                B, C, H, W = out.shape
                out = out.reshape(B, C, H*W)
                out = self.cross_attention_norm_block[i](out)
                out = torch.transpose(out, 1, 2)
                
                context = self.context_projection_block[i](context)
                out, _ = self.cross_attention_block[i](out, context, context)
                out = torch.transpose(out, 1, 2)
                out = out.reshape(B, C, H, W)
                
                out = cross_attn_in + out
        
        out = self.downsample_block(out)
        return out







#####################
# M I D - B L O C K #
#####################




class mid_block(nn.Module):
    
    """
    A mid block, can be used to create an encoder-decoder or U-Net.
    
    Args: 
        config (dict): A config file in .yaml format, loaded using yaml.safe_load. \n
        in_channels (int): Input channels. \n
        out_channels (int): Output channels. \n
        time_embedding_dim (int): Time embedding dimension. Also decides if time embedding block is included in the architecture. (Default: None) \n
        mid_attn_flag (boolean): Mid attention flag. Decides if self-attention block is included in the architecture. (Default: False) \n
        num_heads (int): Number of heads in self-attention and cross-attention blocks. (Default: None) \n
        context_flag (boolean): Context flag. Decides if cross-attention blocks are included in the architecture. (Default: False) \n
        context_dim (int): Context dimension. (Default: None) \n
    """
    
    def __init__(self, config, in_channels, out_channels, time_embedding_dim = None, mid_attn_flag = False, num_heads = None,
                 context_flag = False, context_dim = None):
        super().__init__()
        
        self.norm_channels = config['norm_channels']
        self.num_mid_layers = config['num_mid_layers']
        self.mid_attn_flag = mid_attn_flag
        self.context_flag = context_flag
        
        #################
        # ResNet Blocks # 
        #################
        
        self.first_resnet_block = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups = self.norm_channels,
                             num_channels = in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels = in_channels if i == 0 else out_channels,
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1)
                ) for i in range(self.num_mid_layers + 1)])
        
        self.second_resnet_block = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups = self.norm_channels,
                             num_channels = out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels = out_channels,
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1)
                ) for _ in range(self.num_mid_layers + 1)])
        
        self.residual_conv_block = nn.ModuleList([
            nn.Conv2d(in_channels = in_channels if i == 0 else out_channels,
                      out_channels = out_channels,
                      kernel_size = 1,
                      stride = 1,
                      padding = 0)
            for i in range(self.num_mid_layers + 1)])
        
        
        ###################################################################
        # Activate time embedding block if time embedding dim is not None # 
        ###################################################################
        
        if time_embedding_dim is not None:
            self.time_embedding_block = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(in_features = time_embedding_dim,
                              out_features = out_channels)
                    ) for _ in range(self.num_mid_layers + 1)])
        
        ############################################################
        # Activate attention blocks if mid attention flag is True # 
        ############################################################
        
        if self.mid_attn_flag:
            assert num_heads is not None, "[WARNING] Attention block is added. Hence, number of heads cannot be None."
            
            self.attention_norm_block = nn.ModuleList([
                nn.GroupNorm(num_groups = self.norm_channels,
                             num_channels = out_channels)
                for _ in range(self.num_mid_layers)])
            
            self.attention_block = nn.ModuleList([
                nn.MultiheadAttention(embed_dim = out_channels,
                                      num_heads = num_heads,
                                      batch_first = True)
                for _ in range(self.num_mid_layers)])
        
        ##########################################################
        # Activate cross attention block is context flag is True # 
        ##########################################################
        
        if self.context_flag:
            
            assert context_dim is not None, "[WARNING] Context is involved. Cross Attention block is added. Hence, context dimension cannot be None."
            assert num_heads is not None, "[WARNING] Context is involved. Cross Attention block is added. Hence, number of heads cannot be None."
            
            self.context_projection_block = nn.ModuleList([
                nn.Linear(in_features = context_dim,
                          out_features = out_channels)
                for _ in range(self.num_mid_layers)])
            
            self.cross_attention_norm_block = nn.ModuleList([
                nn.GroupNorm(num_groups = self.norm_channels,
                             num_channels = out_channels)
                for _ in range(self.num_mid_layers)])
            
            self.cross_attention_block = nn.ModuleList([
                nn.MultiheadAttention(embed_dim = out_channels,
                                      num_heads = num_heads,
                                      batch_first = True)
                for _ in range(self.num_mid_layers)])
    
    def forward(self, x, time_embedding = None, context = None):
        
        """
        Forward propgation of a mid block.
        
        Args:
            x (float tensor): Input tensor, (B, in_C, H, W), on cuda device. \n
            time_embedding (float tensor): Time embedding tensor, (B, time_embedding_dim), on cuda device. \n
            context (float tensor): Context tensor, to be used as Key and Value in Cross Attention blocks, (B, token_length, context_dimension), on cuda device. \n
        
        Returns:
            out (float tensor): Output tensor, (B, out_C, H, W), on cuda device
        """
        
        out = x
        
        res_in = out
        out = self.first_resnet_block[0](out)
        if time_embedding is not None:
            out = out + self.time_embedding_block[0](time_embedding)[:, None, None]
        out = self.second_resnet_block[0](out)
        out = out + self.residual_conv_block[0](res_in)
        
        
        for i in range(self.num_mid_layers):
            
            if self.mid_attn_flag:
                
                attn_in = out
                
                B, C, H, W = out.shape
                out = out.reshape(B, C, H*W)
                out = self.attention_norm_block[i](out)
                out = torch.transpose(out, 1, 2)
                out, _ = self.attention_block[i](out, out, out)
                out = torch.transpose(out, 1, 2)
                out = out.reshape(B, C, H, W)
                
                out = attn_in + out
            
            if self.context_flag:
                
                assert context is not None, "[WARNING] Cross Attention block is added. Hence, context cannot be None."
                
                cross_attn_in = out
                
                B, C, H, W = out.shape
                out = out.reshape(B, C, H*W)
                out = self.cross_attention_norm_block[i](out)
                out = torch.transpose(out, 1, 2)
                
                context = self.context_projection_block[i](context)
                out, _ = self.cross_attention_block[i](out, context, context)
                out = torch.transpose(out, 1, 2)
                out = out.reshape(B, C, H, W)
                
                out = cross_attn_in + out
            
            res_in = out
            out = self.first_resnet_block[i+1](out)
            if time_embedding is not None:
                out = out + self.time_embedding_block[i+1](time_embedding)[:, None, None]
            out = self.second_resnet_block[i+1](out)
            out = out + self.residual_conv_block[i+1](res_in)
        
        return out
                
     





###################
# U P - B L O C K #
###################

class up_block(nn.Module):
    
    """
    A up block, can be used to create an encoder-decoder or U-Net.
    
    Args: 
        config (dict): A config file in .yaml format, loaded using yaml.safe_load. \n
        in_channels (int): Input channels. \n
        out_channels (int): Output channels. \n
        time_embedding_dim (int): Time embedding dimension. Also decides if time embedding block is included in the architecture. (Default: None) \n
        up_attn_flag (boolean): Up attention flag. Decides if self-attention block is included in the architecture. (Default: False) \n
        num_heads (int): Number of heads in self-attention and cross-attention blocks. (Default: None) \n
        context_flag (boolean): Context flag. Decides if cross-attention blocks are included in the architecture. (Default: False) \n
        context_dim (int): Context dimension. (Default: None) \n
        up_sample_flag (boolean): Up sample flag. Decides if up sample blocks are included in the architecture. (Default: True) \n
    """
    
    def __init__(self, config, in_channels, out_channels, time_embedding_dim = None, up_attn_flag = False, num_heads = None,
                 context_flag = False, context_dim = None, up_sample_flag = True):
        super().__init__()
        
        self.norm_channels = config['norm_channels']
        self.num_up_layers = config['num_up_layers']
        self.up_attn_flag = up_attn_flag
        self.context_flag = context_flag
        
        #################
        # ResNet Blocks # 
        #################
        
        self.first_resnet_block = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups = self.norm_channels,
                             num_channels = in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels = in_channels if i == 0 else out_channels,
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1)
                ) for i in range(self.num_up_layers)])
        
        self.second_resnet_block = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups = self.norm_channels,
                             num_channels = out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels = out_channels,
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1)
                ) for _ in range(self.num_up_layers)])
        
        self.residual_conv_block = nn.ModuleList([
            nn.Conv2d(in_channels = in_channels if i == 0 else out_channels,
                      out_channels = out_channels,
                      kernel_size = 1,
                      stride = 1,
                      padding = 0)
            for i in range(self.num_up_layers)])
        
        ######################
        # Up sample blocks #
        ######################
        
        self.upsample_block = nn.ConvTranspose2d(in_channels = in_channels,
                                                 out_channels = in_channels,
                                                 kernel_size = 4,
                                                 stride = 2, 
                                                 padding = 1) if up_sample_flag else nn.Identity()
        
        # When this architecture is created for a UNet, it expects that feature maps from down blocks will be passed on and concatenated
        # Therefore, the expected in_channels would be 2 times the ones given in config file
        # But this upsample block is only applied onto the feature maps in the up block streams, hence half of the in_channels is expected
        self.upsample_block_UNet = nn.ConvTranspose2d(in_channels = in_channels // 2,
                                                      out_channels = in_channels // 2,
                                                      kernel_size = 4,
                                                      stride = 2,
                                                      padding = 1) if up_sample_flag else nn.Identity()
        
        ###################################################################
        # Activate time embedding block if time embedding dim is not None # 
        ###################################################################
        
        if time_embedding_dim is not None:
            self.time_embedding_block = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(in_features = time_embedding_dim,
                              out_features = out_channels)
                    ) for _ in range(self.num_up_layers)])
        
        ############################################################
        # Activate attention blocks if up attention flag is True # 
        ############################################################
        
        if self.up_attn_flag:
            assert num_heads is not None, "[WARNING] Attention block is added. Hence, number of heads cannot be None."
            
            self.attention_norm_block = nn.ModuleList([
                nn.GroupNorm(num_groups = self.norm_channels,
                             num_channels = out_channels)
                for _ in range(self.num_up_layers)])
            
            self.attention_block = nn.ModuleList([
                nn.MultiheadAttention(embed_dim = out_channels,
                                      num_heads = num_heads,
                                      batch_first = True)
                for _ in range(self.num_up_layers)])
        
        ##########################################################
        # Activate cross attention block is context flag is True # 
        ##########################################################
        
        if self.context_flag:
            
            assert context_dim is not None, "[WARNING] Context is involved. Cross Attention block is added. Hence, context dimension cannot be None."
            assert num_heads is not None, "[WARNING] Context is involved. Cross Attention block is added. Hence, number of heads cannot be None."
            
            self.context_projection_block = nn.ModuleList([
                nn.Linear(in_features = context_dim,
                          out_features = out_channels)
                for _ in range(self.num_up_layers)])
            
            self.cross_attention_norm_block = nn.ModuleList([
                nn.GroupNorm(num_groups = self.norm_channels,
                             num_channels = out_channels)
                for _ in range(self.num_up_layers)])
            
            self.cross_attention_block = nn.ModuleList([
                nn.MultiheadAttention(embed_dim = out_channels,
                                      num_heads = num_heads,
                                      batch_first = True)
                for _ in range(self.num_up_layers)])
    
    def forward(self, x, down_features = None, time_embedding = None, context = None):
        
        """
        Forward propgation of a down block.
        
        Args:
            x (float tensor): Input tensor, (B, in_C, H/2, W/2) or (B, in_C/2, H/2, W/2) if down_features is not None, on cuda device. \n
            down_features (float tensor): Feature maps from down blocks, (B, in_C/2, H, W)
            time_embedding (float tensor): Time embedding tensor, (B, time_embedding_dim), on cuda device. \n
            context (float tensor): Context tensor, to be used as Key and Value in Cross Attention blocks, (B, token_length, context_dimension), on cuda device. \n
        
        Returns:
            out (float tensor): Output tensor, (B, out_C, H/2, W/2) or (B, out_C, H, W) if down sample flag is True, on cuda device
        """
        
        out = x
        
        if down_features == None:
            out = self.upsample_block(out)
        else:
            out = self.upsample_block_UNet(out)
            out = torch.cat([out, down_features], dim = 1)
        
        
        for i in range(self.num_up_layers):
            
            res_in = out
            out = self.first_resnet_block[i](out)
            if time_embedding is not None:
                out = out + self.time_embedding_block[i](time_embedding)[:, None, None]
            out = self.second_resnet_block[i](out)
            out = out + self.residual_conv_block[i](res_in)
            
            if self.up_attn_flag:
                
                attn_in = out
                
                B, C, H, W = out.shape
                out = out.reshape(B, C, H*W)
                out = self.attention_norm_block[i](out)
                out = torch.transpose(out, 1, 2)
                out, _ = self.attention_block[i](out, out, out)
                out = torch.transpose(out, 1, 2)
                out = out.reshape(B, C, H, W)
                
                out = attn_in + out
            
            if self.context_flag:
                
                assert context is not None, "[WARNING] Cross Attention block is added. Hence, context cannot be None."
                
                cross_attn_in = out
                
                B, C, H, W = out.shape
                out = out.reshape(B, C, H*W)
                out = self.cross_attention_norm_block[i](out)
                out = torch.transpose(out, 1, 2)
                
                context = self.context_projection_block[i](context)
                out, _ = self.cross_attention_block[i](out, context, context)
                out = torch.transpose(out, 1, 2)
                out = out.reshape(B, C, H, W)
                
                out = cross_attn_in + out

        return out