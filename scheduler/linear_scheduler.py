import torch

class linear_scheduler():
    
    """
    Linear scheduler. Used to add noise in Forward Diffusion and remove noise in Reverse Diffusion.
    
    Args:
        config (dict): A config file in .yaml format, loaded using yaml.safe_load. \n
    """
    
    def __init__(self, config, device):
        
        self.beta_start = config['beta_start']
        self.beta_end = config['beta_end']
        self.num_timesteps = config['num_timesteps']
        
        self.beta = torch.linspace(start = self.beta_start,
                                   end = self.beta_end,
                                   steps = self.num_timesteps).to(device)
        
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim = 0)
        
    
    def add_noise(self, x, noise, timestep):
        """
        Add noise into a batch of input images. Used in Forward Diffusion.
        
        Args:
            x (float tensor): A batch of input images, (B, C, H, W). \n
            noise (float tensor) : A batch of noise to be added, (B, C, H ,W). \n
            timestep (int tensor) : A batch of timestep, (B, ). \n
        
        Returns:
            noisy_out (float tensor): A batch of output images with added noise, (B, C, H, W). \n
        """
        
        
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat[timestep])[:, None, None, None]
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[timestep])[:, None, None, None]
        
        noisy_out = self.sqrt_alpha_hat * x + self.sqrt_one_minus_alpha_hat * noise
        return noisy_out
        

    def reverse_noise(self, x, noise, timestep):
        """
        Remove noise from a batch of input images. Used in Reverse Diffusion.
        
        Args:
            x (float tensor): A batch of input images, (B, C, H, W). \n
            noise (float tensor) : A batch of noise to be removed, (B, C, H ,W). \n
            timestep (int) : A single timestep, (B, ). \n
        
        Returns:
            denoised_out (float tensor): A batch of denoised output images. (B, C, H, W). \n
        """
        
        
        mean = (1 - self.alpha[timestep]) / (torch.sqrt(1 - self.alpha_hat[timestep])) * noise
        mean = x - mean
        mean = mean / torch.sqrt(self.alpha[timestep])
        
        if timestep == 0:
            return mean
        
        else:
            variance = (1 - self.alpha[timestep]) * (1 - self.alpha_hat[timestep - 1]) / (1 - self.alpha_hat[timestep])
            std_dev = torch.sqrt(variance)
            z = torch.randn_like(mean)
            return mean + std_dev * z




