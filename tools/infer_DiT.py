import torch
import yaml
from tqdm import tqdm
from pathlib import Path
from models.VAE import VAE
from models.DiT import DiT
import os
from scheduler.linear_scheduler import linear_scheduler
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import argparse

def infer(config_file, load_dir):
    
    """
    Carry out the inference process for Diffusion Transformer. \n
    
    Args:
        config_path (path): A config with .yaml extension loaded by yaml.safe_load \n
        load_file (path): A directories that consist of the file for trained VAE, discriminator and optimizer. If provided, training will be resumed
                          from the provided check point in load_file\n

    """
    
    ####################
    # Read config file #
    ####################
    with open(config_file, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            tqdm.write(exc)
    
    tqdm.write("[INFO] The loaded config is as following: ")
    for key in config.keys():
        tqdm.write(f"{key}: {config[key]}")
    tqdm.write("\n")

    latent_flag = config['latent_flag']
    z_channels = config['z_channels']
    img_size = config['img_size']
    down_sample_flags = config['down_sample']
    autoencoder_load_file = config['autoencoder_load_file']
    num_timesteps = config['num_timesteps']
    
    # Calculate the expected latent size
    latent_size = img_size // (2 ** sum(down_sample_flags))
    

    ##########
    # Device #
    ##########
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ######################
    # Result save folder #
    ######################
    
    load_dir = Path(load_dir)
    
    result_dir = load_dir
    if not result_dir.is_dir():
        result_dir.mkdir(parents = True,
                         exist_ok = True)
    
    image_generation_dir = result_dir / 'Generated_Img'
    if not image_generation_dir.is_dir():
        image_generation_dir.mkdir(parents = True,
                                   exist_ok = True)
    
    ################
    # Create Model #
    ################
    # DiT
    model = DiT(config = config,
                img_channels = z_channels,
                img_size = latent_size).to(device)
    
    
    model_load_file = load_dir / 'DiT.pt'
    model.load_state_dict(torch.load(f = model_load_file,
                                     weights_only = True))
    
    model.eval()
    
    # VAE
    autoencoder = VAE(config = config).to(device)
    
    if os.path.exists(autoencoder_load_file):
        autoencoder.load_state_dict(torch.load(f = autoencoder_load_file,
                                               weights_only = True))
    autoencoder.eval()
    for params in autoencoder.parameters():
        params.requires_grad = False
        
    #####################################
    # Create training tools - scheduler #
    #####################################
    scheduler = linear_scheduler(config = config,
                                 device = device)
    
                                            ####################
                                            # Image Generation #
                                            ####################
    
    ##############################
    # Sample initial latent data #
    ##############################
    num_samples = 8
    sample_noise = torch.randn(num_samples, z_channels, latent_size, latent_size).to(device)
    out_latent_data = sample_noise
    
    ##########################
    # Reverse Diffusion Loop #
    ##########################
    with torch.inference_mode():
        for timestep in tqdm(list(reversed(range(num_timesteps)))):
            
            # Predict noise that was added at this particular time step
            # The timestep is expanded so each latent data in the batch share the same timestep
            pred_noise = model(out_latent_data, torch.tensor(timestep).unsqueeze(0))
            
            # Denoise the latent data
            out_latent_data = scheduler.reverse_noise(out_latent_data, pred_noise, timestep)
            
            # Decode and save the generated image for every 1000 steps
            if (timestep+1) % 100 == 0 or timestep == 0:
                
                # Decode the denoised latent data
                out_img = autoencoder.decode(out_latent_data)
                
                # Modify the image
                out_img = torch.clamp(out_img, -1, 1)
                out_img = ((out_img + 1) / 2).detach().cpu()
                
                # Create a grid image and convert to PIL images
                out_grid = make_grid(out_img,
                                     nrow = num_samples)
                out_grid_img = ToPILImage()(out_grid)
                
                # Save the images
                out_grid_img_file = image_generation_dir / f'{timestep}.png'
                out_grid_img.save(out_grid_img_file)
                tqdm.write(f"[INFO] The denoised images have been saveed into {out_grid_img_file}. ")



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str,
                        help = 'The config file in the format of .yaml')
    parser.add_argument('--load_dir', type = str,
                        help = 'The directories that consists of save files of models and checkpoints.')
    
    args = parser.parse_args()
    config_file = args.config
    load_dir = args.load_dir
    
    infer(config_file = config_file,
          load_dir = load_dir)