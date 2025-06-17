import torch
import torch.nn as nn
from tqdm import tqdm
import yaml
import argparse
from pathlib import Path
from dataset.custom_dataset import CELEB_dataset
from torch.utils.data import DataLoader
from models.DiT import DiT
from scheduler.linear_scheduler import linear_scheduler
from torchinfo import summary
import os
from models.VAE import VAE
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

def test_autoencoder(model, train_data, device, reconstruction_dir):
    """
    To test if the loaded autoencoder is performing up to expectation through reconstruction of images.
    
    Args:
        model (nn.Module): An autoencoder with loaded weight. (Expected the model is in eval mode and is frozen). \n
        train_data (Dataset): A dataset consists of images to be reconstructed. \n
        device (str): 'cpu' or 'cuda'
        reconstruction_dir (path): A directory to store the reconstructed image. \n
    """
    
    num_samples = 8
    rand_num = torch.randint(0, len(train_data), (num_samples,))
    original_img = torch.cat([train_data[num][None,:] for num in rand_num], dim = 0) # (8, img_channels, img_size, img_size)
    
    ##########################################
    # Forward propagation for reconstruction #
    ##########################################
    
    original_img = original_img.to(device)
    recon_img, _ = model(original_img)
    
    ###########################################
    # Save original and reconcstructed images #
    ###########################################
    
    # Preproces original images
    out_img = ((original_img + 1) / 2).detach().cpu()
    
    # Preprocess construction images
    out_reconstruction = torch.clamp(recon_img, -1, 1)
    out_reconstruction = ((out_reconstruction + 1) / 2).detach().cpu()
    
    # Concatenate the images and form grid
    output = torch.cat([out_img, out_reconstruction], dim = 0)
    grid = make_grid(output,
                     nrow = num_samples)
    grid_img = ToPILImage()(grid)
    
    # Save grid images
    grid_save_file = reconstruction_dir / 'test_autoencoder.png'
    grid_img.save(grid_save_file)
    tqdm.write(f"[INFO] Reconstructed images have been successfully saved into {grid_save_file}.")



def train(config_file, load_dir = None):
    
    """
    Carry out the training process
    
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
    
    num_workers = config['dit_num_workers']
    batch_size = config['dit_batch_size']
    epochs = config['dit_epochs']
    lr = config['dit_lr']
    acc_steps = config['dit_acc_steps']
    latent_flag = config['latent_flag']
    z_channels = config['z_channels']
    img_size = config['img_size']
    down_sample_flags = config['down_sample']
    num_timesteps = config['num_timesteps']
    autoencoder_load_file = config['autoencoder_load_file']
    
    # Calculate the expected latent size
    latent_size = img_size // (2 ** sum(down_sample_flags))

    
    ##########
    # Device #
    ##########
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
        
    ######################
    # Result save folder #
    ######################
    
    result_dir = Path('./result/DiT_5')
    if not result_dir.is_dir():
        result_dir.mkdir(parents = True,
                         exist_ok = True)
        
    reconstruction_dir = result_dir / 'Reconstruction'
    if not reconstruction_dir.is_dir():
        reconstruction_dir.mkdir(parents = True,
                                 exist_ok = True)
        
    model_save_file = result_dir / 'DiT.pt'
    checkpoint_file = result_dir / 'ckpt.pt'

    
    ################
    # Load dataset #
    ################
    
    train_path = Path('./data/CelebAMask-HQ/CelebA-HQ-img')
    train_data = CELEB_dataset(directory = train_path,
                               config = config,
                               latent_flag = latent_flag)
    
    ################
    # Test dataset #
    ################
    
    rand_num = torch.randint(0, len(train_data), (1,))
    train_img = train_data[rand_num]

    
    tqdm.write("[INFO] The loaded dataset is as following: ")
    tqdm.write(f"The number of images in the dataset: {len(train_data)}")
    tqdm.write(f"The size of an image in the dataset: {train_img.shape}")
    tqdm.write(f"The range of value in the image: {train_img.min()} to {train_img.max()}")
    tqdm.write("\n")
    
    ###################
    # Test dataloader #
    ###################
    
    train_dataloader = DataLoader(dataset = train_data,
                                  batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = num_workers)
    
    # Comment out because taking too long
# =============================================================================
#     train_img_batch = next(iter(train_dataloader))
#     
#     tqdm.write("[INFO] The loaded data loader is as following: ")
#     tqdm.write(f"Total number of batches in the data loader: {len(train_dataloader)}")
#     tqdm.write(f"The number of images in a batch: {train_img_batch.shape[0]}")
#     tqdm.write(f"The size of an image in the batch: {train_img_batch[0].shape}")
#     tqdm.write(f"The range of value in the image: {train_img_batch[0].min()} to {train_img_batch[0].max()}")
#     tqdm.write("\n")
# =============================================================================
    
    ################
    # Create Model #
    ################
    
    model = DiT(config = config,
                img_channels = z_channels,
                img_size = latent_size).to(device)
    
    summary(model = model,
            input_size = (1, z_channels, latent_size, latent_size),
            col_names = ['input_size', 'output_size', 'num_params', 'trainable'],
            row_settings = ['var_names'])
    
    if not latent_flag:
        autoencoder = VAE(config = config).to(device)
        
        if os.path.exists(autoencoder_load_file):
            autoencoder.load_state_dict(torch.load(f = autoencoder_load_file,
                                                   weights_only = True))

        autoencoder.eval()
        for params in autoencoder.parameters():
            params.requires_grad = False
        
        test_autoencoder(model = autoencoder, 
                         train_data = train_data, 
                         device = device, 
                         reconstruction_dir = reconstruction_dir)
    ###################################################################
    # Create training tools - Loss function, optimizers and scheduler #
    ###################################################################
    
    # Optimizer - Adam
# =============================================================================
#     optimizer = torch.optim.Adam(params = model.parameters(),
#                                  lr = lr)
# =============================================================================
    
    optimizer = torch.optim.AdamW(params = model.parameters(),
                                  lr = lr,
                                  weight_decay = 0.01)
    
    # Loss - Mean Squared Error Loss
    loss_fn = nn.MSELoss()
    
    # Scheduler - Linear Scheduler
    scheduler = linear_scheduler(config = config,
                                 device = device)
    
    ##############################################
    # Load previouly trained model and optimizer #
    ##############################################
    last_epoch = 0
    last_step = 0
    if load_dir is not None:
        
        load_dir = Path(load_dir)
        
        model_load_file = load_dir / 'DiT.pt'
        if os.path.exists(model_load_file):
            model.load_state_dict(torch.load(f = model_load_file,
                                         weights_only = True))
            tqdm.write("[INFO] The model from previous training has been loaded successfully.")
        
        
        checkpoint_load_file = load_dir / 'ckpt.pt'
        if os.path.exists(checkpoint_load_file):
            loaded_checkpoint = torch.load(checkpoint_load_file)
            tqdm.write("[INFO] The loaded checkpoint is as following: ")
            for key in loaded_checkpoint.keys():
                if key != 'optimizer':
                    print(f"{key}: {loaded_checkpoint[key]}")
                    
            last_epoch = loaded_checkpoint['epoch'] + 1
            last_step = last_epoch * len(train_dataloader)
        
    
    
    #################
    # Training loop #
    #################
    step = 0
    step += last_step
    model.train()
    optimizer.zero_grad()
    
    for epoch in range(epochs - last_epoch):
        
        loss_list = []
        epoch = epoch + last_epoch
        
        for batch, image in enumerate(tqdm(train_dataloader)):
            
            step += 1
            
            ###########################################
            # Sample a data from the mean and log var #
            ###########################################
            
            if latent_flag:
                latent_data = image
                
                latent_data = latent_data.to(device)
                mean, log_var = torch.chunk(latent_data, chunks = 2, dim = 1)
                std = torch.exp(log_var) ** 0.5
                z = torch.randn_like(mean)
                sample = mean + std*z
            else:
                with torch.inference_mode():
                    image = image.to(device)
                    sample, _ = autoencoder.encode(image)
                
            ##################################################################
            # Sample a noise and timestep, add the noise to the input sample #
            ##################################################################
            noise = torch.randn_like(sample)
            timestep = torch.randint(0, num_timesteps, (sample.shape[0],))
            noisy_sample = scheduler.add_noise(sample, noise, timestep)
            
            ####################################
            # Predict the noise that was added #
            ####################################
            pred_noise = model(noisy_sample, timestep)
            
            ######################################
            # Loss computation & Backpropagation #
            ######################################
            loss = loss_fn(pred_noise, noise)
            loss_list.append(loss.item())
            
            loss = loss / acc_steps
            loss.backward()
            
            ####################
            # Update parameter #
            ####################
            if step % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
        
        #########################################
        # Handle leftover gradients by updating #
        #########################################
        optimizer.step()
        optimizer.zero_grad()
        
        ########################################
        # Announce training performance (loss) #
        ########################################
        loss_per_epoch = sum(loss_list) / len(loss_list)
        
        tqdm.write(f"[INFO] Current Epoch: {epoch}")
        tqdm.write(f"Training Loss: {loss_per_epoch:.4f}")
        
        ##############################
        # Save models and checkpoint #
        ##############################
        torch.save(obj = model.state_dict(),
                   f = model_save_file)
        
        checkpoint = {'epoch': epoch,
                      'optimizer': optimizer.state_dict(),
                      'loss': loss_per_epoch}
        torch.save(obj = checkpoint,
                   f = checkpoint_file)
        
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str,
                        help = 'The config file in the format of .yaml')
    parser.add_argument('--load_dir', type = str,
                        help = 'The directories that consists of save files of models and checkpoints.')
    
    args = parser.parse_args()
    config_file = args.config
    load_dir = args.load_dir
    
    train(config_file = config_file,
          load_dir = load_dir)