import yaml
import torch
import argparse
from dataset.custom_dataset import CELEB_dataset
from pathlib import Path
import numpy as np
import random
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from models.VAE import VAE
from models.discriminator import discriminator 
from models.lpips import LPIPS
from torchinfo import summary
import torch.nn as nn
from tqdm import tqdm
import os
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

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
    
    latent_flag = config['latent_flag']
    num_workers = config['ac_num_workers']
    batch_size = config['ac_batch_size']
    epochs = config['ac_epochs']
    lr = config['ac_lr']
    acc_steps = config['ac_acc_steps']
    disc_start_steps = config['disc_start_steps']
    adv_weight = config['adv_weight']
    perceptual_weight = config['perceptual_weight']
    kl_weight = config['kl_weight']
    img_save_steps = config['img_save_steps']
    sample_size = config['sample_size']
    
    ###################
    # Device and Seed #
    ###################
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        
    ######################
    # Result save folder #
    ######################
    
    result_dir = Path('./result/VAE')
    if not result_dir.is_dir():
        result_dir.mkdir(parents = True,
                         exist_ok = True)
    
    dataset_test_dir = result_dir / 'dataset_test'
    if not dataset_test_dir.is_dir():
        dataset_test_dir.mkdir(parents = True,
                               exist_ok = True)
        
    training_test_dir = result_dir / 'training_test'
    if not training_test_dir.is_dir():
        training_test_dir.mkdir(parents = True,
                                exist_ok = True)
    
    model_save_file = result_dir / 'VAE.pt'
    disc_save_file = result_dir / 'discriminator.pt'
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
    
    if latent_flag:
        rand_num = torch.randint(0, len(train_data), (1,))
        train_img = train_data[rand_num]
      
    else:
        rand_num = torch.randint(0, len(train_data), (9,))
        for idx, num in enumerate(rand_num):
            train_img = train_data[num]
            
            train_img_plt = (train_img + 1) / 2 # Convert into range of 0.0 to 1.0
            train_img_plt = train_img_plt.permute(1,2,0) # Convert into (H,W,C)
            
            plt.subplot(3,3, idx+1)
            plt.imshow(train_img_plt)
            plt.axis(False)
        
        plt.tight_layout()
        save_img_file = dataset_test_dir / 'dataset_test.png'
        plt.savefig(save_img_file)
        tqdm.write(f"[INFO] 9 images have been selected from the dataset and plotted. The plot is saved in {save_img_file}.")
    
    
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
    
    train_img_batch = next(iter(train_dataloader))
    
    tqdm.write("[INFO] The loaded data loader is as following: ")
    tqdm.write(f"Total number of batches in the data loader: {len(train_dataloader)}")
    tqdm.write(f"The number of images in a batch: {train_img_batch.shape[0]}")
    tqdm.write(f"The size of an image in the batch: {train_img_batch[0].shape}")
    tqdm.write(f"The range of value in the image: {train_img_batch[0].min()} to {train_img_batch[0].max()}")
    tqdm.write("\n")
    
    
    #################
    # Create Models #
    #################
    
    # VAE
    model = VAE(config = config).to(device)
    
    summary(model = model,
            input_size = (1, config['img_channels'], config['img_size'], config['img_size']),
            col_names = ['input_size', 'output_size', 'num_params', 'trainable'],
            row_settings = ['var_names'])
    
    
    # Discriminator
    model_disc = discriminator(config = config).to(device)
    
    summary(model = model_disc,
            input_size = (1, config['img_channels'], config['img_size'], config['img_size']),
            col_names = ['input_size', 'output_size', 'num_params', 'trainable'],
            row_settings = ['var_names'])
    
    
    # LPIPS
    # No need to freeze the model because it has been taken care of in lpips.py
    model_lpips = LPIPS().to(device)
    
    
    
    #################################################
    # Training tools - Optimizer and Loss Functions #
    #################################################
    
    # Optimizer for the generator / VAE
    optimizer_g = torch.optim.Adam(params = model.parameters(),
                                   lr = lr,
                                   betas = (0.5, 0.999))
    
    # Optimier for the discriminator
    optimizer_d = torch.optim.Adam(params = model_disc.parameters(),
                                   lr = lr,
                                   betas = (0.5, 0.999))
    
    # Reconstruction loss function - Mean Squared Error
    recon_loss_fn = nn.MSELoss()
    
    # Discriminator loss function - Binary Cross Entropy Loss
    disc_loss_fn = nn.BCEWithLogitsLoss()
    
    
    
    ###############
    # Load models #
    ###############
    last_epoch = 0
    last_step = 0
    if load_dir is not None:
        
        load_dir = Path(load_dir)
        
        load_model_file = load_dir / 'VAE.pt'
        if os.path.exists(load_model_file):
            tqdm.write("[INFO] VAE model save file is found. Proceed to load it....")
            model.load_state_dict(torch.load(f = load_model_file,
                                             weights_only = True))
        
        load_model_disc_file = load_dir / 'discriminator.pt'
        if os.path.exists(load_model_disc_file):
            tqdm.write("[INFO] Discriminator model save file is found. Proceed to load it....")
            model_disc.load_state_dict(torch.load(f = load_model_disc_file,
                                                  weights_only = True))
        
        load_checkpoint_file = load_dir / 'ckpt.pt'
        if os.path.exists(load_checkpoint_file):
            tqdm.write("[INFO] Checkpoint is found. Proceed to load it....")
            last_checkpoint = torch.load(f = load_checkpoint_file)
            
            tqdm.write("The loaded checkpoint is as following: ")
            for key in last_checkpoint.keys():
                if key != 'optimizer_g' and key != 'optimizer_d':
                    tqdm.write(f"{key}: {last_checkpoint[key]}")
            
            tqdm.write("Loading the optimizer....")
            optimizer_g.load_state_dict(last_checkpoint['optimizer_g'])
            optimizer_d.load_state_dict(last_checkpoint['optimizer_d'])
            
            last_epoch = last_checkpoint['epoch'] + 1 
            last_step = last_epoch * len(train_dataloader) # To determine if discriminator to be activated or not
            
            
            
    

    #################
    # Training Loop #
    #################
    
    step = 0
    step = step + last_step
    
    model.train()
    model_disc.train()
    model_lpips.eval()
    
    optimizer_g.zero_grad()
    optimizer_d.zero_grad()
    
    for epoch in tqdm(range(epochs - last_epoch)):
        
        epoch = epoch + last_epoch
        
        reconstruction_loss_list = []
        kl_loss_list = []
        perceptual_loss_list = []
        g_adv_loss_list = []
        g_loss_list = []
        d_adv_loss_list = []
        
        for batch, img in enumerate(tqdm(train_dataloader)):
            
            step += 1
            
            ##########################################
            # Forward propagation for reconstruction #
            ##########################################
            img = img.to(device)
            reconstruction, mean_log_var = model(img)
            
            
            ##########################################
            # Image Generation for Progression Check #
            ##########################################
            
            if step % img_save_steps == 0 or step == 1:
                
                # Pick the smaller, between the config value or the current batch value as number of imgs to be sampled
                num_img = min(sample_size, img.shape[0])
                
                # Preproces original images
                out_img = ((img[:num_img] + 1) / 2).detach().cpu()
                
                # Preprocess construction images
                out_reconstruction = torch.clamp(reconstruction[:num_img], -1, 1)
                out_reconstruction = ((out_reconstruction + 1) / 2).detach().cpu()
                
                # Concatenate the images and form grid
                output = torch.cat([out_img, out_reconstruction], dim = 0)
                grid = make_grid(output,
                                 nrow = num_img)
                grid_img = ToPILImage()(grid)
                
                # Save grid images
                grid_save_file = training_test_dir / f'{step}.png'
                grid_img.save(grid_save_file)
                
            
                                ####################
                                #### Generator #####
                                ####################
            
            #######################
            # Reconstruction Loss #
            #######################
            reconstruction_loss = recon_loss_fn(reconstruction, img)
            
            reconstruction_loss_list.append(reconstruction_loss.item()) # Store loss values
            
            ###########
            # KL Loss #
            ###########
            mean, log_var = torch.chunk(mean_log_var, chunks = 2, dim = 1)
            var = torch.exp(log_var)
            kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - (mean ** 2) - var, dim = [1,2,3]))
            
            kl_loss_list.append(kl_loss) # Store loss values
            
            ###################
            # Perceptual Loss #
            ###################
            perceptual_loss = torch.mean(model_lpips(img, reconstruction))
            
            perceptual_loss_list.append(perceptual_loss) # Store loss values
            
            ####################
            # Adversarial Loss #
            ####################
            g_adv_loss = 0
            if step > disc_start_steps:
                g_pred = model_disc(reconstruction)
                g_adv_loss = disc_loss_fn(g_pred, torch.ones_like(g_pred))
                
                g_adv_loss_list.append(g_adv_loss.item()) # Store loss values
                
            ##################
            # Generator Loss #
            ##################
            g_loss = reconstruction_loss + \
                        kl_loss * kl_weight + \
                            perceptual_loss * perceptual_weight + \
                                g_adv_loss * adv_weight
            
            g_loss_list.append(g_loss.item()) # Store loss values
            
            g_loss = g_loss / acc_steps
            g_loss.backward()
            
            #################################
            # Backpropagation for Generator #
            #################################
            if step % acc_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()
            
            
                                #######################
                                #### Discriminator ####
                                #######################
            
            if step > disc_start_steps:
                
                # Evaluate real images
                d_real_pred = model_disc(img)
                d_adv_real_loss = disc_loss_fn(d_real_pred, torch.ones_like(d_real_pred))
                
                # Evaluate fake / reconstructed images
                d_fake_pred = model_disc(reconstruction.detach())
                d_adv_fake_loss = disc_loss_fn(d_fake_pred, torch.zeros_like(d_fake_pred))
                
                # Average the 2 losses
                d_adv_loss = 0.5 * (d_adv_real_loss + d_adv_fake_loss) 
                d_adv_loss_list.append(d_adv_loss.item()) # Store loss values
                
                # Multiply with the weight
                d_adv_loss = d_adv_loss * adv_weight
                
                #####################################
                # Backpropagation for Discriminator #
                #####################################
                d_adv_loss = d_adv_loss / acc_steps
                d_adv_loss.backward()
                if step % acc_steps == 0:
                    optimizer_d.step()
                    optimizer_d.zero_grad()
        
        #############################################################
        # Final Backpropagation in case there are gradients remnant #
        #############################################################
        optimizer_g.step()
        optimizer_g.zero_grad()
        
        optimizer_d.step()
        optimizer_d.zero_grad()
                    
        ############################
        # Calculate loss per epoch #
        ############################
        reconstruction_loss_per_epoch = sum(reconstruction_loss_list) / len(reconstruction_loss_list)
        kl_loss_per_epoch = sum(kl_loss_list) / len(kl_loss_list)
        perceptual_loss_per_epoch = sum(perceptual_loss_list) / len(perceptual_loss_list)
        g_loss_per_epoch = sum(g_loss_list) / len(g_loss_list)
        
        if len(g_adv_loss_list) > 0:
            g_adv_loss_per_epoch = sum(g_adv_loss_list) / len(g_adv_loss_list)
        else:
            g_adv_loss_per_epoch = 0
        
        if len(d_adv_loss_list) > 0:
            d_adv_loss_per_epoch = sum(d_adv_loss_list) / len(d_adv_loss_list)
        else:
            d_adv_loss_per_epoch = 0
            
        ########################
        # tqdm.write loss per epoch #
        ########################
        tqdm.write(f"[INFO] Current epoch: {epoch}")
        tqdm.write(f"Reconstruction Loss            : {reconstruction_loss_per_epoch:.6f}")
        tqdm.write(f"KL Loss                        : {kl_loss_per_epoch:.6f}")
        tqdm.write(f"Perceptual Loss                : {perceptual_loss_per_epoch:.6f}")
        tqdm.write(f"Generator Adversarial Loss     : {g_adv_loss_per_epoch:.6f}")
        tqdm.write(f"Generator Loss                 : {g_loss_per_epoch:.6f}")
        tqdm.write(f"Discriminator Adversarial Loss : {d_adv_loss_per_epoch:.6f}")
        
        ####################
        # Save check point #
        ####################
        
        # Save VAE
        torch.save(obj = model.state_dict(),
                   f = model_save_file)
        
        # Save discriminator
        torch.save(obj = model_disc.state_dict(),
                   f = disc_save_file)
        
        # Save checkpoint
        checkpoint = {'epoch': epoch,
                      'optimizer_g': optimizer_g.state_dict(),
                      'optimizer_d': optimizer_d.state_dict(),
                      'reconstruction_loss': reconstruction_loss_per_epoch,
                      'kl_loss': kl_loss_per_epoch,
                      'perceptual_loss': perceptual_loss_per_epoch,
                      'generator_adversarial_loss': g_adv_loss_per_epoch,
                      'generator_loss': g_loss_per_epoch,
                      'discriminator_adversarial_loss': d_adv_loss_per_epoch}
        
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

