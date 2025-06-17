import torch
import yaml
from models.VAE import VAE
from pathlib import Path
import os
from dataset.custom_dataset import CELEB_dataset
import argparse
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

def infer(config_file, load_dir, reconstruction_flag, save_latent_flag, reconstruction_latent_flag):
    
    """
    Carry out the inference process for a Variational Autoencoder (VAE).
    
    Args:
        config_path (path): A config with .yaml extension loaded by yaml.safe_load \n
        load_file (path): A directories that consist of the file for trained VAE, discriminator and optimizer. If provided, training will be resumed
                          from the provided check point in load_file\n
        reconstruction_flag (boolean): Decides if executing image reconstruction mode. \n
        save_latent_flag (boolean): Decides if executing latent saving mode. \n
        reconstruction_latent_flag (boolean): Decides if reconstruct from randomly selected latent data. \n
        
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
    
    num_workers = config['ac_num_workers']
    latent_flag = config['latent_flag']
    

    ##########
    # Device #
    ##########
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ######################
    # Result save folder #
    ######################
    
    result_dir = Path('./result/VAE')
    if not result_dir.is_dir():
        result_dir.mkdir(parents = True,
                         exist_ok = True)
    
    reconstruction_dir = result_dir / 'reconstruction'
    if not reconstruction_dir.is_dir():
        reconstruction_dir.mkdir(parents = True,
                                 exist_ok = True)
        
    latent_data_dir = result_dir / 'latent_data'
    if not latent_data_dir.is_dir():
        latent_data_dir.mkdir(parents = True,
                              exist_ok = True)
    
    ################
    # Load dataset #
    ################
    
    train_path = Path('./data/CelebAMask-HQ/CelebA-HQ-img')
    train_data = CELEB_dataset(directory = train_path,
                               config = config,
                               latent_flag = latent_flag)
    
    #################
    # Create Models #
    #################
    
    # VAE
    model = VAE(config = config).to(device)
    
    ###############
    # Load models #
    ###############
    if load_dir is not None:
        
        load_dir = Path(load_dir)
        
        load_model_file = load_dir / 'VAE.pt'
        if os.path.exists(load_model_file):
            tqdm.write("[INFO] VAE model save file is found. Proceed to load it....")
            model.load_state_dict(torch.load(f = load_model_file,
                                             weights_only = True))
    
                                    ####################################################################
                                    # Reconstruction Mode - Randomly select few images and reconstruct #
                                    ####################################################################
    
    if reconstruction_flag:
        
        ################################################
        # Randomly select original images from dataset #
        ################################################
        
        num_samples = 8
        rand_num = torch.randint(0, len(train_data), (num_samples,))
        original_img = torch.cat([train_data[num][None,:] for num in rand_num], dim = 0) # (8, img_channels, img_size, img_size)
        
        ##########################################
        # Forward propagation for reconstruction #
        ##########################################
        
        model.eval()
        with torch.inference_mode():
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
            grid_save_file = reconstruction_dir / 'reconstruction.png'
            grid_img.save(grid_save_file)
            tqdm.write(f"[INFO] Reconstructed images have been successfully saved into {grid_save_file}.")
    
    
                                                    #######################################
                                                    # Save Latent Mode - Save latent data #
                                                    #######################################      
            
    if save_latent_flag:
        
        #########################################################
        # Create dataloader without shuffle and batch size of 1 #
        #########################################################
        
        train_dataloader = DataLoader(dataset = train_data,
                                      batch_size = 1,
                                      shuffle = False,
                                      num_workers = num_workers)
        
        ###################################################
        # Setup place holder to keep track of latent data #
        ###################################################
        
        latent_data_count = 0 # For different file name. Every 1000 latent data saved into one pickle file.
        latent_data = {} # A dict {path : latent_data}
        
        ############################################
        # Encoding - Save the mean and log variace #
        ############################################
        
        model.eval()
        with torch.inference_mode():
            
            for num, img in enumerate(tqdm(train_dataloader)):
                
                ####################################
                # Forward propagation for encoding #
                ####################################
                img = img.to(device)
                _, mean_log_var = model.encode(img)
                
                #######################################
                # Assign mean_log_var_to a dictionary #
                #######################################
                save_key = train_data.path_list[num] # Obtain the path of this image (Valid since the data loader is not shuffle)
                latent_data[save_key] = mean_log_var[0].detach().cpu() # Remove the batch dimensionality with [0]
                
                ######################################################
                # Dump into a pickle file for every 1000 latent data #
                ######################################################
                if (num+1) % 1000 == 0:
            
                    latent_data_save_file = latent_data_dir / f'latent_data_{latent_data_count}.pkl'
                    
                    with open(latent_data_save_file, 'wb') as f:
                        pickle.dump(obj = latent_data,
                                    file = f)
                        tqdm.write(f"[INFO] Latent data has been successfully saved into {latent_data_save_file}.")
                        
                    latent_data_count += 1
                    latent_data = {}
            
            ############################################################################
            # Dump remaining latent data into a pickle file for every 1000 latent data #
            ############################################################################
            
            if len(latent_data) > 0:
                
                latent_data_save_file = latent_data_dir / f'latent_data_{latent_data_count}.pkl'
                
                with open(latent_data_save_file, 'wb') as f:
                    pickle.dump(obj = latent_data,
                                file = f)
                    tqdm.write(f"[INFO] Latent data has been successfully saved into {latent_data_save_file}.")
                    
                latent_data_count += 1
                latent_data = {}
                
                
                
                                ##################################################################################
                                # Reconstruction Latent Mode - Reconstruction from randomly selected latent data #
                                ##################################################################################
                
    if reconstruction_latent_flag:       
        
        ###################################
        # Randomly select original images #
        ###################################
        
        num_samples = 8
        rand_num = torch.randint(0, len(train_data), (num_samples,))
        original_img = torch.cat([train_data[num][None,:] for num in rand_num], dim = 0) # (8, img_channels, img_size, img_size)
        
        ###################################
        # Select their corresponding path #
        ###################################
        original_img_path = [train_data.path_list[num] for num in rand_num]
        
        #######################################################
        # Find and concatentate the latent data from the path #
        #######################################################
        test_latent_data = None
        for i in range(len(original_img_path)):
            
            # Locate the corresponding latent data file
            latent_data_count = int(rand_num[i] / 1000)
            latent_data_file = latent_data_dir / f'latent_data_{latent_data_count}.pkl'
            
            # Load the latent data file
            with open(latent_data_file, 'rb') as f:
                latent_data = pickle.load(f)
            
            # Concatentate them
            if test_latent_data == None:
                test_latent_data = latent_data[original_img_path[i]][None, :]
            else:   
                test_latent_data = torch.cat([test_latent_data, latent_data[original_img_path[i]][None, :]], dim=0) # (8, z_channels, latent_size, latent_size)
        
        ####################################
        # Forward propagation for decoding #
        ####################################
        model.eval()
        with torch.inference_mode():
            
            ###########################################
            # Sample a data from the mean and log var #
            ###########################################
            test_latent_data = test_latent_data.to(device)
            mean, log_var = torch.chunk(test_latent_data, chunks = 2, dim = 1)
            std = torch.exp(log_var) ** 0.5
            z = torch.randn_like(mean)
            sample = mean + std*z
            
            ###########################
            # Decode the sampled data #
            ###########################
            recon_img = model.decode(sample)
        
            ###########################################
            # Save original and reconcstructed images #
            ###########################################
            
            # Preproces original images
            out_img = ((original_img + 1) / 2)
            
            # Preprocess construction images
            out_reconstruction = torch.clamp(recon_img, -1, 1)
            out_reconstruction = ((out_reconstruction + 1) / 2).detach().cpu()
            
            # Concatenate the images and form grid
            output = torch.cat([out_img, out_reconstruction], dim = 0)
            grid = make_grid(output,
                             nrow = num_samples)
            grid_img = ToPILImage()(grid)
            
            # Save grid images
            grid_save_file = reconstruction_dir / 'reconstruction_using_latent_data.png'
            grid_img.save(grid_save_file)
            tqdm.write(f"[INFO] Reconstructed images have been successfully saved into {grid_save_file}.")
                




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str,
                        help = 'The config file in the format of .yaml')
    parser.add_argument('--load_dir', type = str,
                        help = 'The directories that consists of save files of models and checkpoints.')
    parser.add_argument('--reconstruction_flag', action = 'store_true',
                        help = 'Decides if executing image reconstruction mode.')
    parser.add_argument('--save_latent_flag', action = 'store_true',
                        help = 'Decides if executing latent saving mode.')
    parser.add_argument('--reconstruction_latent_flag', action = 'store_true',
                        help = 'Decides if reconstruct from randomly selected latent data.')
    
    args = parser.parse_args()
    config_file = args.config
    load_dir = args.load_dir
    reconstruction_flag = args.reconstruction_flag
    save_latent_flag = args.save_latent_flag
    reconstruction_latent_flag = args.reconstruction_latent_flag
    
    infer(config_file = config_file,
          load_dir = load_dir,
          reconstruction_flag = reconstruction_flag,
          save_latent_flag = save_latent_flag,
          reconstruction_latent_flag = reconstruction_latent_flag)
