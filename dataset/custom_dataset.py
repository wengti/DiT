
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
from utils.diffusion_utils import merge_latent_data

class CELEB_dataset(Dataset):
    """
    Create a custom dataset for CELEB-HQ.
    
    Args:
        directory (path): Expecting a directory that consists of all the images of the dataset. \n
        config (dict): Expecting a config file loaded using yaml.safe_load. \n
        latent_flag (boolean): Decides whether to output latent data. \n
    """
    
    def __init__(self, directory, config, latent_flag):
        directory = Path(directory)
        self.path_list = list(directory.glob("*.jpg"))
        self.config = config
        self.latent_flag = latent_flag
        
    def load_image(self, index):
        img = Image.open(self.path_list[index])
        return img
    
    def __len__(self):
        return len(self.path_list)
        
    def __getitem__(self, index):
        
        if self.latent_flag:
            
            # Obtain the directories that store all the latent data file
            self.latent_dir = self.config['latent_dir']
            assert self.latent_dir is not None, "To use latent data, the diretories to access the latent data must be provided."
            
            # Merge all the latent data from different files into 1 data structure (dictionary)
            full_latent_data = merge_latent_data(self.latent_dir)
            
            # From index -> Find the path name -> Find the latent data using the path name
            img_path = self.path_list[index]
            latent_data = full_latent_data[img_path]
            
            return latent_data
            
        
        else:
            # Load the image using PIL
            img = self.load_image(index)
            
            # Transform into Tensors and Resize into the specified shape in config
            # This transformation returns a tensor in the shape of (C, H, W) in the range of 0.0 to 1.0
            img_size = self.config['img_size']
            simple_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize((img_size, img_size)),
                                                   transforms.CenterCrop((img_size, img_size))])
            img = simple_transform(img)
            
            # This transformation returns a tensor in the shape of (C, H, W) in the range of -1.0 to 1.0
            img = (img*2) - 1
            
            return img
            

