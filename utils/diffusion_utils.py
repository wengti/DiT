from pathlib import Path
import pickle

def merge_latent_data(latent_dir):
    
    """
    Merge all the latent_data in different file into 1 single data.
    
    Args:
        latent_dir (path): Expected in the following structure './result/{model_name}/latent_data', within folder consists of all the latent data file
    
    Returns:
        full_latent_data (dict): A dictionary in the form of {path: latent_data (z_channels, latent_size, latent_size)}
    """
    
    latent_dir = Path(latent_dir)
    latent_data_file_list = list(latent_dir.glob('*.pkl'))
    
    full_latent_data = {}
    for file in latent_data_file_list:

        with open(file, 'rb') as f:
            single_latent_data = pickle.load(f)
            # full_latent_data.update(single_latent_data)
            for k, v in single_latent_data.items():
                full_latent_data[k] = v
    
    return full_latent_data
            

