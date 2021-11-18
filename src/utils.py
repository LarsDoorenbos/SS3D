import torch

@torch.no_grad()
def get_single_spectrum(images, network):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    images = images.to(device)
    
    features = network.module.extract_features(images).mean(dim=(2, 3))

    ret = features.cpu().numpy()

    return ret        
