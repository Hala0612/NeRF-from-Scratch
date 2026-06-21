import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from tqdm import tqdm
from rendering import rendering



def train(model, optimizer, scheduler, dataloader,nb_epochs, tn, tf, white_bckrd, device = 'cuda', nb_bins=192):

    
    training_loss = []
    for epoch in tqdm(range(nb_epochs)):

        for batch in dataloader:

            batch = batch.to(device)
            rays_origins, rays_directions, target_px_values = batch[:, :3], batch[:, 3:6], batch[:, -3:]


            regenerated_px_values = rendering(model, rays_origins, rays_directions, tn, tf, nb_bins, white_bckrd)


            loss = ((regenerated_px_values - target_px_values )**2).mean()

            training_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('it done')

        scheduler.step()
        
    return training_loss

def mse2psnr(mse):
    psnr = 20 * np.log10(1 / np.sqrt(mse))
    return psnr

@ torch.no_grad()
def test(model, o, d, target, tn, tf, nb_bins, white_bckrd, H, W, chunk_size=10, device='cpu'):

    # diviser en plusieurs batch si on a plusieurs voxels, pour eviter le 'out of memory'
    o = o.chunk(chunk_size)
    d = d.chunk(chunk_size)

    image = []
    for o_batch, d_batch in zip(o, d):

        image.append(rendering(model, o_batch.to(device), d_batch.to(device), tn, tf, nb_bins, white_bckrd).data.cpu().numpy())

    image= np.concatenate(image)
    



    metrics = {}

    if target is not None : 
        metrics['mse'] = ((image - target)**2).mean()

        metrics['psnr']= mse2psnr(metrics['mse'])


    return image.reshape((H, W, 3)), metrics 