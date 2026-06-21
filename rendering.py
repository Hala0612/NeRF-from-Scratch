import numpy as np 

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_accumulated_transmitance(alphas):
    accumulated_transmitance = torch.cumprod(alphas, 1)
    return torch.cat([torch.ones_like(accumulated_transmitance[:, :1]),
                      accumulated_transmitance[:, :-1]], dim=1)


def rendering(model, rays_o, rays_d, tn, tf, nb_bins, white_bckrd):


    rays_o = rays_o.to(device)
    rays_d = rays_d.to(device)
    
    
    t = torch.linspace(tn, tf, nb_bins).expand(rays_o.shape[0], nb_bins)
    t = t.to(device)
    
    delta = t[:, 1:] - t[:, :-1] # delta a seulement 99 valeurs 
    #delta = torch.cat([delta, torch.zeros(rays_o.shape[0], 1) + 1e10], dim=1).to(device) # on ajoute 1 valeur à delta
    delta = torch.cat([delta, torch.zeros(rays_o.shape[0], 1, device=device) + 1e10], dim=1)

    delta = delta.to(device)
    # (N, 1, 3) => origine
    # (N, nb_bins, 1) => t 
    # (N, 1, 3) => direction
    x = rays_o.unsqueeze(1) + t.unsqueeze(2) * rays_d.unsqueeze(1) # [N, nb_bins, 3]

    density, color = model.intersect(x.reshape(-1, 3), rays_d.expand(x.shape[1], x.shape[0], 3).transpose(0, 1).reshape(-1, 3))

    color = color.reshape(x.shape) # [nb_rays, nb_bins, 3]
    density = density.reshape((x.shape[0], x.shape[1])) # [nb_rays, nb_bins]

    alpha = 1. - torch.exp(-density * delta) #[N, nb_bins]

    if white_bckrd :

        weight = compute_accumulated_transmitance(1 - alpha).unsqueeze(2)  * alpha.unsqueeze(2) # [nb_rays, nb_bins, 1]
        c = (weight * color).sum(dim = 1) # [N, 3]
        weight_sum = weight.sum(dim=-1).sum(dim=-1) # [nb_rays]
        return c  + 1 - weight_sum.unsqueeze(-1)


    else: 
        T = compute_accumulated_transmitance(1 - alpha) #[N, nb_bins]

        img = (T.unsqueeze(2) * alpha.unsqueeze(2) * color).sum(dim = 1) # [N, 3]

        return img