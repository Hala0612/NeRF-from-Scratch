
import torch
import torch.nn as nn
import torch.nn.functional as F
# Voxels
class Voxels(nn.Module): 

    def __init__(self, scale, nb_voxels, device='cuda'):
        super(Voxels, self).__init__()
        
        self.scale = scale 
        self.nb_voxels = nb_voxels
        self.device = device 

        # l'optimisateur va optimiser ces paramètres
        self.voxels = torch.nn.Parameter(
            torch.rand((nb_voxels, nb_voxels, nb_voxels, 4),
                       device=device,
                       requires_grad=True))

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, positions, d=None):

        print(positions[:, 0].min(), positions[:, 0].max(), positions[:, 0].mean())
        print(positions[:, 1].min(), positions[:, 1].max(), positions[:, 1].mean())
        print(positions[:, 2].min(), positions[:, 2].max(), positions[:, 2].mean())

        mask = (positions[:, 0].abs() < (self.scale / 2)) & \
               (positions[:, 1].abs() < (self.scale / 2)) & \
               (positions[:, 2].abs() < (self.scale / 2))
        



        colors = torch.zeros_like(positions).type(self.voxels.dtype)
        density = torch.zeros_like(positions[:, 0]).type(self.voxels.dtype)

        x = (positions[mask, 0] / (self.scale / self.nb_voxels) + self.nb_voxels /2).type(torch.long)
        y = (positions[mask, 1] / (self.scale / self.nb_voxels) + self.nb_voxels /2).type(torch.long)
        z = (positions[mask, 2] / (self.scale / self.nb_voxels) + self.nb_voxels /2).type(torch.long)


        density[mask] = self.voxels[x, y, z, -1]
        colors[mask] = self.voxels[x, y, z, :3]


        return self.relu(density), self.sigmoid(colors)

    def intersect(self, positions, d=None):
        density, color = self.forward(positions)
        return density, color

# ------------------------------------------------------------------------------------------------------------------------------

# Nerf
class Nerf(nn.Module): 
    # positional encoding
    def __init__(self, Lpos = 10, Ldir=4, hidden_dim=265, device='cuda'):
        super(Nerf, self).__init__()
        self.block1 = nn.Sequential(nn.Linear(Lpos * 6 + 3, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        

        self.block2 = nn.Sequential(nn.Linear(hidden_dim + Lpos * 6 + 3, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim + 1)) # +1 pour prédire sigma
        
        self.rgb_head = nn.Sequential(nn.Linear(hidden_dim + Ldir * 6 + 3, hidden_dim //2), nn.ReLU(),
                                      nn.Linear(hidden_dim //2, 3), nn.Sigmoid())
        
        self.Lpos = Lpos
        self.Ldir = Ldir

    def positional_encoding(self, x, L):
        out = torch.empty((x.shape[0], x.shape[1] * 2 * L + x.shape[1]), device=x.device)
        for i in range(x.shape[1]):
            for j in range(L):
                out [:, i * (2*L) + 2 * j] = torch.sin(2 ** j * x[:, i])
                out [:, i * (2*L) + 2 * j + 1] = torch.cos(2 ** j * x[:, i])
        out[:, - x.shape[1]:] = x

        return out



    def forward(self, positions, d):
        x_emb = self.positional_encoding(positions, self.Lpos)
        x_dir = self.positional_encoding(d, self.Ldir)


        h = self.block1(x_emb)
        h = self.block2(torch.cat((x_emb, h), dim=1))
        h, density = h[:, :-1], h[:, -1]
        c = self.rgb_head(torch.cat((x_dir, h), dim=1))

        return F.softplus(density), c

    def intersect(self, positions, d):
        return self.forward(positions, d)