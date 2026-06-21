import numpy as np
import torch
from skimage import measure
import mcubes, trimesh 


def get_mesh(model, N=100, scale = 1.5, chuck_size = 10, threshold = 1, device='cuda'):

    density = np.zeros((N, N, N), dtype=np.float32)
    

    x = np.linspace(-scale, scale, N).astype(np.float32)
    y = np.linspace(-scale, scale, N).astype(np.float32)
    z = np.linspace(-scale, scale, N).astype(np.float32)

    x, y, z = np.meshgrid(x, y, z)

    xyz = np.concatenate((x.reshape(-1, 1),
                        y.reshape(-1, 1),
                        z.reshape(-1, 1)), axis = 1)
    
    
    points_per_chunk = 1000000  # par exemple 1 million de points par chunk
    chuck_size = int(np.ceil(xyz.shape[0] / points_per_chunk))

    xyz = np.array_split(xyz, chuck_size)
    density_list = []

    for xyz_batch in xyz:
        o = torch.from_numpy(xyz_batch).to(device)
        density_list.append(model(o, torch.zeros_like(o))[0].data.cpu().numpy())

    density[:]= np.concatenate(density_list).reshape(density.shape)

    vertices, triangles = mcubes.marching_cubes(density, threshold * np.mean(density))
    mesh = trimesh.Trimesh(vertices / 100, triangles)  

  

    return mesh

'''

def colorize_mesh(model, mesh, device='cuda'):
    verts = torch.from_numpy(mesh.vertices).float().to(device)
    dirs = torch.zeros_like(verts).to(device)  # Directions dummy, car couleur indépendante ici.

    _, rgb = model(verts, dirs)
    colors = rgb.cpu().numpy()  # valeurs entre 0 et 1

    # Ajouter les couleurs (en 0-255 si nécessaire)
    colors = (colors * 255).astype(np.uint8)
    mesh.visual.vertex_colors = colors

    return mesh 


@torch.no_grad()
def get_mesh(model, N=256, scale=1.0, points_per_chunk=500000, threshold=20, device='cuda'):
    model.to(device)
    model.eval()

    # Générer la grille 3D de coordonnées normalisées
    x = np.linspace(-scale, scale, N)
    y = np.linspace(-scale, scale, N)
    z = np.linspace(-scale, scale, N)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
    xyz = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)

    print(f"[INFO] Total points to evaluate: {xyz.shape[0]}")

    densities = []
    n_chunks = int(np.ceil(xyz.shape[0] / points_per_chunk))
    print(f"[INFO] Splitting into {n_chunks} chunks of ~{points_per_chunk} points each")

    for i in range(n_chunks):
        xyz_chunk = xyz[i * points_per_chunk : (i + 1) * points_per_chunk]
        xyz_tensor = torch.from_numpy(xyz_chunk).float().to(device)
        dirs = torch.zeros_like(xyz_tensor).to(device)  # direction dummy = zeros

        density, _ = model(xyz_tensor, dirs)
        densities.append(density.cpu().numpy())

    densities = np.concatenate(densities, axis=0)
    densities = densities.reshape(N, N, N)

    print("[INFO] Running marching cubes...")

    verts, faces, normals, _ = measure.marching_cubes(densities, level=threshold, spacing=(2*scale/N, 2*scale/N, 2*scale/N))
    
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    return mesh, densities
'''

