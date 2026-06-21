

import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import imageio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# train data 

# Generate rays (ray_origin, ray_direction) and associates them with pixel colors from input images

def get_rays(mode = 'train') :

    # Load file names
    root = './Fox_Dataset/fox'

    # import poses =>  camera-to-world transformation matrices
    pose_file_names = os.listdir(root + f'/{mode}/pose')

    # import intrinsics => camera projection matrices.
    intrinsics_file_names = os.listdir(root + f'/{mode}/intrinsics')

    # Check that there are as many pose files as intrinsic files
    assert len(pose_file_names) == len(intrinsics_file_names)


    # import images
    img_file_names = [ e for e in os.listdir(root + '/imgs') if mode in e]
    
    # Check that there are as many pose files as image files
    assert len(pose_file_names) == len(img_file_names)


    # initialize arrays 
    N = len(pose_file_names)
    poses = np.zeros((N, 4, 4))
    intrinsics = np.zeros((N, 4, 4))

    images = []

    # Load text files and images 
    for i in range(N): 

        name = pose_file_names[i]
        pose = open(root + f'/{mode}/pose/' + name).read().split('\n') # read txt pose file
        poses[i] = np.array([float(x) for x in pose]).reshape(4, 4) # convert to a 4x4 float arrays

        tmp = open(root + f'/{mode}/intrinsics/' + name).read().split('\n') # read txt intrinsic file
        intrinsics[i] = np.array([float(x) for x in tmp]).reshape(4, 4)  # convert to a 4x4 float arrays


        # Read images, normalize [0,1] and add batch dimension for stacking later 
        img = imageio.imread(root + '/imgs/' + name.replace('txt', 'png')) / 255.
        # on ajoute 1 dimension en plus pour ensuite concatener toutes les images par rapport à cette dimension, ça nous donnera le nombre d'images
        images.append(img[None, ...]) 

    # combine list of imgs into a single array of shape (N, H, W, 3 or 4)
    images = np.concatenate(images)    

    # Handle alpha channel
    if images.shape[-1] == 4 :
        images = images[:, :, :, :3] * images[:, :, :, -1:] + (1 - images[:, :, :, -1:]) 


    H, W = images.shape[1], images.shape[2]

    # generate rays
    rays_o = np.zeros((N, H * W, 3)) # camera positions
    rays_d = np.zeros((N, H * W, 3))
    rays_target_color = np.zeros((N, H * W, 3)) # RGB color per ray

    # ray generation per image
    for i  in range(N):

        focal_length = intrinsics[i][0, 0] # extract focal length
        c2w = poses[i] # camera-to-world 

        # create pixel grid
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        u = u.reshape(-1).astype(np.float32)
        v = v.reshape(-1).astype(np.float32)

        # compute ray directions
        ray_d = np.stack((u - W / 2.,  -(v - H / 2.), - np.ones_like(u) * focal_length),
                        axis=-1)

        # Rotate ray directions to world coordinates using the camera's rotation matrix.
        ray_d = (c2w[:3, :3] @ ray_d[..., None]).squeeze(-1)

        rays_d[i] = ray_d / np.linalg.norm(ray_d, axis=-1, keepdims=True) # normalize ray directions to unit vectors

        # rays start from camera position
        rays_o[i] = np.tile(c2w[:3, 3], (ray_d.shape[0], 1)) # (H*W, 3)

        rays_target_color[i] = images[i].reshape(-1, 3) # associate color pixel to its ray
    
    return rays_o, rays_d, rays_target_color, rays_o.reshape(N, H, W, 3)[:, int(H/4):int(3*H/4), int(W/4):int(3*W/4), :].reshape(N, -1, 3), rays_d.reshape(N, H, W, 3)[:, int(H/4):int(3*H/4), int(W/4):int(3*W/4), :].reshape(N, -1, 3), rays_target_color.reshape(N, H, W, 3)[:, int(H/4):int(3*H/4), int(W/4):int(3*W/4), :].reshape(N, -1, 3)




# Custom ray generator used in NeRF for rendering novel views of a scene 
# from virtual camera positions defined by spherical coordinates (theta, phi, r).
"""
def generate_rays(theta, phi, r, focal_length=1333, H=400, W=400) : 

    c2w = np.eye(4)

    # move camera back in z axis
    t = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, r],
                  [0, 0, 0, 1]])
    c2w =  t @ c2w

    # apply elevation (vertical angle) and rotation around X axis
    rot_phi = np.array([[1, 0, 0, 0],
                        [0, np.cos(phi), -np.sin(phi), 0],
                        [0, np.sin(phi), np.cos(phi), 0],
                        [0, 0, 0, 1]])
    
    c2w = rot_phi @  c2w 

    # apply azimuth (horizontal angle) and rotation around Y axis
    rot_theta = np.array([[np.cos(theta), 0, -np.sin(theta), 0],
                         [0, 1, 0, 0],
                         [np.sin(theta), 0, np.cos(theta), 0],
                         [0, 0, 0, 1]])
    c2w = rot_theta @  c2w 

    # Coordinate system conversion : invert Y and Z (from Blender/OpenGL to Nerf convention)
    c2w = np.array([[-1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]]) @ c2w
    
    # Generate pixel grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.reshape(-1).astype(np.float32)
    v = v.reshape(-1).astype(np.float32)

    # Compute ray directions in camera space
    ray_d = np.stack((u - W / 2.,  -(v - H / 2.), - np.ones_like(u) * focal_length),
                    axis=-1)

    # Transform ray directions to world space
    ray_d = (c2w[:3, :3] @ ray_d[..., None]).squeeze(-1)

    # Normalize directions and set ray origins
    rays_d = ray_d / np.linalg.norm(ray_d, axis=-1, keepdims=True)
    rays_o = np.tile(c2w[:3, 3], (ray_d.shape[0], 1)) # (H*W, 3)

    return rays_o, rays_d"""

def generate_rays(theta, phi, r, focal_length=1333, H=400, W=400): 
    # Matrice de base
    c2w = np.eye(4)

    # Caméra en (0, 0, r)
    t = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, r],
                  [0, 0, 0, 1]])
    c2w = t @ c2w

    # Inclinaison (vue de dessus)
    rot_phi = np.array([[1, 0, 0, 0],
                        [0, np.cos(phi), -np.sin(phi), 0],
                        [0, np.sin(phi), np.cos(phi), 0],
                        [0, 0, 0, 1]])
    c2w = rot_phi @ c2w 

    # Rotation horizontale
    rot_theta = np.array([[np.cos(theta), 0, -np.sin(theta), 0],
                          [0, 1, 0, 0],
                          [np.sin(theta), 0, np.cos(theta), 0],
                          [0, 0, 0, 1]])
    c2w = rot_theta @ c2w 

    # Correction de convention (Blender -> NeRF)
    c2w = np.array([[-1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]]) @ c2w

    # Grille de pixels
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.reshape(-1).astype(np.float32)
    v = v.reshape(-1).astype(np.float32)

    # Rayons en espace caméra
    ray_d = np.stack([(u - W/2), -(v - H/2), -np.ones_like(u) * focal_length], axis=-1)

    # Transfert vers espace monde
    ray_d = (c2w[:3, :3] @ ray_d[..., None]).squeeze(-1)
    rays_d = ray_d / np.linalg.norm(ray_d, axis=-1, keepdims=True)

    # Origine = position caméra
    rays_o = np.tile(c2w[:3, 3], (ray_d.shape[0], 1))

    return rays_o, rays_d


