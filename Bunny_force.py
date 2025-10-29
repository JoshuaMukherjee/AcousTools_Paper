from acoustools.Mesh import load_scatterer
from acoustools.BEM import compute_E, propagate_BEM_pressure
from acoustools.BEM.Force import BEM_compute_force
from acoustools.Utilities import create_points, TOP_BOARD, TRANSDUCERS
from acoustools.Solvers import wgs

from acoustools.Visualiser import ABC, Visualise_single, get_image_positions, get_point_pos

import torch

torch.manual_seed(1)

bunny = load_scatterer('data/bunny-lam2.stl', rotz=90)
path = './data'

p = create_points(2,1,y=0, min_pos=0.02, max_pos=0.04)
board = TOP_BOARD

E,F,G,H = compute_E(bunny, points=p, board=board, path=path, return_components=True)

x = wgs(p, board=board, A=E)

r = 500
size = 0.05
abc = ABC(size)

im = Visualise_single(*abc, x, res=(r,r),
          colour_function=propagate_BEM_pressure, 
          colour_function_args={"scatterer":bunny, "H":H,"path":path,"board":board, 'smooth_distance':7e-4})


scale = 10
pt = (p[:,:,0] + p[:,:,1]) / 2
pt = pt.unsqueeze(2)
img_pts = get_image_positions(*ABC(0.015, origin=pt), res=(int(r/scale), int(r/scale)))

img_pts[:,0] += ((size/scale)/(2*r)) 
img_pts[:,2] -= ((size/scale)/(2*r)) 
 

force= BEM_compute_force(x, img_pts, board=board, H=H, path=path, scatterer=bunny)

force_x = force[0,:].detach().numpy()
force_z = force[2,:].detach().numpy()



img_pos = get_point_pos(*abc, img_pts, res=(r,r))

img_pos = torch.stack(img_pos).T

img_x = img_pos[1].cpu().numpy()
img_z = img_pos[0].cpu().numpy()


import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 40, 'font.family' : 'times',})


# plt.subplot(2,1,1)
plt.matshow(im, cmap='hot', vmax=5000)

plt.xticks([])
plt.yticks([])
plt.colorbar(label="Pressure (Pa)")


plt.quiver(img_x, img_z, force_x, force_z, scale=2e-2 , width=0.001, color='blue', label = 'Force')


plt.show()