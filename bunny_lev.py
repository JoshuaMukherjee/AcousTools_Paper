from acoustools.Mesh import load_scatterer
from acoustools.BEM import compute_E, propagate_BEM_pressure
from acoustools.BEM.Force import BEM_compute_force
from acoustools.Utilities import create_points, TOP_BOARD, add_lev_sig
from acoustools.Solvers import wgs

from acoustools.Visualiser import ABC, Visualise
from acoustools.Levitator import LevitatorController

import torch

torch.manual_seed(1)

bunny = load_scatterer('data/bunny-lam2.stl', rotz=90)
path = './data'

p = create_points(1,1,x=0,y=0, z=0.04)
board = TOP_BOARD

E,F,G,H = compute_E(bunny, points=p, board=board, path=path, return_components=True)

x = wgs(p, A=E, board=board)
x = add_lev_sig(x, board=board, mode='Twin')


# r = 200
# Visualise(*ABC(0.1, origin=create_points(1,1,0,0,0.04)), x, res=(r,r), 
#           colour_functions=[propagate_BEM_pressure], 
#           colour_function_args=[{'board':board,'path':path,'scatterer':bunny} ], vmax = 5000)

lev = LevitatorController(ids=(999,))

lev.levitate(x)

input()

lev.disconnect()