from acoustools.Export.Holo import load_holograms
from acoustools.Levitator import LevitatorController

from acoustools.Utilities import create_points,TOP_BOARD
from acoustools.BEM import propagate_BEM_pressure
from acoustools.Visualiser import Visualise, ABC
from acoustools.Mesh import load_scatterer

board = TOP_BOARD

root = "./data/" 
path = root+"flat-lam4.stl"


reflector = load_scatterer(path) #Change dz to be the position of the reflector

import torch


x = load_holograms('./data/outputs/DropletLev.holo')[0]

Vis=False

if Vis:

    r = 200
    Visualise(*ABC(0.1, origin=create_points(1,1,0,0,0.04)), x, res=(r,r), colour_functions=[propagate_BEM_pressure], colour_function_args=[{'board':board,'path':root,'scatterer':reflector} ], vmax = 5000)

else:

    print(torch.angle(x), torch.abs(x))
    lev = LevitatorController(ids=(999,))

    lev.levitate(x)

    input()

    lev.disconnect()