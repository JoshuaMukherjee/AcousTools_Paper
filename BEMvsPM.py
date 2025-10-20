from acoustools.Mesh import load_scatterer
from acoustools.BEM import compute_E, propagate_BEM_pressure
from acoustools.Solvers import iterative_backpropagation
from acoustools.Utilities import create_points, TOP_BOARD

from acoustools.Visualiser import Visualise, ABC


path='data/'
reflector = load_scatterer('data/flat-lam4.stl')

p = create_points(1,x=0,y=0,z=0.03)
board = TOP_BOARD

x_pm = iterative_backpropagation(points=p, board=board)

E = compute_E(reflector, points=p, board=board, path=path)
x_BEM = iterative_backpropagation(points=p, board=board, A=E)

r = 500
Visualise(*ABC(0.06, origin = p), [x_pm, x_BEM] ,res=(r,r), 
          colour_functions=[propagate_BEM_pressure, propagate_BEM_pressure], 
          colour_function_args=[{'scatterer':reflector, 'board':board, 'path':path},
                                {'scatterer':reflector, 'board':board, 'path':path}],
                                arangement=(2,1))