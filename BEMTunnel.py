from acoustools.Utilities import TOP_BOARD, create_points
from acoustools.Mesh import load_scatterer
from acoustools.BEM import compute_E, propagate_BEM_pressure
from acoustools.Solvers import iterative_backpropagation
from acoustools.Visualiser import Visualise, ABC


board = TOP_BOARD

p = create_points(1,x=0,y=0,z=0.01)

scatterer = load_scatterer('./data/Tunnel-lam4.stl', rotz=90, dz=-0.025)

E,F,G,H = compute_E(scatterer, p, board, path='./data/',return_components=True)

x = iterative_backpropagation(p, board=board, A=E)
x_p, = iterative_backpropagation(p, board=board)

r = 500
Visualise(*ABC(0.1), [x_p,x], points=p, colour_functions=[propagate_BEM_pressure,propagate_BEM_pressure], res=(r,r), vmax=5000, arangement=(2,1),
                        colour_function_args=[{'scatterer':scatterer,'board':board, 'H':H, 'path':'./data/'},
                                              {'scatterer':scatterer,'board':board, 'H':H, 'path':'./data/'}])