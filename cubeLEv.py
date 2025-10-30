from acoustools.Utilities import create_points, TRANSDUCERS, add_lev_sig
from acoustools.Solvers import wgs
from acoustools.Levitator import LevitatorController
from acoustools.Visualiser import Visualise, ABC

ps = create_points(8,1,x=[0.01, 0.01, 0.01, 0.01, -0.01, -0.01, -0.01, -0.01],
                       y=[0.01, 0.01, -0.01, -0.01, 0.01, 0.01, -0.01, -0.01],
                       z=[0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01]) 

board = TRANSDUCERS

x = wgs(ps, board = board)
x = add_lev_sig(x)

# Visualise(*ABC(0.04, origin=create_points(1,1, x=0, y=0.03, z=0)), x)

lev = LevitatorController(ids=(999,1000))

lev.levitate(x)

input()

lev.disconnect()