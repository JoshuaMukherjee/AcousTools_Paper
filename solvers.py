from acoustools.Utilities import TRANSDUCERS, create_points, propagate_abs
from acoustools.Solvers import naive, iterative_backpropagation, gspat, wgs

from acoustools.Visualiser import Visualise, ABC

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 50, 'font.family' : 'times',})

board = TRANSDUCERS

N = 8
p = create_points(N, y=0)

x_niave = naive(p, board=board)
p_naive = propagate_abs(x_niave, p, board=board)

x_ib = iterative_backpropagation(p, board=board)
p_ib = propagate_abs(x_ib, p, board=board)

x_gspat = gspat(p, board=board)
p_gspat = propagate_abs(x_gspat, p, board=board)

x_wgs = wgs(p, board=board)
p_wgs = propagate_abs(x_wgs, p, board=board)

r = 500
# Visualise(*ABC(0.07), [x_niave, x_ib, x_gspat, x_wgs], colour_functions=[propagate_abs,propagate_abs,propagate_abs,propagate_abs], res=(r,r))


pos = 0
plt.bar([pos+0.7/N*i for i in range(N)], p_naive.cpu().detach().numpy().squeeze(), width=0.075)
pos += 0.75
plt.bar([pos+0.7/N*i for i in range(N)], p_ib.cpu().detach().numpy().squeeze(), width=0.075)
pos += 0.75
plt.bar([pos+0.7/N*i for i in range(N)], p_gspat.cpu().detach().numpy().squeeze(), width=0.075)
pos += 0.75
plt.bar([pos+0.7/N*i for i in range(N)], p_wgs.cpu().detach().numpy().squeeze(), width=0.075)


plt.ylabel('Point Pressure (Pa)')
plt.xticks([0.305 + 0.75 * i for i in range(4)], labels=['Naive', 'IB', 'GS-PAT', 'WGS'])
plt.show()