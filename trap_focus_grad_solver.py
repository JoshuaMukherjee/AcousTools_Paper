from acoustools.Utilities import create_points, TRANSDUCERS, propagate_abs
from acoustools.Solvers import gradient_descent_solver
from acoustools.Optimise.Objectives import propagate_abs_sum_objective, gorkov_analytical_sum_objective
from acoustools.Gorkov import gorkov

import torch
torch.random.manual_seed(5)

from acoustools.Visualiser import Visualise, ABC

board = TRANSDUCERS

def objective(transducer_phases, points, board, targets = None, **objective_params):

    alpha = objective_params['alpha']

    p1 = points[:,:,0].unsqueeze(2)
    p2 = points[:,:,1].unsqueeze(2)

    pressure = propagate_abs(transducer_phases, p1, board=board).squeeze()
    U = gorkov(transducer_phases, p2, board=board ).squeeze()

    return (-1*pressure + alpha * U).unsqueeze(0)


p = create_points(2, y=0, max_pos=0.04, min_pos=-0.04)

x = gradient_descent_solver(p, objective=objective, board=board, log=False, objective_params={'alpha':1.32e10}, iters=1000)


r = 500
Visualise(*ABC(0.05), x, points=p, colour_functions=[propagate_abs, gorkov], res = (r,r),
          link_ax=None, arangement=(2,1), clr_labels=["Pressure (Pa)", "Gor'kov Potential (J)"])
