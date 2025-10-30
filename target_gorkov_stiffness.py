from acoustools.Utilities import create_points, TOP_BOARD, propagate_abs, device, DTYPE
from acoustools.Solvers import gradient_descent_solver
from acoustools.Optimise.Objectives import target_gorkov_BEM_mse_sine_objective
from acoustools.Optimise.Constraints import sine_amplitude, constrain_phase_only
from acoustools.Gorkov import gorkov
from acoustools.BEM import propagate_BEM_pressure, compute_E, stiffness_finite_differences_BEM, BEM_gorkov_analytical
from acoustools.Mesh import load_scatterer
from acoustools.Export.Holo import save_holograms

import torch
torch.random.manual_seed(5)

from acoustools.Visualiser import Visualise, ABC

board = TOP_BOARD

root = "./data/" 
path = root+"flat-lam4.stl"

torch.autograd.set_detect_anomaly(True)

reflector = load_scatterer(path) #Change dz to be the position of the reflector


def objective(transducer_phases, points, board, targets = None, **objective_params):

    alpha = objective_params['alpha']
    reflector = objective_params['reflector']
    root = objective_params['root']

    E1, E2 = objective_params['Es']


    p1 = points[:,:,0].unsqueeze(2)
    p2 = points[:,:,1].unsqueeze(2)

    gorkov_obj = target_gorkov_BEM_mse_sine_objective(transducer_phases, p1, board, targets=targets, reflector=reflector, root=root, E=E1).squeeze()
    # presure_obj = propagate_BEM_pressure(transducer_phases, p2, scatterer=reflector, path=root, E=E2).squeeze()
    # stiffness_obj = stiffness_finite_differences_BEM(transducer_phases, p2, board, scatterer=reflector, H=H)
    gorkov_max = BEM_gorkov_analytical(transducer_phases, p2, reflector, board, E=E2,path=root).squeeze()

    print(gorkov_obj.item(), gorkov_max.item())

    return (gorkov_max + alpha * gorkov_obj).unsqueeze(0)


p = create_points(2, x=[-0.02, 0.02], y=0, z=0.02)
p1 = p[:,:,0].unsqueeze(2)
p2 = p[:,:,1].unsqueeze(2)

U_target = torch.tensor([-1e-7,]).to(device).to(DTYPE)

E1, F1, G1, H = compute_E(reflector, p1, board, path=root, return_components=True)
E2, F2, G2, H = compute_E(reflector, p2, board, path=root, return_components=True, H=H)


x = gradient_descent_solver(p, objective=objective, board=board, targets=U_target, log=True, constrains=constrain_phase_only, objective_params={'alpha':1e8,"root":root, 'Es':[E1, E2], "reflector":reflector}, iters=300, lr=1e5)

print(BEM_gorkov_analytical(x, p1, reflector, board=board, path=root))
print(BEM_gorkov_analytical(x, p2, reflector, board=board, path=root))

print(stiffness_finite_differences_BEM(x, p2, board, scatterer=reflector, H=H, path=root))

save_holograms([x,], './data/outputs/DropletLev.holo')

r = 200
Visualise(*ABC(0.1, origin=create_points(1,1,0,0,0.04)), x,points=p, res=(r,r), colour_functions=[propagate_BEM_pressure], colour_function_args=[{'board':board,'path':root,'scatterer':reflector,"H":H} ], vmax = 5000)