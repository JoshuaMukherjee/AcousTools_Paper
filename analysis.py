from acoustools.Utilities import create_points, \
    TRANSDUCERS, propagate_abs
from acoustools.Solvers import wgs

from acoustools.Gorkov import gorkov
from acoustools.Force import compute_force
from acoustools.Stiffness import stiffness_finite_differences

from acoustools.Visualiser import ABC, Visualise

p = create_points(1,x=0,y=0,z=0)

board = TRANSDUCERS

x = wgs(p + create_points(1,x=0,y=0.0001,z=0.00),
         board=board)

def force_x(activations, points , board=board):
    Fx,Fy,Fz = compute_force(activations,points,board, 
                             return_components=True)
    return Fx

def force_y(activations, points , board=board):
    Fx,Fy,Fz = compute_force(activations,points,board, 
                             return_components=True)
    return Fy

def force_z(activations, points , board=board):
    Fx,Fy,Fz = compute_force(activations,points,board, 
                             return_components=True)
    return Fz


r = 500

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 40, 'font.family' : 'times',})

Visualise(*ABC(0.01, origin=p), x,
          colour_functions=[propagate_abs, 
                            gorkov, 
                            force_x, force_y, force_z,
                            stiffness_finite_differences], 
          arangement=(3,2), res=(r,r), link_ax=None,
          clr_labels=["Pressure (Pa)", 
                        "Gor'kov Potential (J)", 
                        "Fx (N)", "Fy (N)", "Fz (N)", 
                        "Stiffness (N/m)" ]
                        )

