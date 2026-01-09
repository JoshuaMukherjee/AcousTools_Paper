from acoustools.Utilities import create_points, propagate_abs_normalised, propagate_abs
from acoustools.Solvers import wgs 
from acoustools.Constants import c_0, pi, k

from acoustools.Visualiser import Visualise, ABC


fs_kHz = [40, 160]
fs = [f * 1000 for f in fs_kHz]

ks = [(2 * pi * f)/c_0 for f in fs]

p = create_points(1,1,0.05,0.03,0.05)


xs = []
for k in ks:
    x = wgs(p, k=k)
    xs.append(x)


import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'font.family' : 'times',})


Visualise(*ABC(0.01, origin=p), xs, res=(400,400), link_ax=None, depth=-1,
          colour_functions=[propagate_abs,propagate_abs],
        colour_function_args=[{"k":k} for k in ks],
        clr_labels=["Normalised\nPressure", "Normalised'\nPressure"],
        titles=['40kHz', '160kHz'])
