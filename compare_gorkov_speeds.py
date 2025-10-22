from acoustools.Utilities import create_points, add_lev_sig
from acoustools.Solvers import wgs
from acoustools.Gorkov import gorkov_analytical, gorkov_fin_diff

import time, torch
N=1
M = 3000 

points= []
xs = []
for _ in range(M):
    p = create_points(N)
    points.append(p)

    x = wgs(p)
    x = add_lev_sig(x)
    xs.append(x)

# U_ag = gorkov_autograd(x,points)

ts_analytic = []
ts_finite_differences = []
ts_autograd = []


for p,x in zip(points, xs):
    t1 = time.time_ns()
    U_fd = gorkov_fin_diff(x,p)
    t2 = time.time_ns()
    U_a = gorkov_analytical(x,p)
    t3 = time.time_ns()

    ts_finite_differences.append((t2-t1)/1e9)
    ts_analytic.append((t3-t2)/1e9)

    torch.clear_autocast_cache()

print(sum(ts_analytic))
print(sum(ts_finite_differences))

fps_analytics = M / sum(ts_analytic)
fps_fd = M / sum(ts_finite_differences)

print(fps_analytics, fps_fd)

# import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 50, 'font.family' : 'times',})

# plt.bar(0, fps_analytics)
# plt.bar(1, fps_fd)

# plt.ylabel("Solutions Per Second")

# plt.xticks([0, 1], ["Analytical","Finite Differences"])

# plt.show()