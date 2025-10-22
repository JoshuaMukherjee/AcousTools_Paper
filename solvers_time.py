from acoustools.Utilities import create_points, BOTTOM_BOARD, forward_model
from acoustools.Solvers import naive, iterative_backpropagation, gspat, wgs

import time, torch


points = []
Fs = []
M = 10000
for i in range(M):
    p = create_points(1)
    F = forward_model(p, transducers=  BOTTOM_BOARD)
    points.append(p)
    Fs.append(F)


solvers = [naive, iterative_backpropagation, gspat, wgs]

fps = {}
fps_F = {}
with torch.no_grad():

    for solver in solvers:
        start = time.time_ns()
        for p in points:
            x = solver(points=p, board=BOTTOM_BOARD, iterations=10)
        
        end = time.time_ns()
        fps[solver] =  M / ((end - start) / 1e9)
        print(fps[solver])

        start = time.time_ns()
        for F in Fs:
            x = solver(points=p, board=BOTTOM_BOARD, iterations=10, A=F)
        
        end = time.time_ns()
        fps_F[solver] =  M / ((end - start) / 1e9)
        print(fps_F[solver])


import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 50, 'font.family' : 'times',})


colours = {naive:"tab:blue", iterative_backpropagation:"tab:orange", gspat:"tab:green", wgs:"tab:red"}

pos = 0
for solver in solvers:
    t = fps[solver]
    fF = fps_F[solver]
    plt.bar(pos, t, width=0.4, color = colours[solver])
    plt.bar(pos+0.45, fF, width=0.4, color = colours[solver])
    pos += 1

plt.yscale('log')
plt.yticks([1e3, 2e3, 3e3, 4e4 , 5e3, 1e4, 2e4, 3e4, 4e4, 5e4, 7e4, 10e4], [1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 50000, 70000, 100000])
plt.ylabel("Frames Per Second")

# plt.xticks([0.22, 1.22, 2.22, 3.22], ["Naive", "IB", "GS-PAT", "WGS"])
plt.xticks([0, 0.45, 1, 1.45, 2, 2.45, 3, 3.45], ["Naive", "Naive \n Pre-computed F", "IB", "IB \n Pre-computed F", "GS-PAT", "GS-PAT\n Pre-computed F", "WGS", "WGS \n Pre-computed F"])

plt.show()