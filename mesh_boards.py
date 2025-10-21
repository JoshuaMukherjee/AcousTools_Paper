from acoustools.Utilities import create_points, propagate_abs, BOARD_POSITIONS
from acoustools.Solvers import gspat
from acoustools.Mesh import mesh_to_board

from acoustools.Visualiser import Visualise, ABC

p = create_points(1,x=0,y=0,z=0)

flat_board, flat_norms = mesh_to_board('./data/flat-lam1.stl', flip_normals=False, dz=-1 * BOARD_POSITIONS, centre=False)
x_flat = gspat(p, board=flat_board, norms=flat_norms)

sphere_board, sphere_norms = mesh_to_board('./data/Sphere-lam1.stl')
x_sphere = gspat(p, board=sphere_board, norms=sphere_norms)

cube_board, cube_norms = mesh_to_board('./data/Cube-lam4.stl')
x_cube = gspat(p, board=cube_board, norms=cube_norms)

teapot_board, teapot_norms = mesh_to_board('./data/Teapot_smooth.stl')
x_teapot = gspat(p, board=teapot_board, norms=teapot_norms)


r = 200
Visualise(*ABC(0.1),[x_flat,x_sphere, x_cube, x_teapot] ,res=(r,r), vmax=8000, arangement=(2,2),
          colour_functions=[propagate_abs,propagate_abs,propagate_abs, propagate_abs], 
          colour_function_args=[{'board':flat_board, 'norms':flat_norms},
                                {'board':sphere_board, 'norms':sphere_norms},
                                {'board':cube_board, 'norms':cube_norms},
                                {'board':teapot_board, 'norms':teapot_norms}])