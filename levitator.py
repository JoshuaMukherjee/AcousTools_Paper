from acoustools.Levitator import LevitatorController
from acoustools.Utilities import create_points
from acoustools.Solvers import gspat

lev = LevitatorController()

p = create_points(1,x=0,y=0,z=0)
x = gspat(p)

lev.levitate(x)
input() #Wait for user input before disconnecting
lev.disconnect()


