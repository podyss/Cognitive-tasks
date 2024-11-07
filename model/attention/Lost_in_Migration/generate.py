from random import randint, choice
import numpy as np
# import matplotlib.pyplot as plt

relative_pos = [[[-2,-2],[-1,-1],[0,0],[1,-1],[2,-2]],[[-2,0],[-1,0],[0,0],[1,0],[2,0]]]

def generate() -> tuple[list, int]:
    data = []
    label = 0
    idx = randint(0,len(relative_pos)-1)
    pos = [randint(-10,10), randint(-10,10)]
    x = choice([-1,1])
    y = choice([-1,1])
    for i in range(5):
        now = [pos[0]+x*relative_pos[idx][i][0], pos[1]+y*relative_pos[idx][i][1], randint(0,3)]
        data.append(now)
        # plt.plot(now[0],now[1],'ro')
    # plt.show()
    label = data[2][2]
    return data, label
    