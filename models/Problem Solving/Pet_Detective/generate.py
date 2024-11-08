from random import randint, choice
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

walk = [[0,1],[0,-1],[-1,0],[1,0]]
mp = set()
cnt = []
def init(n):
    """
    初始化
    """
    global mp, cnt
    mp = set()
    cnt = np.zeros((n,n),dtype=np.int32)
    cnt.fill(4)
    for i in range(n):
        cnt[0][i] -= 1
        cnt[n-1][i] -= 1
    for i in range(n):
        cnt[i][0] -= 1
        cnt[i][n-1] -= 1

def check(start, lim):
    return start[0]<0 or start[0]>=lim or start[1]<0 or start[1]>=lim

def dfs(data: list, start, n):
    """
    dfs建连通图
    """
    # print(type(start))
    if mp & {start}:
        return
    mp.add(start)
    for move in walk:
        to = tuple(np.add(start,move))
        if check(to,n):
            continue
        cnt[to[0]][to[1]] -= 1
    # print(start)
    while cnt[start[0]][start[1]]:
        move = randint(0,3)
        to = tuple(np.add(start,walk[move]))
        if check(to,n):
            continue
        data[start[0]][start[1]][move] = 1
        dfs(data, to, n)

def generate(n: int, m: int) -> np.ndarray:
    """
    n表示矩阵的大小, m表示动物的数量
    """
    all = set()
    animals = set()
    houses = set()
    start = ()
    
    data = np.zeros((n,n,11), dtype=np.int32)
    # 前7个元素对应：与上下左右是否有连边，房子的编号(0表示不存在), 动物的编号(0表示不存在)，车是否存在
    # 如果车存在，则后4个元素对应车里的4只动物，否则无意义
    start = (randint(0,n-1),randint(0,n-1))
    all.add(start)
    # data[start[0]][start[1]][6] = 1
    data[start][6] = 1
    id = 1
    while len(animals) < m:
        pos = (randint(0,n-1),randint(0,n-1))
        if all & {pos}:
            continue
        animals.add(pos)
        all.add(pos)
        # data[pos[0]][pos[1]][5] = id
        data[pos][5] = id
        id += 1
    id = 1
    while len(houses) < m:
        pos = (randint(0,n-1),randint(0,n-1))
        if all & {pos}:
            continue
        all.add(pos)
        houses.add(pos)
        # data[pos[0]][pos[1]][4] = id
        data[pos][4] = id
        id += 1
    init(n)
    dfs(data,start,n)
    show(data,n)
    return data

def show(data,n):
    animal_img = Image.open('models/Problem Solving/Pet_Detective/animal.png')
    car_img = Image.open('models/Problem Solving/Pet_Detective/car.png')
    house_img = Image.open('models/Problem Solving/Pet_Detective/house.png')
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    
    for i in range(n):
        for j in range(n):
            house_id = data[i, j, 4]
            if house_id > 0:
                ax.imshow(house_img, extent=(i-0.5, i+0.5, j-0.5, j+0.5))
                ax.text(i, j, str(house_id), fontsize=12, ha='center', va='center', color='yellow')
                
            animal_id = data[i, j, 5]
            if animal_id > 0:
                ax.imshow(animal_img, extent=(i-0.5, i+0.5, j-0.5, j+0.5))
                ax.text(i, j, f'A:{animal_id}', fontsize=10, ha='center', va='center', color='yellow')
            if data[i, j, 6] > 0:
                ax.imshow(car_img, extent=(i-0.5, i+0.5, j-0.5, j+0.5))
                ax.text(i, j, 'car', fontsize=12, ha='center', va='center', color='red')

            for k in range(4):
                if data[i][j][k] > 0:
                    to = tuple(np.add([i,j],walk[k]))
                    ax.plot([i, to[0]], [j, to[1]], color='gray')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(range(n))
    ax.set_yticklabels(range(n))
    plt.grid(False)
    plt.show()


# generate(10,25)