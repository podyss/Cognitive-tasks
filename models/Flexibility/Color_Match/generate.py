from random import randint, choice
import numpy as np

colors = [0,1,2,3] # 4种颜色

def generate() -> tuple[list, int]:
    data = []
    label = 0
    for i in range(2):
        # [mean_color, text_color] mean_color是单词的意思，text_color是单词的颜色
        data.append([choice(colors), choice(colors)])

    label = data[0][0] == data[1][1]
    return data, label
    