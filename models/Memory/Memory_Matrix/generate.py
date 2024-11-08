from random import randint, choice
import numpy as np

def generate(k: int) -> tuple[list, list]:
    data = []
    label = []
    data = np.random.randint(0,2,(k,k))
    # 0表示没有颜色，1表示有颜色
    
    label = data.copy()
    return data, label
