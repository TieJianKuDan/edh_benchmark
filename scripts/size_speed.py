import math

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

model_names = [
    "ConvLSTM",
    "PredRNN",
    "CrevNet",
    "SA-ConvLSTM",
    "PredRNN-v2",
    "SmaAt-UNet",
    "SimVP",
    "SimVP2",
    "TAU",
    "MMVP",
]

params = [
    1.3,
    8.9,
    47.8,
    1.8,
    8.9,
    4.1,
    33,
    5.5,
    5.2,
    20.4,
]

speed = [
    16.91,
    47.70,
    223.30,
    36.28,
    51.96,
    12.35,
    23.91,
    11.33,
    10.61,
    224.23,
]

mae = [
    2.0695,
    2.0168,
    3.8086,
    2.1302,
    2.0234,
    2.0514,
    2.0375,
    1.9731,
    1.9760,
    2.6730,
]

if __name__ == "__main__":
    params = [param * 1e6 for param in params]
    speed = [math.log(ele) for ele in speed]
    info = {  
        'speed': speed,  
        'mae': mae,
        'model': model_names,
        'param': params  
    }
    plt.figure(figsize=(5, 5))
    df = pd.DataFrame(info)
    sns.scatterplot(
        x='speed',
        y='mae', 
        hue='model', 
        size="param",
        data=df,
        sizes=(100, 500)
    )  
    plt.xlabel('Speed(min/epo)', fontdict={'family': 'Times New Roman', 'size': 18})
    plt.ylabel('MAE', fontdict={'family': 'Times New Roman', 'size': 18})
    plt.xlim([2.3, 5.7])
    plt.xticks(np.arange(2.5, 5.7, 1), fontsize=18, fontname='Times New Roman')
    plt.ylim((1.7, 4))
    plt.yticks([1.8, 2.5, 3.2, 3.9], fontsize=18, fontname='Times New Roman')
    plt.legend(loc='center right', borderaxespad=3.5)
    plt.tight_layout()
    plt.grid()
    plt.savefig("imgs/size_speed_mae1.jpg", dpi=1000)

    print("pause")