from typing import Optional, TypeVar, Callable, Optional
import os
import pickle
import warnings
import pandas as pd
from qlib.backtest import backtest, executor as exec
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.report.analysis_position import report_graph
from alphagen.data.expression import *

from alphagen_qlib.stock_data import StockData
from alphagen_generic.features import *
from alphagen_qlib.strategy import TopKSwapNStrategy
from alphagen_qlib.calculator import QLibStockDataCalculator
import json


_T = TypeVar("_T")

data = StockData(instrument='csi300',
                    start_time='2020-01-01',
                    end_time='2020-12-31')

target = Ref(close, -20) / close - 1
calculator = QLibStockDataCalculator(data=data, target=target)




def foo(file_name):
    with open(f"/home/zmao/github/alphagen/train/checkpoints/new_csi300_10_1_20241206134324/{file_name}.json", "r") as f:
        pool_expr = json.load(f)
    new_exprs = []
    weights = []
    for expr, weight in zip(pool_expr["exprs"], pool_expr["weights"]):
        expr = expr.replace("$", "").replace("open", "open_")
        new_exprs.append(eval(expr))
        weights.append(weight)
        # new_exprs.append(f"Mul({expr},Constant({weight}))")
    return new_exprs, weights

def calc(file_name):
    exprs, weights = foo(file_name)
    ret = calculator.calc_pool_all_ret(exprs, weights)
    print(f"{file_name}: {ret}")
    return ret

def plot(data_list):
    import matplotlib.pyplot as plt
    num_columns = len(data_list)
    fig, axes = plt.subplots(num_columns, 1, figsize=(6, 4 * num_columns))  # 调整figsize以适应所有子图

    for i, data in enumerate(data_list):
        axes[i].plot(range(len(data)), data, marker='.')  # 使用折线图
        axes[i].set_title(f'{i}')  # 设置标题
        axes[i].set_xlabel('index')  # 设置x轴标签
        axes[i].grid(True)  # 显示网格

    # 调整布局以防止重叠
    plt.tight_layout()

    # 保存整体图像
    plt.savefig('plot_ic_ric.png', format='png', dpi=300)

if __name__ == "__main__":
    ic = []
    ric = []
    for step in range(2048, 47104, 2048):
        file_name = f"{step}_steps_pool"
        ret = calc(file_name)
        ic.append(ret[0])
        ric.append(ret[1])
    
    plot([ic, ric])


