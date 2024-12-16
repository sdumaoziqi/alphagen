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


_T = TypeVar("_T")


def _create_parents(path: str) -> None:
    dir = os.path.dirname(path)
    if dir != "":
        os.makedirs(dir, exist_ok=True)


def write_all_text(path: str, text: str) -> None:
    _create_parents(path)
    with open(path, "w") as f:
        f.write(text)


def dump_pickle(path: str,
                factory: Callable[[], _T],
                invalidate_cache: bool = False) -> Optional[_T]:
    if invalidate_cache or not os.path.exists(path):
        _create_parents(path)
        obj = factory()
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return obj


class BacktestResult(dict):
    sharpe: float
    annual_return: float
    max_drawdown: float
    information_ratio: float
    annual_excess_return: float
    excess_max_drawdown: float


class QlibBacktest:
    def __init__(
        self,
        benchmark: str = "SH000300",
        top_k: int = 30,
        n_drop: Optional[int] = 5,
        deal: str = "close",
        open_cost: float = 0.0015,
        close_cost: float = 0.0015,
        min_cost: float = 5,
    ):
        self._benchmark = benchmark
        self._top_k = top_k
        self._n_drop = n_drop if n_drop is not None else top_k
        self._deal_price = deal
        self._open_cost = open_cost
        self._close_cost = close_cost
        self._min_cost = min_cost

    def run(
        self,
        prediction: pd.Series,
        output_prefix: Optional[str] = None,
        return_report: bool = False
    ) -> BacktestResult:
        prediction = prediction.sort_index()
        index: pd.MultiIndex = prediction.index.remove_unused_levels()  # type: ignore
        dates = index.levels[0]

        def backtest_impl(last: int = -1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                strategy=TopKSwapNStrategy(
                    K=self._top_k,
                    n_swap=self._n_drop,
                    signal=prediction,
                    min_hold_days=20,
                    only_tradable=True,
                )
                executor=exec.SimulatorExecutor(
                    time_per_step="day",
                    # verbose=True,
                    # track_data=True,
                    generate_portfolio_metrics=True
                )
                return backtest(
                    strategy=strategy,
                    executor=executor,
                    start_time=dates[0],
                    end_time=dates[last],
                    account=100_000_000,
                    benchmark=self._benchmark,
                    exchange_kwargs={
                        "limit_threshold": 0.095,
                        "deal_price": self._deal_price,
                        "open_cost": self._open_cost,
                        "close_cost": self._close_cost,
                        "min_cost": self._min_cost,
                    }
                )[0]

        try:
            portfolio_metric = backtest_impl()
        except IndexError:
            print("Cannot backtest till the last day, trying again with one less day")
            portfolio_metric = backtest_impl(-2)

        report, _ = portfolio_metric["1day"]    # type: ignore
        result = self._analyze_report(report)
        if output_prefix is not None:
            def dump_df_png(df):
                import matplotlib.pyplot as plt
                
                num_columns = df.shape[1]
                fig, axes = plt.subplots(num_columns, 1, figsize=(6, 4 * num_columns))  # 调整figsize以适应所有子图

                for i, column in enumerate(df.columns):
                    axes[i].plot(df.index, df[column], marker='.')  # 使用折线图
                    axes[i].set_title(f'{column}')  # 设置标题
                    axes[i].set_xlabel('Date')  # 设置x轴标签
                    axes[i].set_ylabel(column)  # 设置y轴标签
                    axes[i].grid(True)  # 显示网格

                # 调整布局以防止重叠
                plt.tight_layout()

                # 保存整体图像
                plt.savefig(output_prefix + '-combined_plot.png', format='png', dpi=300)
            report["excess"] = report["return"] - report["bench"] - report["cost"]
            report["return"] = report["return"] - report["cost"]

            report["cum_return"] = report["return"].cumsum()
            report["cum_bench"] = report["bench"].cumsum()
            # report["cum_cost"] = report["cost"].cumsum()
            report["cum_excess"] = report["excess"].cumsum()
            cum_cols = [col for col in report.columns if "cum_" in col]
            cum_df = report[cum_cols]
            dump_df_png(cum_df)
            # dump_pickle(output_prefix + "-report.pkl", lambda: report, True)
            # dump_pickle(output_prefix + "-graph.pkl", lambda: graph, True)
            import json
            write_all_text(output_prefix + "-result.json", json.dumps(result, indent=4))

        # print(report)
        # print(result)
        return report if return_report else result

    def _analyze_report(self, report: pd.DataFrame) -> BacktestResult:
        print(report)
        excess = risk_analysis(report["return"] - report["bench"] - report["cost"])["risk"]
        returns = risk_analysis(report["return"] - report["cost"])["risk"]

        def loc(series: pd.Series, field: str) -> float:
            return series.loc[field]    # type: ignore

        # print(returns)
        return BacktestResult(
            sharpe=loc(returns, "information_ratio"),
            annual_return=loc(returns, "annualized_return"),
            max_drawdown=loc(returns, "max_drawdown"),
            information_ratio=loc(excess, "information_ratio"),
            annual_excess_return=loc(excess, "annualized_return"),
            excess_max_drawdown=loc(excess, "max_drawdown"),
        )

def foo(file_name):
    with open(file_name, "r") as f:
        pool_expr = json.load(f)
    new_exprs = []
    weights = []
    for expr, weight in zip(pool_expr["exprs"], pool_expr["weights"]):
        expr = expr.replace("$", "").replace("open", "open_")
        new_exprs.append(eval(expr))
        weights.append(weight)
    return new_exprs, weights


if __name__ == "__main__":
    import json
    import time
    qlib_backtest = QlibBacktest()

    data = StockData(instrument='csi300',
                     start_time='2020-01-01',
                     end_time='2020-12-31')
    target = Ref(close, -20) / close - 1
    calculator = QLibStockDataCalculator(data=data, target=target)
    
    t0 = time.time()
    cp_path = "/home/zmao/github/alphagen/train/checkpoints/new_csi300_20_1_20241212173142"
    # step_name = "2048_steps_pool"
    # step_name = "30720_steps_pool"
    step_name = "81920_steps_pool"
    file_name = f"{cp_path}/{step_name}.json"
    exprs, weights = foo(file_name)
    fcst = calculator.make_ensemble_alpha(exprs, weights)
    t1 = time.time()

    data_df = data.make_dataframe(fcst)
    t2 = time.time()
    qlib_backtest.run(data_df, output_prefix = f"{cp_path}/{step_name}")
    t3 = time.time()
    print(f"{t1 - t0:.3f}, {t2 - t1:.3f}, {t3 - t2:.3f}")

