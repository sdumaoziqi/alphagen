import json
import os
from typing import Optional, Tuple
from datetime import datetime
import fire

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from alphagen.data.calculator import AlphaCalculator

from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool, AlphaPoolBase
from alphagen.rl.env.wrapper import AlphaEnv, ExcessEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils.random import reseed_everything
from alphagen.rl.env.core import AlphaEnvCore
from alphagen_qlib.calculator import QLibStockDataCalculator
from backtest import QlibBacktest

class CustomCallback(BaseCallback):
    def __init__(self,
                 save_freq: int,
                 show_freq: int,
                 save_path: str,
                 data_valid: StockData,
                 data_test: StockData,
                 name_prefix: str = 'rl_model',
                 timestamp: Optional[str] = None,
                 verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.show_freq = show_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

        self.data_valid = data_valid
        self.data_test = data_test

        self.test_qlib_backtest = QlibBacktest()
        self.test_calculator = QLibStockDataCalculator(data=self.data_test, target=None)

        if timestamp is None:
            self.timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        else:
            self.timestamp = timestamp

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        assert self.logger is not None

        self.logger.record("train_excess/size", len(self.best_reward))
        if len(self.best_reward) > 0:
            self.logger.record("train_excess/max", np.max(np.array(self.best_reward)))
            self.logger.record("train_excess/mean", np.mean(np.array(self.best_reward)))
        
        test_reward = []
        for expr in self.best_expr:
            fcst = self.test_calculator._calc_alpha(expr)
            data_df = self.data_test.make_dataframe(fcst)
            ret = self.test_qlib_backtest.run(data_df)
            ret = ret["annual_excess_return"]
            test_reward.append(ret)
        if len(test_reward) > 0:
            self.logger.record("test_excess/max", np.max(np.array(test_reward)))
            self.logger.record("test_excess/mean", np.mean(np.array(test_reward)))
        self.save_checkpoint()

    def save_checkpoint(self):
        path = os.path.join(self.save_path, f'{self.name_prefix}_{self.timestamp}', f'{self.num_timesteps}_steps')
        self.model.save(path)   # type: ignore
        if self.verbose > 1:
            print(f'Saving model checkpoint to {path}')
        with open(f'{path}_pool.json', 'w') as f:
            json.dump({"exprs": [str(expr) for expr in self.best_expr], "reward": self.best_reward}, f)

    @property
    def best_reward(self):
        return self.env_core.best_reward
    @property
    def best_expr(self):
        return self.env_core.best_expr

    @property
    def env_core(self) -> AlphaEnvCore:
        return self.training_env.envs[0].unwrapped  # type: ignore


def main(
    seed: int = 0,
    instruments: str = "csi300",
    pool_capacity: int = 10,
    steps: int = 200_000
):
    reseed_everything(seed)

    device = torch.device('cpu')
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    # You can re-implement AlphaCalculator instead of using QLibStockDataCalculator.
    data_train = StockData(instrument=instruments,
                           start_time='2017-01-01',
                           end_time='2019-07-31')
    data_valid = StockData(instrument=instruments,
                           start_time='2020-01-01',
                           end_time='2021-12-31')
    data_test = StockData(instrument=instruments,
                          start_time='2020-01-01',
                          end_time='2022-12-31')
    env = ExcessEnv(data_train, data_valid)

    name_prefix = f"new_{instruments}_{pool_capacity}_{seed}"
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    checkpoint_callback = CustomCallback(
        save_freq=10000,
        show_freq=10000,
        save_path='/home/zmao/github/alphagen/train/checkpoints',
        data_valid=data_valid,
        data_test=data_test,
        name_prefix=name_prefix,
        timestamp=timestamp,
        verbose=1,
    )

    model = MaskablePPO(
        'MlpPolicy',
        env,
        policy_kwargs=dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=2,
                d_model=128,
                dropout=0.1,
                device=device,
            ),
        ),
        gamma=1.,
        ent_coef=0.01,
        batch_size=128,
        tensorboard_log='/home/zmao/github/alphagen/train/tb/log',
        device=device,
        verbose=1,
    )
    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        tb_log_name=f'{name_prefix}_{timestamp}',
    )


def fire_helper(
    seed: Union[int, Tuple[int]],
    code: str,
    pool: int,
    step: int = None
):
    if isinstance(seed, int):
        seed = (seed, )
    default_steps = {
        5: 250_000,
        10: 250_000,
        20: 300_000,
        50: 350_000,
        100: 400_000
    }
    for _seed in seed:
        main(_seed,
             code,
             pool,
             default_steps[int(pool)] if step is None else int(step)
             )


if __name__ == '__main__':
    fire.Fire(fire_helper)
