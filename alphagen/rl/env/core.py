from typing import Tuple, Optional
import gymnasium as gym
import math

from alphagen.config import MAX_EXPR_LENGTH
from alphagen.data.tokens import *
from alphagen.data.expression import *
from alphagen.data.tree import ExpressionBuilder
from alphagen.models.alpha_pool import AlphaPoolBase, AlphaPool
from alphagen.utils import reseed_everything
import numpy as np


class AlphaEnvCore(gym.Env):
    pool: AlphaPoolBase
    _tokens: List[Token]
    _builder: ExpressionBuilder
    _print_expr: bool

    def __init__(self,
                 pool: AlphaPoolBase,
                 device: torch.device = torch.device('cuda:0'),
                 print_expr: bool = False
                 ):
        super().__init__()

        self.pool = pool
        self._print_expr = print_expr
        self._device = device

        self.eval_cnt = 0

        self.render_mode = None

    def reset(
        self, *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ) -> Tuple[List[Token], dict]:
        reseed_everything(seed)
        self._tokens = [BEG_TOKEN]
        self._builder = ExpressionBuilder()
        return self._tokens, self._valid_action_types()

    def step(self, action: Token) -> Tuple[List[Token], float, bool, bool, dict]:
        if (isinstance(action, SequenceIndicatorToken) and
                action.indicator == SequenceIndicatorType.SEP):
            reward = self._evaluate()
            done = True
        elif len(self._tokens) < MAX_EXPR_LENGTH:
            self._tokens.append(action)
            self._builder.add_token(action)
            done = False
            reward = 0.0
        else:
            done = True
            reward = self._evaluate() if self._builder.is_valid() else -1.

        if math.isnan(reward):
            reward = 0.

        truncated = False  # Fk gymnasium
        return self._tokens, reward, done, truncated, self._valid_action_types()

    def _evaluate(self):
        expr: Expression = self._builder.get_tree()
        if self._print_expr:
            print(expr)
        try:
            ret = self.pool.try_new_expr(expr)
            self.eval_cnt += 1
            return ret
        except OutOfDataRangeError:
            return 0.

    def _valid_action_types(self) -> dict:
        valid_op_unary = self._builder.validate_op(UnaryOperator)
        valid_op_binary = self._builder.validate_op(BinaryOperator)
        valid_op_rolling = self._builder.validate_op(RollingOperator)
        valid_op_pair_rolling = self._builder.validate_op(PairRollingOperator)

        valid_op = valid_op_unary or valid_op_binary or valid_op_rolling or valid_op_pair_rolling
        valid_dt = self._builder.validate_dt()
        valid_const = self._builder.validate_const()
        valid_feature = self._builder.validate_feature()
        valid_stop = self._builder.is_valid()

        ret = {
            'select': [valid_op, valid_feature, valid_const, valid_dt, valid_stop],
            'op': {
                UnaryOperator: valid_op_unary,
                BinaryOperator: valid_op_binary,
                RollingOperator: valid_op_rolling,
                PairRollingOperator: valid_op_pair_rolling
            }
        }
        return ret

    def valid_action_types(self) -> dict:
        return self._valid_action_types()

    def render(self, mode='human'):
        pass

from alphagen_generic.features import *
class EcxessEnvCore(gym.Env):
    _tokens: List[Token]
    _builder: ExpressionBuilder
    _print_expr: bool

    def __init__(self,
                 data_train: StockData,
                 data_valid: StockData,
                 device: torch.device = torch.device('cuda:0'),
                 print_expr: bool = False
                 ):
        super().__init__()

        self._print_expr = print_expr
        self._device = device
        self.eval_cnt = 0
        self.render_mode = None
        from backtest import QlibBacktest
        from alphagen_qlib.calculator import QLibStockDataCalculator
        self.qlib_backtest = QlibBacktest()
        self.data = data_train
        self.calculator = QLibStockDataCalculator(data=self.data, target=None)

        self.pool_size = 10
        self.best_expr = []
        self.best_reward = []


    def reset(
        self, *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ) -> Tuple[List[Token], dict]:
        reseed_everything(seed)
        self._tokens = [BEG_TOKEN]
        self._builder = ExpressionBuilder()
        return self._tokens, self._valid_action_types()
    
    def step(self, action: Token) -> Tuple[List[Token], float, bool, bool, dict]:
        if (isinstance(action, SequenceIndicatorToken) and
                action.indicator == SequenceIndicatorType.SEP):
            reward = self._evaluate()
            done = True
        elif len(self._tokens) < MAX_EXPR_LENGTH:
            self._tokens.append(action)
            self._builder.add_token(action)
            done = False
            reward = 0.0
        else:
            done = True
            reward = self._evaluate() if self._builder.is_valid() else -1.

        if math.isnan(reward):
            reward = 0.

        truncated = False  # Fk gymnasium
        return self._tokens, reward, done, truncated, self._valid_action_types()

    def _evaluate(self):
        expr: Expression = self._builder.get_tree()
        if self._print_expr:
            print(">>>>>>>> ", expr)
        try:
            fcst = self.calculator._calc_alpha(expr)
            data_df = self.data.make_dataframe(fcst)
            ret = self.qlib_backtest.run(data_df)
            ret = ret["annual_excess_return"]
            self.eval_cnt += 1
            if ret > 0:
                if len(self.best_expr) < self.pool_size:
                    self.best_expr.append(expr)
                    self.best_reward.append(ret)
                else:
                    idx = np.argmin(np.array(self.best_reward))
                    self.best_expr[idx] = expr
                    self.best_reward[idx] = ret
            return ret
        except OutOfDataRangeError:
            return 0.

    def _valid_action_types(self) -> dict:
        valid_op_unary = self._builder.validate_op(UnaryOperator)
        valid_op_binary = self._builder.validate_op(BinaryOperator)
        valid_op_rolling = self._builder.validate_op(RollingOperator)
        valid_op_pair_rolling = self._builder.validate_op(PairRollingOperator)

        valid_op = valid_op_unary or valid_op_binary or valid_op_rolling or valid_op_pair_rolling
        valid_dt = self._builder.validate_dt()
        valid_const = self._builder.validate_const()
        valid_feature = self._builder.validate_feature()
        valid_stop = self._builder.is_valid()

        ret = {
            'select': [valid_op, valid_feature, valid_const, valid_dt, valid_stop],
            'op': {
                UnaryOperator: valid_op_unary,
                BinaryOperator: valid_op_binary,
                RollingOperator: valid_op_rolling,
                PairRollingOperator: valid_op_pair_rolling
            }
        }
        return ret
    
    

    def valid_action_types(self) -> dict:
        return self._valid_action_types()

    def render(self, mode='human'):
        pass