import math
from typing import Tuple

import torch

from normalizing_flows.bijections.finite.autoregressive.transformers.base import ScalarTransformer
from normalizing_flows.utils import get_batch_shape, sum_except_batch

class LogMixCdf(...):
    pass