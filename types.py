from typing import Optional, Tuple

from torch import Tensor

SparseTensor = type(None)

Adj = Tensor
OptTensor = Optional[Tensor]
OptPairTensor = Optional[Tuple[Tensor, Tensor]]