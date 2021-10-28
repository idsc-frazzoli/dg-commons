from casadi import *
from dg_commons_dev.utils import SemiDef
from dataclasses import dataclass
from typing import Union, List, Dict
from dg_commons_dev.utils import BaseParams

""" Fully casadi compatible """


class Empty:
    pass


@dataclass
class QuadraticParams(BaseParams):
    q: Union[List[SemiDef], SemiDef] = SemiDef([0])
    r: Union[List[SemiDef], SemiDef] = SemiDef([0])


class QuadraticCost:
    def __init__(self, params: QuadraticParams):
        self.params = params

    def cost_function(self, x, u):
        r = SX(self.params.r.matrix)
        q = SX(self.params.q.matrix)

        dim_x = len(x)
        dim_u = len(u)
        helper1 = GenSX_zeros(dim_x)
        helper2 = GenSX_zeros(dim_u)

        for i in range(dim_x):
            helper1[i] = x[i]

        for i in range(dim_u):
            helper2[i] = u[i]

        return bilin(q, helper1, helper1) + bilin(r, helper2, helper2), bilin(q, helper1, helper1)


CostFunctions = Union[Empty, QuadraticCost]
CostParameters = Union[Empty, QuadraticParams]
MapCostParam: Dict[type(CostFunctions), type(CostParameters)] = {QuadraticCost: QuadraticParams}
