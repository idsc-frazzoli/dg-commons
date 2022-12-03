from dataclasses import dataclass

from commonroad.visualization.draw_params import (
    LaneletNetworkParams,
    MPDrawParams,
    OptionalSpecificOrAllDrawParams,
    BaseParam,
)
from commonroad.visualization.drawable import IDrawable
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.renderer import IRenderer

from dg_commons import X
from dg_commons.sim.scenarios import load_commonroad_scenario


@dataclass
class TestParam(BaseParam):
    foo: bool = True


class DrawableState(IDrawable):
    def __init__(self, state: X):
        self.state = state

    def draw(self, renderer: IRenderer, draw_params: OptionalSpecificOrAllDrawParams[TestParam] = None) -> None:
        pass


def test_dev_vis():
    r = MPRenderer()

    scenario_name = "USA_Lanker-1_1_T-1"
    scenario, _ = load_commonroad_scenario(scenario_name)

    draw_params = MPDrawParams()
    draw_params.save("default_params.yaml")

    lanelet_net_params = LaneletNetworkParams()
    lanelet_net_params.traffic_light.draw_traffic_lights = True

    scenario.lanelet_network.draw(r, draw_params=lanelet_net_params)
    r.render(filename="vis_tes")
