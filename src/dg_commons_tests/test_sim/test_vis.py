from typing import Union, Tuple, Optional

from commonroad.visualization.drawable import IDrawable
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.param_server import ParamServer, write_default_params
from commonroad.visualization.renderer import IRenderer

from dg_commons import X
from dg_commons.sim.scenarios import load_commonroad_scenario


class DrawableState(IDrawable):
    def __init__(self, state: X):
        self.state = state

    def draw(
        self, renderer: IRenderer, draw_params: Union[ParamServer, dict, None], call_stack: Optional[Tuple[str, ...]]
    ) -> None:
        renderer.draw_rectangle()


def test_dev_vis():
    r = MPRenderer()
    ps = ParamServer()

    scenario_name = "USA_Lanker-1_1_T-1"
    scenario, _ = load_commonroad_scenario(scenario_name)

    write_default_params("default_params.json")

    scenario.lanelet_network.draw(r, draw_params={"draw_traffic_lights": True})
    r.render(filename="vis_tes")
