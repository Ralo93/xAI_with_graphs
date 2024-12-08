from typing import Optional
from pathlib import Path
from pyvis.network import Network
from .templates import VisualizationManager


class BaseVisualizer:
    """Base class for graph visualization"""

    def __init__(
        self,
        height: str = "800px",
        width: str = "100%",
        bgcolor: str = "#222222",
        font_color: str = "white",
    ):
        self.height = height
        self.width = width
        self.bgcolor = bgcolor
        self.font_color = font_color
        self.viz_manager = VisualizationManager()

    def create_network(self) -> Network:
        """Create and configure Pyvis network"""
        net = Network(
            height=self.height,
            width=self.width,
            bgcolor=self.bgcolor,
            font_color=self.font_color,
        )
        self._set_network_options(net)
        return net

    def _set_network_options(self, net: Network) -> None:
        """Set network visualization options"""
        net.set_options(
            """
        var options = {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
          }
        }
        """
        )

    def save_visualization(
        self, net: Network, prefix: str, output_path: Optional[Path] = None
    ) -> Path:
        """Save network visualization"""
        if output_path is None:
            output_path = self.viz_manager.get_output_path(prefix)

        net.write_html(str(output_path))
        return output_path
