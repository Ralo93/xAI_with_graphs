from pathlib import Path
from typing import Optional
from datetime import datetime


class VisualizationManager:
    """Manage visualization templates and output"""

    DEFAULT_TEMPLATE = """
    :::: {.card style="width: 100%"}
    ::: {#mynetwork .card-body}
    :::
    ::::
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("src/visualization/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_output_path(self, prefix: str) -> Path:
        """Generate unique output path for visualization"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.output_dir / f"{prefix}_{timestamp}.html"

    def save_visualization(self, content: str, prefix: str = "graph") -> Path:
        """Save visualization to file"""
        output_path = self.get_output_path(prefix)
        output_path.write_text(content)
        return output_path

    def clean_old_visualizations(self, max_age_days: int = 7) -> None:
        """Remove visualizations older than specified days"""
        now = datetime.now()
        for file in self.output_dir.glob("*.html"):
            if (
                file.stat().st_mtime
                < (now - datetime.timedelta(days=max_age_days)).timestamp()
            ):
                file.unlink()


# Add to .gitignore
gitignore_entry = """
# Visualization outputs
src/visualization/output/
"""
