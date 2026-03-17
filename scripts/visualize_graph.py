"""Generate LangGraph visualization as PNG images."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.graph import build_agent_graph, build_simple_graph


def main():
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    simple_graph = build_simple_graph()
    agent_graph = build_agent_graph()

    simple_png = simple_graph.get_graph().draw_mermaid_png()
    (output_dir / "simple_graph.png").write_bytes(simple_png)
    print(f"Saved: {output_dir / 'simple_graph.png'}")

    agent_png = agent_graph.get_graph().draw_mermaid_png()
    (output_dir / "agent_graph.png").write_bytes(agent_png)
    print(f"Saved: {output_dir / 'agent_graph.png'}")


if __name__ == "__main__":
    main()
