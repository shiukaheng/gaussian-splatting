import networkx as nx
from sgs2.scene import Scene

def build_camera_graph(scene: Scene):
    print(f"Found {len(scene)} cameras in scene")