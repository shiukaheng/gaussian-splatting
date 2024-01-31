from dataclasses import dataclass


@dataclass
class PipelineParams:
    debug: bool = False
    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False