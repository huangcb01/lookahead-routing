from typing import Type

from .base import BaseRouter
from .fixed import FixedRouter
from .random import RandomRouter
from .oracle import OracleRouter
from .reward import RewardRouter
from .knn import KNNRouter
from .soft_cluster import SoftClusterRouter
from .hard_cluster import HardClusterRouter


ROUTERS: dict[str, Type[BaseRouter]] = {
    "random": RandomRouter,
    "fixed": FixedRouter,
    "oracle": OracleRouter,
    "reward": RewardRouter,
    "knn": KNNRouter,
    "soft_cluster": SoftClusterRouter,
    "hard_cluster": HardClusterRouter,
}

__all__ = [
    "ROUTERS",
    "BaseRouter",
    "RandomRouter",
    "FixedRouter",
    "OracleRouter",
    "RewardRouter",
    "KNNRouter",
    "SoftClusterRouter",
    "HardClusterRouter",
]
