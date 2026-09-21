"""资产清单：记录每个已转换 USD 的来源、校验和与体检结果。

用途：
- 让场景构建器按「实体名 → USD 路径」解析资产；
- 记录转换时的关键决策（fix_base、注入自由关节、damping）与体检结果，
  供物理审计（experiments/physics_audit.py）复核。
"""

from __future__ import annotations

import json
import os
import hashlib
from typing import Dict, Optional

DEFAULT_CACHE_DIR = os.environ.get(
    "LIBERO_ISAAC_ASSETS_DIR", os.path.expanduser("~/.cache/libero_isaac_sim")
)

REGISTRY_FILENAME = "asset_registry.json"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class AssetRegistry:
    """``<cache>/usd/asset_registry.json`` 的读写封装。"""

    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        self.cache_dir = cache_dir
        self.path = os.path.join(cache_dir, "usd", REGISTRY_FILENAME)
        self._data: Dict[str, dict] = {}
        if os.path.exists(self.path):
            with open(self.path) as f:
                self._data = json.load(f)

    def register(
        self,
        entity_name: str,
        category: str,
        kind: str,
        usd_path: str,
        source_mjcf: str,
        fix_base: bool,
        report: Optional[dict] = None,
    ) -> None:
        self._data[entity_name] = {
            "category": category,
            "kind": kind,
            "usd_path": os.path.abspath(usd_path),
            "usd_sha256": _sha256(usd_path) if os.path.exists(usd_path) else None,
            "source_mjcf": os.path.abspath(source_mjcf),
            "fix_base": bool(fix_base),
            "report": report or {},
        }

    def get(self, entity_name: str) -> dict:
        assert entity_name in self._data, (
            f"实体 {entity_name} 未注册；请先运行 converters/mjcf_to_usd.py"
        )
        return self._data[entity_name]

    def usd_path(self, entity_name: str) -> str:
        return self.get(entity_name)["usd_path"]

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def __contains__(self, entity_name: str) -> bool:
        return entity_name in self._data

    def __len__(self) -> int:
        return len(self._data)
