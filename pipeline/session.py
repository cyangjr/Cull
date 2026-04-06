from __future__ import annotations

import json
import uuid
from pathlib import Path

from .config import PipelineConfig
from .orchestrator import CullPipeline
from .utils import ImageRecord


class SessionManager:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.session_id: str | None = None
        self.folder_path: str | None = None
        self.records: list[ImageRecord] = []
        self.pipeline = CullPipeline(config=config)

    def start(self, folder_path: str, progress_callback=None) -> str:
        self.session_id = str(uuid.uuid4())
        self.folder_path = folder_path
        self.records = self.pipeline.run(folder_path, progress_callback=progress_callback)
        return self.session_id

    def get_kept(self) -> list[ImageRecord]:
        return [r for r in self.records if r.passed_gate and not r.is_duplicate]

    def export_json(self, output_path: str) -> None:
        kept = self.get_kept()
        payload = {
            "session_id": self.session_id,
            "folder_path": self.folder_path,
            "records": [r.to_export_dict() for r in kept],
        }
        out = Path(output_path)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def load_json(json_path: str) -> list[ImageRecord]:
        p = Path(json_path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return [ImageRecord.from_json_dict(d) for d in (data.get("records") or [])]

