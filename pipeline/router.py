from __future__ import annotations

from .utils import ImageRecord


class SceneRouter:
    def classify(self, record: ImageRecord) -> str:
        """
        Milestone C: lightweight heuristic router.
        - If a face is detected, classify as 'object'.
        - Otherwise default to 'scene'.
        """
        if record.has_faces:
            record.scene_type = "object"
        else:
            record.scene_type = record.scene_type or "scene"
        return record.scene_type

