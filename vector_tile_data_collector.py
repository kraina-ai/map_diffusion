"""This module contains classes of strategy for handling downloaded tiles."""
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class VectorDataCollector(ABC):
    """Stores collected images."""

    @abstractmethod
    def store(self, x: int, y: int, data: Any) -> Any:
        pass

    def exists(self, x: int, y: int) -> bool:
        pass

    def restore(self, x: int, y: int) -> Any:
        pass


class SavingVectorDataCollector(VectorDataCollector):
    """
    Saves in disk collected images.

    Stores paths.
    """

    def __init__(self, save_path: str | Path, f_extension: str) -> None:
        super().__init__()
        self.save_path = save_path
        self.format = f_extension

    def store(self, x: int, y: int, data: Any) -> str:
        if data is None:
            return None
        path = os.path.join(self.save_path, f"{x}_{y}.{self.format}")
        tile_file = open(path, 'wb')
        tile_file.write(data)
        tile_file.close()
        return path

    def exists(self, x: int, y: int) -> bool:
        path = os.path.join(self.save_path, f"{x}_{y}.{self.format}")
        return os.path.exists(path)

    def restore(self, x: int, y: int) -> Any:
        return os.path.join(self.save_path, f"{x}_{y}.{self.format}")


class InMemoryVectorDataCollector(VectorDataCollector):
    """Stores data in object memory."""

    def __init__(self) -> None:
        """Initialize InMemoryDataCollector."""
        super().__init__()

    def store(self, x: int, y: int, data: Any) -> Any:
        return data

    def exists(self, x: int, y: int) -> bool:
        return False

    def restore(self, x: int, y: int) -> Any:
        pass
