from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent.parent / ".env")


@dataclass
class SearchResult:
    url: str = ""
    title: str = ""
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Searcher(ABC):
    name: str = "base"

    @abstractmethod
    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        pass

