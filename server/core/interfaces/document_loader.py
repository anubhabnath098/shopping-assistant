from abc import ABC, abstractmethod
from typing import List
from core.models.schemas import Document


class BaseDocumentLoader(ABC):
    """Contract for turning a raw source file into a list of Documents."""

    @abstractmethod
    def load(self, source_path: str) -> List[Document]:
        raise NotImplementedError