import os
import uuid
import fitz  # PyMuPDF
from typing import List
from core.interfaces.document_loader import BaseDocumentLoader
from core.models.schemas import Document


class PDFDocumentLoader(BaseDocumentLoader):
    """
    Loads a PDF page-by-page, extracting both the page's text (baseline
    TextRAG source) and any embedded images (VisRAG visual source).
    """

    def __init__(self, image_output_dir: str, min_image_size: int = 100):
        self._image_output_dir = image_output_dir
        self._min_image_size = min_image_size
        os.makedirs(self._image_output_dir, exist_ok=True)

    def load(self, source_path: str) -> List[Document]:
        documents = []
        pdf = fitz.open(source_path)
        base_name = os.path.splitext(os.path.basename(source_path))[0]

        for page_index in range(len(pdf)):
            page = pdf[page_index]
            text = page.get_text("text").strip()
            image_paths = self._extract_images(pdf, page, page_index, base_name)
            doc_id = f"{base_name}_p{page_index}_{uuid.uuid4().hex[:8]}"
            documents.append(
                Document(
                    doc_id=doc_id,
                    source_path=source_path,
                    page_number=page_index + 1,
                    text=text,
                    image_paths=image_paths,
                    metadata={"file_name": os.path.basename(source_path)},
                )
            )
        pdf.close()
        return documents

    def _extract_images(self, pdf, page, page_index: int, base_name: str) -> List[str]:
        saved_paths = []
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                base_image = pdf.extract_image(xref)
            except Exception:
                continue
            if base_image.get("width", 0) < self._min_image_size or base_image.get("height", 0) < self._min_image_size:
                continue
            ext = base_image.get("ext", "png")
            file_name = f"{base_name}_p{page_index}_img{img_index}.{ext}"
            file_path = os.path.join(self._image_output_dir, file_name)
            with open(file_path, "wb") as f:
                f.write(base_image["image"])
            saved_paths.append(file_path)
        return saved_paths