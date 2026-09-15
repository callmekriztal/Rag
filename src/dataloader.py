import json
import logging
from pathlib import Path
from typing import List, Union

import docx2txt
import pandas as pd
from langchain_core.documents import Document
from pypdf import PdfReader

logger = logging.getLogger("new_rag.dataloader")
logging.basicConfig(level=logging.INFO)


def load_pdf(file_path: Path) -> List[Document]:
    docs = []
    reader = PdfReader(str(file_path))

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path.name,
                        "full_path": str(file_path),
                        "type": "pdf",
                        "page": page_num,
                    },
                )
            )

    return docs


def load_txt(file_path: Path) -> List[Document]:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return []

    return [
        Document(
            page_content=text,
            metadata={
                "source": file_path.name,
                "full_path": str(file_path),
                "type": "txt",
            },
        )
    ]


def load_csv(file_path: Path) -> List[Document]:
    df = pd.read_csv(file_path)
    if df.empty:
        return []

    records = df.to_dict(orient="records")
    row_texts = []
    for idx, row in enumerate(records, start=1):
        formatted_row = ", ".join([f"{k}: {v}" for k, v in row.items() if pd.notna(v)])
        row_texts.append(f"Row {idx}: {formatted_row}")

    full_text = "\n".join(row_texts)
    return [
        Document(
            page_content=full_text,
            metadata={
                "source": file_path.name,
                "full_path": str(file_path),
                "type": "csv",
                "total_rows": len(records),
            },
        )
    ]


def load_excel(file_path: Path) -> List[Document]:
    sheets = pd.read_excel(file_path, sheet_name=None)
    docs = []

    for sheet_name, df in sheets.items():
        if df.empty:
            continue
        records = df.to_dict(orient="records")
        row_texts = []
        for idx, row in enumerate(records, start=1):
            formatted_row = ", ".join([f"{k}: {v}" for k, v in row.items() if pd.notna(v)])
            row_texts.append(f"Row {idx}: {formatted_row}")

        full_text = "\n".join(row_texts)
        docs.append(
            Document(
                page_content=full_text,
                metadata={
                    "source": file_path.name,
                    "full_path": str(file_path),
                    "type": "xlsx",
                    "sheet": sheet_name,
                    "total_rows": len(records),
                },
            )
        )

    return docs


def load_docx(file_path: Path) -> List[Document]:
    text = docx2txt.process(str(file_path)) or ""
    if not text.strip():
        return []

    return [
        Document(
            page_content=text,
            metadata={
                "source": file_path.name,
                "full_path": str(file_path),
                "type": "docx",
            },
        )
    ]


def load_json(file_path: Path) -> List[Document]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        Document(
            page_content=json.dumps(data, indent=2, ensure_ascii=False),
            metadata={
                "source": file_path.name,
                "full_path": str(file_path),
                "type": "json",
            },
        )
    ]


LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_txt,
    ".csv": load_csv,
    ".xlsx": load_excel,
    ".docx": load_docx,
    ".json": load_json,
}


def load_single_file(file_path: Union[str, Path]) -> List[Document]:
    path = Path(file_path).resolve()
    ext = path.suffix.lower()

    if ext not in LOADERS:
        logger.warning(f"Unsupported file extension: {ext} for file {path}")
        return []

    try:
        docs = LOADERS[ext](path)
        logger.info(f"Loaded {len(docs)} document chunks/pages from {path.name}")
        return docs
    except Exception as e:
        logger.error(f"Failed loading {path}: {e}", exc_info=True)
        return []


def load_all_documents(data_dir: Union[str, Path]) -> List[Document]:
    data_path = Path(data_dir).resolve()
    if not data_path.exists():
        logger.warning(f"Data directory does not exist: {data_path}")
        return []

    documents = []
    patterns = ["*.pdf", "*.txt", "*.csv", "*.xlsx", "*.docx", "*.json"]

    for pattern in patterns:
        files = list(data_path.glob(f"**/{pattern}"))
        logger.info(f"Found {len(files)} files matching {pattern}")

        for file_path in files:
            docs = load_single_file(file_path)
            documents.extend(docs)

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents