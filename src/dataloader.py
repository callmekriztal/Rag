from pathlib import Path
from typing import List
import json

import pandas as pd
import docx2txt
from pypdf import PdfReader

from langchain_core.documents import Document


def load_pdf(file_path: Path) -> List[Document]:
    docs = []

    reader = PdfReader(str(file_path))

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "type": "pdf",
                    "page": page_num,
                },
            )
        )

    return docs


def load_txt(file_path: Path) -> List[Document]:
    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    return [
        Document(
            page_content=text,
            metadata={
                "source": str(file_path),
                "type": "txt",
            },
        )
    ]


def load_csv(file_path: Path) -> List[Document]:
    df = pd.read_csv(file_path)

    return [
        Document(
            page_content=df.to_string(index=False),
            metadata={
                "source": str(file_path),
                "type": "csv",
            },
        )
    ]


def load_excel(file_path: Path) -> List[Document]:
    sheets = pd.read_excel(
        file_path,
        sheet_name=None,
    )

    docs = []

    for sheet_name, df in sheets.items():
        docs.append(
            Document(
                page_content=df.to_string(index=False),
                metadata={
                    "source": str(file_path),
                    "type": "xlsx",
                    "sheet": sheet_name,
                },
            )
        )

    return docs


def load_docx(file_path: Path) -> List[Document]:
    text = docx2txt.process(str(file_path))

    return [
        Document(
            page_content=text,
            metadata={
                "source": str(file_path),
                "type": "docx",
            },
        )
    ]


def load_json(file_path: Path) -> List[Document]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        Document(
            page_content=json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            ),
            metadata={
                "source": str(file_path),
                "type": "json",
            },
        )
    ]


def load_all_documents(data_dir: str) -> List[Document]:
    data_path = Path(data_dir).resolve()

    documents = []

    loaders = {
        "*.pdf": load_pdf,
        "*.txt": load_txt,
        "*.csv": load_csv,
        "*.xlsx": load_excel,
        "*.docx": load_docx,
        "*.json": load_json,
    }

    for pattern, loader_func in loaders.items():
        files = list(data_path.glob(f"**/{pattern}"))

        print(f"[INFO] Found {len(files)} files matching {pattern}")

        for file_path in files:
            try:
                docs = loader_func(file_path)
                documents.extend(docs)

                print(
                    f"[INFO] Loaded {len(docs)} docs from {file_path.name}"
                )

            except Exception as e:
                print(
                    f"[ERROR] Failed loading {file_path}: {e}"
                )

    print(f"[INFO] Total documents loaded: {len(documents)}")

    return documents