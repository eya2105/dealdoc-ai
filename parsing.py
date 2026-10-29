"""HTML filing -> clean text."""
import re

from bs4 import BeautifulSoup


def parse_document(file_path: str) -> str:
    """Extract clean text from an HTML SEC filing.

    The file is read as bytes so BeautifulSoup can honour the charset declared
    in the document (<meta charset=...>) instead of hard-coding it.
    """
    with open(file_path, "rb") as f:
        raw = f.read()
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()
