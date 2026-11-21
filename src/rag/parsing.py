"""Turn a 10-K HTML document into clean text chunks tagged with their section."""

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import config

# Standard 10-K sections ("Item 1A. Risk Factors", "Item 7. MD&A", ...)
ITEM_NAMES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Common Equity",
    "6": "Reserved",
    "7": "Management's Discussion and Analysis",
    "7A": "Market Risk Disclosures",
    "8": "Financial Statements",
    "9": "Changes in Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Foreign Jurisdiction Inspections",
    "10": "Directors and Officers",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Relationships and Transactions",
    "14": "Accountant Fees",
    "15": "Exhibits",
    "16": "Form 10-K Summary",
}
# A short line that starts with "Item 7." / "ITEM 1A:" etc. is treated as a section heading
ITEM_HEADING = re.compile(r"^item\s+(\d{1,2}[a-c]?)\s*[.:\-—–|]", re.IGNORECASE)


@dataclass
class Chunk:
    text: str
    section: str


def html_to_text(html: bytes) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "ix:header"]):  # ix:header = hidden XBRL data
        tag.decompose()

    # Keep tables readable: one row per line, cells separated by " | "
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with("\n" + "\n".join(rows) + "\n")

    lines = (re.sub(r"\s+", " ", line).strip() for line in soup.get_text("\n").splitlines())
    return "\n".join(line for line in lines if line)


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split text into (section name, section text) pairs using Item headings."""
    sections: list[tuple[str, list[str]]] = [("Cover / Other", [])]
    for line in text.splitlines():
        match = ITEM_HEADING.match(line)
        if match and len(line) < 150:
            item = match.group(1).upper()
            if item in ITEM_NAMES:
                sections.append((f"Item {item}: {ITEM_NAMES[item]}", []))
        sections[-1][1].append(line)
    return [(name, "\n".join(lines)) for name, lines in sections if lines]


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
)


def chunk_filing(html: bytes) -> list[Chunk]:
    chunks = []
    for section, body in split_sections(html_to_text(html)):
        for piece in _splitter.split_text(body):
            if len(piece) > 100:  # drop page numbers, stray headers
                chunks.append(Chunk(text=piece, section=section))
    return chunks
