from pathlib import Path
from bs4 import BeautifulSoup
import re


def clean_html(file_path: str) -> str:
    """Extract readable text from an HTML file."""
    path = Path(file_path)

    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")

    # Remove technical elements
    for element in soup([
        "script", "style", "noscript", "svg",
        "iframe", "header", "footer", "nav"
    ]):
        element.decompose()

    # Remove hidden elements
    for element in soup.select(
        '[hidden], [aria-hidden="true"], '
        '[style*="display:none"], [style*="display: none"]'
    ):
        element.decompose()

    # Extract readable text
    text = soup.get_text(separator="\n")

    # Normalize whitespace
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    return "\n".join(lines)


if __name__ == "__main__":
    print("HTML cleaner module ready.")
