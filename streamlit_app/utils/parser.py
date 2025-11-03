# streamlit_app/utils/parser.py
from bs4 import BeautifulSoup

def parse_html_largest_block(html):
    """
    Parse HTML and return (title, body_text, word_count).
    Strategy: pick the largest text block from article/main/section/div tags.
    """
    if not isinstance(html, str) or not html.strip():
        return "", "", 0

    soup = BeautifulSoup(html, 'lxml')
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    # Candidate containers that often hold main content
    candidates = soup.find_all(['article', 'main', 'section', 'div'])
    best_text = ""
    for c in candidates:
        try:
            text = c.get_text(separator=' ', strip=True)
        except Exception:
            text = ""
        if text and len(text) > len(best_text):
            best_text = text

    # Fallback to paragraphs if nothing found
    if not best_text:
        paragraphs = [p.get_text(separator=' ', strip=True) for p in soup.find_all('p')]
        best_text = ' '.join(paragraphs)

    clean_text = ' '.join(best_text.split())
    word_count = len(clean_text.split()) if clean_text else 0
    return title, clean_text, word_count
