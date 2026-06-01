"""
Email HTML patcher — utilidades para emails de CIO en Kepler.

Estrategia (prototipo):
  - Claude recibe el HTML completo del email y devuelve el mismo HTML
    con solo el texto visible actualizado.
  - Al actualizar en CIO: PUT con el HTML de Claude directamente.
  - Canvas muestra texto legible (stripped) para revisión y edición.
"""

import re
from bs4 import BeautifulSoup


def extract_editable_text(full_html: str) -> str:
    """
    Extrae texto legible del HTML completo del email para mostrar en el canvas.
    Devuelve texto plano con una línea por párrafo/ítem significativo.
    """
    if not full_html:
        return ""
    try:
        soup = BeautifulSoup(full_html, "html.parser")
        # Eliminar style, script, head
        for tag in soup(["style", "script", "head"]):
            tag.decompose()
        lines = []
        for el in soup.find_all(["p", "li"]):
            text = el.get_text(separator=" ", strip=True)
            if text and text != "\xa0" and len(text) > 2:
                lines.append(text)
        if lines:
            return "\n".join(lines)
    except Exception:
        pass
    return _fallback_strip(full_html)


def _fallback_strip(html: str) -> str:
    s = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<script[^>]*>.*?</script>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("&nbsp;", " ").replace("&amp;", "&")
    return " ".join(s.split())
