"""
parse_selectors.py
──────────────────
Parses a Playwright codegen output file (.py) and converts it into
selector_map.json — the master index used by the Hushly Guide extension.

Supports both Playwright codegen styles:
  Old:  page.click('[data-testid="create-btn"]')
  New:  page.get_by_role("button", name="Create Asset").click()

Usage:
  python parse_selectors.py hushly_selectors.py
  python parse_selectors.py hushly_selectors.py selector_map.json
  python parse_selectors.py --merge hushly_new.py selector_map.json
"""

import ast
import json
import re
import sys
import uuid
from datetime import date
from pathlib import Path
from urllib.parse import urlparse


# ─────────────────────────────────────────────
# SELECTOR → HUMAN NAME  (best-effort)
# ─────────────────────────────────────────────

def name_from_selector(selector: str, method: str) -> str:
    """Turn a raw CSS / Playwright selector string into a readable name."""

    # data-testid="create-asset-btn"  →  Create Asset Btn
    m = re.search(r'data-testid=["\']([^"\']+)["\']', selector)
    if m:
        return _titleify(m.group(1))

    # aria-label="Upload file"
    m = re.search(r'aria-label=["\']([^"\']+)["\']', selector, re.I)
    if m:
        return m.group(1)

    # placeholder="Search assets"
    m = re.search(r'placeholder=["\']([^"\']+)["\']', selector, re.I)
    if m:
        return m.group(1)

    # name="campaign_name"
    m = re.search(r'\bname=["\']([^"\']+)["\']', selector)
    if m:
        return _titleify(m.group(1))

    # text=Upload  or  text="Upload file"
    m = re.match(r'^text=["\'"]?(.+?)["\'"]?$', selector.strip())
    if m:
        return m.group(1).strip('"\'')

    # #my-id
    m = re.search(r'#([\w-]+)', selector)
    if m:
        return _titleify(m.group(1))

    # .my-class  (first meaningful class)
    m = re.search(r'\.([\w-]{4,})', selector)
    if m:
        name = _titleify(m.group(1))
        suffix = _type_label(method)
        return f"{name} {suffix}" if suffix else name

    return f"Element ({selector[:40]})"


def name_from_locator(method: str, kwargs: dict) -> str:
    """Turn get_by_role / get_by_label / get_by_text kwargs into a readable name."""
    if "name" in kwargs:
        return str(kwargs["name"]).strip('"\'')
    if "text" in kwargs:
        return str(kwargs["text"]).strip('"\'')
    if "label" in kwargs:
        return str(kwargs["label"]).strip('"\'')
    if "placeholder" in kwargs:
        return str(kwargs["placeholder"]).strip('"\'')
    # get_by_role("button") → "Button"
    return _titleify(method.replace("get_by_", ""))


def _titleify(s: str) -> str:
    return re.sub(r'[-_]', ' ', s).title()


def _type_label(method: str) -> str:
    return {
        "click": "Button", "fill": "Input", "type": "Input",
        "select_option": "Dropdown", "check": "Checkbox",
        "uncheck": "Checkbox", "hover": "", "focus": "Field",
        "tap": "Button",
    }.get(method, "")


def element_type(method: str, locator_method: str = "") -> str:
    if locator_method:
        return {
            "get_by_role": "role-element",
            "get_by_label": "input",
            "get_by_placeholder": "input",
            "get_by_text": "text-element",
            "get_by_testid": "element",
            "locator": "element",
        }.get(locator_method, "element")
    return {
        "click": "button", "fill": "input", "type": "input",
        "select_option": "select", "check": "checkbox",
        "uncheck": "checkbox", "hover": "element",
        "focus": "input", "tap": "button",
    }.get(method, "element")


# ─────────────────────────────────────────────
# PLAYWRIGHT LOCATOR → CSS SELECTOR  (approx)
# We reconstruct a usable CSS selector from get_by_* calls so the
# extension's findElement() can actually query the DOM.
# ─────────────────────────────────────────────

def locator_to_selector(locator_method: str, args: list, kwargs: dict) -> str:
    """Convert get_by_* locator into a CSS-compatible selector string."""
    first = str(args[0]).strip('"\'') if args else ""

    if locator_method == "get_by_testid":
        return f'[data-testid="{first}"]'
    if locator_method == "get_by_label":
        label = kwargs.get("label", first).strip('"\'')
        return f'[aria-label="{label}"], label:has-text("{label}") + input'
    if locator_method == "get_by_placeholder":
        ph = kwargs.get("placeholder", first).strip('"\'')
        return f'[placeholder="{ph}"]'
    if locator_method == "get_by_text":
        text = kwargs.get("text", first).strip('"\'')
        return f'text="{text}"'
    if locator_method == "get_by_role":
        role = first
        name = str(kwargs.get("name", "")).strip('"\'')
        if name:
            return f'[role="{role}"][aria-label="{name}"], {role}:has-text("{name}")'
        return f'[role="{role}"]'
    if locator_method == "locator":
        return first  # already a CSS/Playwright selector
    return first


# ─────────────────────────────────────────────
# URL → CHAPTER NAME
# ─────────────────────────────────────────────

def url_to_chapter(url: str) -> str:
    try:
        segments = [s for s in urlparse(url).path.split("/") if s]
        if not segments:
            return "Home"
        # Drop UUIDs / numeric IDs
        segments = [s for s in segments if not re.match(r'^[\da-f-]{8,}$', s, re.I) and not s.isdigit()]
        return _titleify(segments[-1]) if segments else "Home"
    except Exception:
        return "General"


# ─────────────────────────────────────────────
# AST VISITOR
# ─────────────────────────────────────────────

ACTION_METHODS = {"click", "fill", "type", "select_option", "check",
                  "uncheck", "hover", "focus", "tap", "press"}

LOCATOR_METHODS = {"get_by_role", "get_by_label", "get_by_placeholder",
                   "get_by_text", "get_by_testid", "locator"}


def _const(node) -> str | None:
    """Return string value if node is a constant string, else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _kwargs_to_dict(keywords: list) -> dict:
    return {kw.keyword: _const(kw.value) or "" for kw in keywords
            if hasattr(kw, "keyword") and kw.keyword}


def _ast_kwargs(keywords) -> dict:
    result = {}
    for kw in keywords:
        if kw.arg and isinstance(kw.value, ast.Constant):
            result[kw.arg] = kw.value.value
    return result


class PlaywrightVisitor(ast.NodeVisitor):
    def __init__(self):
        self.chapters: dict = {}
        self._current_url: str = ""
        self._current_chapter: str = "General"

    # ── Helpers ──────────────────────────────

    def _ensure_chapter(self, chapter: str, url: str):
        if chapter not in self.chapters:
            self.chapters[chapter] = {"pageUrl": url, "elements": []}

    def _add_element(self, name: str, selector: str, el_type: str):
        ch = self._current_chapter
        self._ensure_chapter(ch, self._current_url)
        existing = [e["selector"] for e in self.chapters[ch]["elements"]]
        if selector and selector not in existing and not selector.startswith("internal:"):
            self.chapters[ch]["elements"].append({
                "id":       f"e_{uuid.uuid4().hex[:8]}",
                "name":     name,
                "selector": selector,
                "type":     el_type,
                "addedAt":  str(date.today()),
            })

    # ── Visitor ──────────────────────────────

    def visit_Expr(self, node):
        """Top-level expression statements — where all page.xxx() calls live."""
        self._process_call(node.value)
        self.generic_visit(node)

    def _process_call(self, node):
        if not isinstance(node, ast.Call):
            return

        func = node.func

        # ── page.goto("url") ──────────────────
        if isinstance(func, ast.Attribute) and func.attr == "goto":
            url = _const(node.args[0]) if node.args else None
            if url:
                self._current_url = url
                self._current_chapter = url_to_chapter(url)
                self._ensure_chapter(self._current_chapter, url)
            return

        # ── Chained locator calls, e.g.:
        #    page.get_by_role("button", name="Create").click()
        #    page.locator(".btn").fill("text")
        # The outer call is the action (.click/.fill); the inner is the locator.
        if isinstance(func, ast.Attribute) and func.attr in ACTION_METHODS:
            action = func.attr
            inner = func.value  # the locator expression

            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
                locator_method = inner.func.attr
                if locator_method in LOCATOR_METHODS:
                    l_args  = [_const(a) for a in inner.args if _const(a) is not None]
                    l_kw    = _ast_kwargs(inner.keywords)
                    sel     = locator_to_selector(locator_method, l_args, l_kw)
                    name    = name_from_locator(locator_method, {**l_kw, **dict(enumerate(l_args))})
                    el_type = element_type(action, locator_method)
                    self._add_element(name, sel, el_type)
                    return

            # ── page.click('[data-testid="btn"]') — old API ──
            sel = _const(node.args[0]) if node.args else None
            if sel:
                name    = name_from_selector(sel, action)
                el_type = element_type(action)
                self._add_element(name, sel, el_type)


# ─────────────────────────────────────────────
# PARSE FILE
# ─────────────────────────────────────────────

def parse_file(filepath: str) -> dict:
    source = Path(filepath).read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"[ERROR] Cannot parse {filepath}: {e}")
        sys.exit(1)

    visitor = PlaywrightVisitor()
    visitor.visit(tree)
    return visitor.chapters


# ─────────────────────────────────────────────
# BUILD / MERGE INDEX
# ─────────────────────────────────────────────

def build_index(chapters: dict) -> dict:
    return {
        "version":   1,
        "updatedAt": str(date.today()),
        "chapters":  chapters,
    }


def merge_into(existing_index: dict, new_chapters: dict) -> dict:
    """Merge new chapters into an existing index without overwriting."""
    ex = existing_index.get("chapters", {})
    for chapter, data in new_chapters.items():
        if chapter not in ex:
            ex[chapter] = data
        else:
            # Merge elements, skip duplicates by selector
            existing_sels = {e["selector"] for e in ex[chapter]["elements"]}
            for el in data["elements"]:
                if el["selector"] not in existing_sels:
                    ex[chapter]["elements"].append(el)
                    existing_sels.add(el["selector"])
    existing_index["chapters"]  = ex
    existing_index["updatedAt"] = str(date.today())
    return existing_index


# ─────────────────────────────────────────────
# PRETTY PRINT SUMMARY
# ─────────────────────────────────────────────

def print_summary(index: dict):
    chapters = index.get("chapters", {})
    total    = sum(len(ch["elements"]) for ch in chapters.values())
    print(f"\n{'─'*52}")
    print(f"  Hushly Index — {total} elements across {len(chapters)} chapters")
    print(f"{'─'*52}")
    for ch_name, data in chapters.items():
        els = data["elements"]
        print(f"\n  📂  {ch_name}  ({len(els)} elements)  [{data['pageUrl']}]")
        for el in els[:5]:
            t = el["type"].ljust(10)
            print(f"       {t}  {el['name']:<30}  {el['selector'][:50]}")
        if len(els) > 5:
            print(f"       … and {len(els) - 5} more")
    print(f"\n{'─'*52}\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    merge_mode = "--merge" in args
    if merge_mode:
        args = [a for a in args if a != "--merge"]

    if not args:
        print(__doc__)
        sys.exit(0)

    input_file  = args[0]
    output_file = args[1] if len(args) > 1 else "selector_map.json"

    if not Path(input_file).exists():
        print(f"[ERROR] File not found: {input_file}")
        sys.exit(1)

    print(f"Parsing {input_file} ...")
    new_chapters = parse_file(input_file)

    if not new_chapters:
        print("[WARN] No selectors found. Is this a valid Playwright codegen file?")
        sys.exit(1)

    # ── Merge into existing index or build fresh ──
    out_path = Path(output_file)
    if merge_mode and out_path.exists():
        print(f"Merging into existing {output_file} ...")
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        index = merge_into(existing, new_chapters)
    else:
        index = build_index(new_chapters)

    print_summary(index)
    out_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓  Saved → {output_file}\n")


if __name__ == "__main__":
    main()
