from rag.parsing import chunk_filing, html_to_text, split_sections
from tests.conftest import FILING_HTML


def test_tables_become_pipe_separated_rows():
    text = html_to_text(b"<table><tr><td>Revenue</td><td></td><td>$5</td></tr></table>")
    assert "Revenue | $5" in text


def test_hidden_xbrl_and_scripts_are_dropped():
    html = b"<html><script>var x=1</script><ix:header>hidden</ix:header><p>Visible</p></html>"
    text = html_to_text(html)
    assert "Visible" in text
    assert "hidden" not in text and "var x" not in text


def test_split_sections_by_item_headings():
    text = "Cover page\nItem 1. Business\nWe sell skates.\nITEM 1A: Risk Factors\nSupplier risk."
    names = [name for name, _ in split_sections(text)]
    assert names == ["Cover / Other", "Item 1: Business", "Item 1A: Risk Factors"]


def test_long_line_starting_with_item_is_not_a_heading():
    text = "Item 1. Business\n" + "Item 7 is discussed below " + "and more words " * 20
    assert [name for name, _ in split_sections(text)] == ["Item 1: Business"]


def test_unknown_item_numbers_are_ignored():
    names = [name for name, _ in split_sections("Item 1. Business\nItem 99. Nonsense\nText")]
    assert names == ["Item 1: Business"]


def test_chunk_filing_tags_each_chunk_with_its_section():
    chunks = chunk_filing(FILING_HTML)
    sections = {c.section for c in chunks}
    assert {"Item 1: Business", "Item 1A: Risk Factors"} <= sections
    risk = next(c for c in chunks if c.section == "Item 1A: Risk Factors")
    assert "single supplier" in risk.text
    assert all(len(c.text) > 100 for c in chunks)
