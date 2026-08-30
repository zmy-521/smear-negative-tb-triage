from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_EXT = {".py", ".md", ".txt", ".json", ".toml", ".yml", ".yaml", ".csv"}
for p in ROOT.rglob("*"):
    if not p.is_file() or p.suffix.lower() not in TEXT_EXT:
        continue
    text = p.read_text(encoding="utf-8", errors="ignore")
    forbidden = ["C:" + "\\Users\\", "张" + "梦雨", "D:" + "\\SKILL备份"]
    hits = [x for x in forbidden if x in text]
    assert not hits, f"Private local path/name found in {p}: {hits}"
print("OK: no known private workstation paths/names found")
