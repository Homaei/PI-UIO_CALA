import re
from pathlib import Path

def extract_placeholders(manuscript_path: Path):
    """
    Parses Manuscript.tex for \colorbox{yellow}{...} and \hl{...}
    without modifying it.
    Returns a dictionary of found placeholders mapped to their lines.
    """
    placeholders = []
    
    if not manuscript_path.exists():
        print(f"[WARNING] Manuscript not found at {manuscript_path}")
        return placeholders
        
    with open(manuscript_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        # Find \colorbox{yellow}{...}
        matches_colorbox = re.finditer(r'\\colorbox\{yellow\}\{(.*?)\}', line)
        for m in matches_colorbox:
            placeholders.append({
                "type": "numeric",
                "line_num": i + 1,
                "content": m.group(1),
                "context": line.strip()
            })
            
        # Find \hl{...}
        matches_hl = re.finditer(r'\\hl\{(.*?)\}', line)
        for m in matches_hl:
            placeholders.append({
                "type": "textual_decision",
                "line_num": i + 1,
                "content": m.group(1),
                "context": line.strip()
            })
            
    return placeholders

def generate_report():
    project_root = Path(__file__).resolve().parent.parent.parent
    manuscript_path = project_root / "Doc" / "Manuscript.tex"
    runs_latest = project_root / "runs" / "latest"
    report_out = runs_latest / "REPORT.md"
    
    runs_latest.mkdir(parents=True, exist_ok=True)
    
    placeholders = extract_placeholders(manuscript_path)
    
    # In a full run, we would read numbers_E*.tex and tab_*.tex from runs_latest
    # and map them to the placeholders. For the generator script, we just aggregate 
    # what we find and flag what needs to be filled.
    
    report_lines = []
    report_lines.append("# Master Report: PI-UIO CALA")
    report_lines.append("This report maps experimental results to the placeholders found in the manuscript.\n")
    
    report_lines.append("## Textual Decisions Required (\\hl)")
    for p in placeholders:
        if p["type"] == "textual_decision":
            report_lines.append(f"- **Line {p['line_num']}**: `{p['content']}`")
            report_lines.append(f"  *Context*: {p['context']}")
            report_lines.append(f"  *Decision*: UNAVAILABLE (Pending manual review based on experiment outcomes)\n")
            
    report_lines.append("## Numeric Placeholders (\\colorbox{yellow})")
    for p in placeholders:
        if p["type"] == "numeric":
            report_lines.append(f"- **Line {p['line_num']}**: Target format `{p['content']}`")
            report_lines.append(f"  *Context*: {p['context']}")
            report_lines.append(f"  *Value*: UNAVAILABLE (Experiment scripts must generate tab_*.tex to fill this)\n")
            
    # Write to runs/latest/REPORT.md
    with open(report_out, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"REPORT.md generated at {report_out}")

if __name__ == "__main__":
    generate_report()
