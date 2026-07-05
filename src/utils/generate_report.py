import re
from pathlib import Path

def extract_placeholders(manuscript_path: Path):
    """Parses Manuscript.tex for placeholders."""
    placeholders = []
    if not manuscript_path.exists():
        print(f"[WARNING] Manuscript not found at {manuscript_path}")
        return placeholders
        
    with open(manuscript_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        matches_colorbox = re.finditer(r'\\colorbox\{yellow\}\{(.*?)\}', line)
        for m in matches_colorbox:
            placeholders.append({
                "type": "numeric",
                "line_num": i + 1,
                "content": m.group(1),
                "context": line.strip()
            })
            
        matches_hl = re.finditer(r'\\hl\{(.*?)\}', line)
        for m in matches_hl:
            placeholders.append({
                "type": "textual_decision",
                "line_num": i + 1,
                "content": m.group(1),
                "context": line.strip()
            })
            
    return placeholders

def load_produced_numbers(runs_dir: Path):
    """Loads all \\newcommand definitions from numbers_*.tex"""
    macros = {}
    for num_file in runs_dir.glob("numbers_*.tex"):
        with open(num_file, "r") as f:
            for line in f:
                # \newcommand{\macroName}{value}
                match = re.match(r'\\newcommand\{\\(.*?)\}\{(.*?)\}', line.strip())
                if match:
                    macros[match.group(1)] = match.group(2)
    return macros

def generate_report():
    project_root = Path(__file__).resolve().parent.parent.parent
    manuscript_path = project_root / "Doc" / "Manuscript.tex"
    runs_latest = project_root / "runs" / "latest"
    report_out = runs_latest / "REPORT.md"
    
    runs_latest.mkdir(parents=True, exist_ok=True)
    placeholders = extract_placeholders(manuscript_path)
    macros = load_produced_numbers(runs_latest)
    
    report_lines = []
    report_lines.append("# Master Report: PI-UIO CALA")
    report_lines.append("This report maps experimental results to the placeholders found in the manuscript.\n")
    
    report_lines.append("## Textual Decisions Required (\\hl)")
    for p in placeholders:
        if p["type"] == "textual_decision":
            report_lines.append(f"- **Line {p['line_num']}**: `{p['content']}`")
            report_lines.append(f"  *Context*: {p['context']}")
            # Text decisions are human resolved based on experiments
            report_lines.append(f"  *Decision*: REQUIRES HUMAN REVIEW based on experimental output.\n")
            
    report_lines.append("## Numeric Placeholders (\\colorbox{yellow})")
    for p in placeholders:
        if p["type"] == "numeric":
            target_macro = p['content'].strip('\\') # remove leading slash if any
            val = macros.get(target_macro, "UNAVAILABLE (Experiment must produce this macro)")
            
            report_lines.append(f"- **Line {p['line_num']}**: `{p['content']}`")
            report_lines.append(f"  *Context*: {p['context']}")
            report_lines.append(f"  *Value*: {val}\n")
            
    with open(report_out, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"REPORT.md generated at {report_out}")

if __name__ == "__main__":
    generate_report()
