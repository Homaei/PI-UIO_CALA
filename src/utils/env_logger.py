import platform
import psutil
import torch
import sys

from pathlib import Path

def log_environment(output_path: Path):
    """Logs system hardware and package versions for reproducibility."""
    lines = []
    
    # OS & Python
    lines.append(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    lines.append(f"Python: {sys.version.split(' ')[0]}")
    
    # CPU
    cpu_model = platform.processor() or "Unknown CPU"
    logical_cores = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)
    lines.append(f"CPU: {cpu_model} ({physical_cores} Physical Cores, {logical_cores} Logical Threads)")
    
    # RAM
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    lines.append(f"RAM: {ram_gb} GB")
    
    # GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        lines.append(f"GPU: {gpu_name} ({gpu_vram} GB VRAM) - CUDA {torch.version.cuda}")
    else:
        lines.append("GPU: Not Available / CPU only")
        
    lines.append("-" * 40)
    lines.append("Packages:")
    
    required_packages = ["numpy", "scipy", "pandas", "wntr", "cvxpy", "mosek", "scikit-learn", "torch", "matplotlib", "tqdm", "joblib"]
    
    import importlib.metadata
    
    for pkg in required_packages:
        try:
            version = importlib.metadata.version(pkg)
            lines.append(f"{pkg}=={version}")
        except importlib.metadata.PackageNotFoundError:
            lines.append(f"{pkg}==NOT FOUND")
            
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    out_file = Path("environment.txt")
    log_environment(out_file)
    print(f"Environment logged to {out_file.absolute()}")
