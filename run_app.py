import sys
import os
from pathlib import Path

# Automatically ensure inner project folder is on python path
PROJECT_ROOT = Path(__file__).resolve().parent
INNER_DIR = PROJECT_ROOT / "Tiket summarization DaaS"

target_dir = INNER_DIR if INNER_DIR.exists() else PROJECT_ROOT
sys.path.insert(0, str(target_dir))
os.chdir(str(target_dir))
os.environ["PYTHONPATH"] = f"{str(target_dir)}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print("  [*] STARTING TICKET SUMMARIZATION DAAS & TESTING STUDIO")
    print("=" * 65)
    print("  -> UI Studio URL : http://localhost:8000")
    print("  -> Swagger API   : http://localhost:8000/docs")
    print("  -> Health Check  : http://localhost:8000/health")
    print("=" * 65 + "\n")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, app_dir=str(target_dir))
