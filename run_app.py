import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))
os.environ["PYTHONPATH"] = f"{str(PROJECT_ROOT)}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print("  [*] STARTING TICKET SUMMARIZATION DAAS & TESTING STUDIO")
    print("=" * 65)
    print("  -> UI Studio URL : http://localhost:8000")
    print("  -> Swagger API   : http://localhost:8000/docs")
    print("  -> Health Check  : http://localhost:8000/health")
    print("=" * 65 + "\n")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, app_dir=str(PROJECT_ROOT))
