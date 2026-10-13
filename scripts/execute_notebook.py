"""Execute the notebook using this Python environment, without a global kernel install."""
import json
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
path = root / "notebook/titanic_analysis.ipynb"
with tempfile.TemporaryDirectory() as directory:
    kernel = Path(directory) / "kernels" / "titanic-local"
    kernel.mkdir(parents=True)
    (kernel / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Titanic local Python", "language": "python",
    }), encoding="utf-8")
    os.environ["JUPYTER_PATH"] = directory + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(notebook, timeout=180, kernel_name="titanic-local", resources={"metadata": {"path": str(root)}}).execute()
    # Keep a portable kernel name in the saved notebook.
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbformat.write(notebook, path)
print(f"Executed {path.name} successfully")
