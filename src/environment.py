import os
import sys
import subprocess

def setup_environment():
    repo_dir = "CLT_BaseModel"
    if not os.path.exists(repo_dir):
        print(f"Cloning CLT_BaseModel...")
        subprocess.run(["git", "clone", "https://github.com/LP-relaxation/CLT_BaseModel.git"], check=True)
    sys.path.append(os.path.abspath(repo_dir))
