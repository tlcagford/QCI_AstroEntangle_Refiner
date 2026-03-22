import subprocess
import sys

result = subprocess.run(['/usr/bin/python3', '/workspaces/QCI_AstroEntangle_Refiner/build.py'])
sys.exit(result.returncode)
