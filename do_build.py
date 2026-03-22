#!/usr/bin/env python3
import subprocess
import sys
import os

os.chdir('/workspaces/QCI_AstroEntangle_Refiner')

# Write status
with open('build_status.txt', 'w') as f:
    f.write('BUILD STARTED\n')
    f.flush()
    
    # Run PyInstaller
    result = subprocess.run(
        [sys.executable, '-m', 'PyInstaller', 
         '--onefile', '--windowed', 
         '--name', 'QCI_AstroEntangle_Refiner',
         'QCI_AstroEntangle_Refiner.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    f.write(f'BUILD COMPLETED\nRETURN CODE: {result.returncode}\n')
    f.write('='*60 + '\n')
    f.write(result.stdout)
    f.flush()

# Check if executable was created
import os.path
exe_path = '/workspaces/QCI_AstroEntangle_Refiner/dist/QCI_AstroEntangle_Refiner'
if os.path.exists(exe_path):
    size = os.path.getsize(exe_path)
    with open('build_status.txt', 'a') as f:
        f.write(f'\n\nSUCCESS: Executable created at {exe_path} ({size/(1024*1024):.1f} MB)\n')
else:
    with open('build_status.txt', 'a') as f:
        f.write(f'\n\nFAILED: Executable not found at {exe_path}\n')

sys.exit(result.returncode)
