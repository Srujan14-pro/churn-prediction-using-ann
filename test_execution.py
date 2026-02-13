
import os
import sys

with open('execution_test.txt', 'w') as f:
    f.write('Python is executing.\n')
    f.write(f'Executable: {sys.executable}\n')
    f.write(f'CWD: {os.getcwd()}\n')
