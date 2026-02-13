import sys
import os

def log(msg):
    with open('debug_log.txt', 'a') as f:
        f.write(msg + '\n')
    print(msg)

if os.path.exists('debug_log.txt'):
    os.remove('debug_log.txt')

log("Starting debug_imports.py")
log(f"Python version: {sys.version}")

try:
    log("Importing numpy...")
    import numpy
    log(f"numpy version: {numpy.__version__}")
except Exception as e:
    log(f"Failed to import numpy: {e}")

try:
    log("Importing pandas...")
    import pandas
    log(f"pandas version: {pandas.__version__}")
except Exception as e:
    log(f"Failed to import pandas: {e}")

try:
    log("Importing pickle...")
    import pickle
    log("pickle imported")
except Exception as e:
    log(f"Failed to import pickle: {e}")

try:
    log("Importing sklearn...")
    import sklearn
    log(f"sklearn version: {sklearn.__version__}")
except Exception as e:
    log(f"Failed to import sklearn: {e}")

try:
    log("Importing tensorflow...")
    import tensorflow
    log(f"tensorflow version: {tensorflow.__version__}")
except Exception as e:
    log(f"Failed to import tensorflow: {e}")

try:
    log("Importing streamlit...")
    import streamlit
    log(f"streamlit version: {streamlit.__version__}")
except Exception as e:
    log(f"Failed to import streamlit: {e}")

log("Finished imports.")
