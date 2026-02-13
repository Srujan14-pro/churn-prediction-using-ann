import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
import os
import sys

def log(msg):
    print(msg)
    sys.stdout.flush()
    with open('verification_log.txt', 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

def check_files():
    required_files = [
        'churn_model.h5', 
        'scaler.pkl', 
        'label_encoder.pkl', 
        'one_hot_encoder.pkl'
    ]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        log(f"MISSING FILES: {missing_files}")
        return False
    log("All files present.")
    return True

def load_assets():
    try:
        log("Loading Model...")
        model = tf.keras.models.load_model('churn_model.h5')
        log("Model loaded successfully.")
        
        log("Loading Encoders...")
        with open('scaler.pkl', 'rb') as f:
            pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            pickle.load(f)
        with open('one_hot_encoder.pkl', 'rb') as f:
            pickle.load(f)
        log("Encoders loaded successfully.")
        return True
    except Exception as e:
        log(f"ERROR LOADING ASSETS: {e}")
        import traceback
        log(traceback.format_exc())
        return False

if __name__ == "__main__":
    if os.path.exists('verification_log.txt'):
        os.remove('verification_log.txt')
    
    if check_files():
        success = load_assets()
        if not success:
            sys.exit(1)
    else:
        sys.exit(1)
