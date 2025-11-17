# run_detector.py

import sounddevice as sd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pyautogui
import time
import csv

# --- 1. SETUP PARAMETERS ---

# How long to wait (in seconds) after a trigger before listening again
# This prevents it from triggering 100 times for one long "creeeak"
COOLDOWN_SECONDS = 520
last_trigger_time = 0

# This is the "confidence" threshold.
# How sure does the model need to be? (0.0 to 1.0)
# You will NEED to adjust this!
CONFIDENCE_THRESHOLD = 0.1

# This is the audio "chunk" size.
# 8192 samples is about 0.5 seconds.
# Smaller = faster reaction, but more CPU.
BLOCKSIZE = 8192 

# YAMNet model requires audio at 16000 Hz
SAMPLERATE = 16000

# --- 2. LOAD THE "BRAIN" (YAMNet ML Model) ---
print("Loading YAMNet model... (This may take a moment the first time)")
model = hub.load('https://tfhub.dev/google/yamnet/1')
print("Model loaded.")

# This function loads the list of all 521 sounds YAMNet knows
def get_yamnet_class_names(model):
    class_map_path = model.class_map_path().numpy().decode('utf-8')
    class_names = []
    with tf.io.gfile.GFile(class_map_path) as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            class_names.append(row[2])
    return class_names

class_names = get_yamnet_class_names(model)
print(f"YAMNet is listening for {len(class_names)} different sounds.")

# --- 3. IMPORTANT: FIND YOUR SOUND ---
# You need to find the name of your sound in YAMNet's brain.
# It's probably "Door", "Knock", or "Creak".
# We will check for ALL of them.
#
# Run this script once, then make your door sound. See what YAMNet
# prints as the "Top guess". Update this list!
SOUNDS_TO_DETECT = ["Door", "Knock", "Sliding door", "Squeak", "Bouncing", "Cough", "Glass", "Chink, clink", "Cutlery, silverware"]


# --- 4. THE "CALLBACK" (The function that runs on every audio chunk) ---

# This function is the heart of the script.
# 'indata' is the raw audio data from your mic.
def audio_callback(indata, frames, time_info, status):
    global last_trigger_time # Use the global 'last_trigger_time'
    
    if status:
        print(status)
        return

    # Check if we are in the "cooldown" period
    if (time.time() - last_trigger_time) < COOLDOWN_SECONDS:
        return

    # Convert audio to the format YAMNet expects
    # It needs floating-point numbers and a single channel (mono)
    mono_audio = np.mean(indata, axis=1, dtype=np.float32)

    # --- 5. THE "THINK" PART ---
    # Feed the audio chunk to the model and get the scores
    scores, embeddings, spectrogram = model(mono_audio)
    
    # 'scores' is a list of confidences for all 521 sounds.
    # We'll just take the average score for this chunk.
    scores = np.mean(scores, axis=0)

    # Find the top 3 guesses from the model
    top_3_indices = np.argsort(scores)[-3:]
    
    print_log = "Top guesses: "
    detected = False
    
    for i in top_3_indices:
        sound_name = class_names[i]
        confidence = scores[i]
        
        # Add to our log so we can see what's happening
        print_log += f"  {sound_name} ({confidence:.2f})  "
        
        # --- 6. THE "ACT" PART ---
        if sound_name in SOUNDS_TO_DETECT and confidence > CONFIDENCE_THRESHOLD:
            detected = True
            
    # Print what the script is hearing (for debugging)
    print(print_log)

    if detected:
        print(f"\n*** DETECTED! Confidence > {CONFIDENCE_THRESHOLD}. Triggering action! ***\n")
        
        # This is the command to switch windows
        pyautogui.hotkey('alt', 'tab')
        
        # Start the cooldown
        last_trigger_time = time.time()


# --- 7. START THE LISTENER ---
print("\nStarting audio listener...")
print("Listening for:", SOUNDS_TO_DETECT)
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
print("Press Ctrl+C in this terminal to stop the script.")

try:
# ... (around line 112)
    with sd.InputStream(
        channels=2, 
        samplerate=SAMPLERATE,
        blocksize=BLOCKSIZE,
        callback=audio_callback
    ):
        print("\nStream is active. Listening...")
        while True:
            time.sleep(10) # This keeps the script alive
            
except KeyboardInterrupt:
#...
    print("\nScript stopped by user.")
except Exception as e:
    print(f"An error occurred: {e}")
    print("Do you have an external mic plugged in?")