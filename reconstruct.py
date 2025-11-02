import cv2
import numpy as np
import os
import shutil
import time
from tqdm import tqdm

# --- Configuration ---
INPUT_VIDEO = "jumbled_video.mp4"
FRAMES_DIR = "frames"
ORDERED_DIR = "ordered_frames_alt"
OUTPUT_VIDEO = "reconstructed_video_alt.mp4"
LOG_FILE = "execution_time_log_alt.txt"
TARGET_FPS = 30
FADE_FRAMES = 10 

# --- Utility Functions ---
def read_and_preprocess_frame(path):
    """Reads a frame, converts to grayscale, and resizes for comparison."""
    img = cv2.imread(path)
    if img is None:
        return None
    # Convert to grayscale for consistent 2D array comparison
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Smaller size (e.g., 64x64) for faster, consistent distance calculation
    img = cv2.resize(img, (64, 64)) 
    return img.flatten().astype(np.float64) # Flatten to a vector for distance calculation

def log_execution_time(start_time, log_file):
    """Calculates and logs the total execution time."""
    end_time = time.time()
    total_time = end_time - start_time
    with open(log_file, 'w') as f:
        f.write(f"Total Execution Time: {total_time:.2f} seconds\n")
    print(f"Total execution time logged to '{log_file}'.")
    return total_time

def calculate_l2_norm(vec1, vec2):
    """Calculates the Euclidean distance (L2 Norm) between two frame vectors."""
    return np.linalg.norm(vec1 - vec2)

# --- Main Reconstruction Logic ---
def reconstruct_video_alternative():
    start_time = time.time()
    
    print("--- Jumbled Frames Reconstruction Challenge (L2 Norm Approach) ---")
    
    # 1. Setup and Cleanup
    if os.path.exists(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR)
    if os.path.exists(ORDERED_DIR):
        shutil.rmtree(ORDERED_DIR)
        
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(ORDERED_DIR, exist_ok=True)

    # --- STEP 1: Extract frames from jumbled video ---
    print("\n[1/4] Extracting frames...")
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Error: Could not open video file {INPUT_VIDEO}.")
        return

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{FRAMES_DIR}/frame_{count:04d}.jpg", frame)
        count += 1
    cap.release()
    n = count
    print(f"Extracted {n} frames to '{FRAMES_DIR}' folder.")

    # --- STEP 2: Compute Dissimilarity Matrix (L2 Norm) ---
    print("\n[2/4] Computing frame L2 Norms (Dissimilarity Matrix)...")
    frame_files = sorted(os.listdir(FRAMES_DIR))
    frame_files = [f for f in frame_files if f.endswith('.jpg')]
    
    # Preprocess all frames into vectors
    frames_vector = [read_and_preprocess_frame(os.path.join(FRAMES_DIR, f)) for f in frame_files]
    n = len(frames_vector)

    # Dissimilarity Matrix: smaller value = better match
    dissimilarity_matrix = np.zeros((n, n)) 
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            if frames_vector[i] is not None and frames_vector[j] is not None:
                # Calculate the L2 Norm (Euclidean Distance)
                distance = calculate_l2_norm(frames_vector[i], frames_vector[j])
                dissimilarity_matrix[i, j] = dissimilarity_matrix[j, i] = distance

    # --- STEP 3: Determine frame order (Greedy Shortest-Path) ---
    print("\n[3/4] Determining correct frame order...")

    visited = [False] * n
    order = []

    # Strategy: Start with the frame that has the HIGHEST average distance to all others.
    # This frame is the statistical outlier, likely the distinct first or last frame.
    average_distance = dissimilarity_matrix.mean(axis=1)
    start_frame_idx = np.argmax(average_distance) 
    
    order.append(start_frame_idx)
    visited[start_frame_idx] = True
    
    # Greedily find the next best frame (the one with the minimum distance)
    for _ in tqdm(range(n - 1)):
        last_idx = order[-1]
        
        # Look for the unvisited frame with the MINIMUM distance to the last frame
        # Use a large number for visited frames so they are never chosen by argmin
        distances = dissimilarity_matrix[last_idx].copy()
        distances[visited] = np.inf # Set distances of visited frames to infinity
        
        next_idx = np.argmin(distances)
        
        if not visited[next_idx]:
             visited[next_idx] = True
             order.append(next_idx)
        else:
            break 

    print("Frame order determined.")

    # --- STEP 4: Rebuild ordered video and apply smooth fade ---
    print("\n[4/4] Rebuilding ordered video...")

    frame_files.sort() 
    for idx, original_frame_index in enumerate(order):
        src = os.path.join(FRAMES_DIR, frame_files[original_frame_index])
        dst = os.path.join(ORDERED_DIR, f"frame_{idx:04d}.jpg")
        shutil.copy(src, dst)

    sample_frame_path = os.path.join(ORDERED_DIR, "frame_0000.jpg")
    sample_frame = cv2.imread(sample_frame_path)
    if sample_frame is None:
        print("Error: Could not read sample frame for video creation.")
        return

    h, w, _ = sample_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, TARGET_FPS, (w, h))

    for i in tqdm(range(len(order))):
        frame = cv2.imread(os.path.join(ORDERED_DIR, f"frame_{i:04d}.jpg"))

        # Apply smooth fade-out
        if i >= len(order) - FADE_FRAMES:
            alpha = (len(order) - i) / FADE_FRAMES 
            frame = (frame * alpha).astype(np.uint8)

        out.write(frame)

    out.release()
    
    total_time = log_execution_time(start_time, LOG_FILE)
    print(f"\nReconstructed video saved as '{OUTPUT_VIDEO}'")
    print("All steps completed successfully!")
    print(f"Total time taken: {total_time:.2f} seconds.")

if __name__ == "__main__":
    reconstruct_video_alternative()