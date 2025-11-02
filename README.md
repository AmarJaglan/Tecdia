# Jumbled Frames Reconstruction Challenge

### Repository: `AmarJaglan/Tecdia`
### Challenge Objective: Reconstruct a jumbled 10-second, 30 FPS video to restore its original sequential order.

---

## Project Overview

This solution utilizes a robust **computer vision and algorithmic approach** to solve the frame ordering problem. The method is designed for accuracy in distinguishing subtle visual continuity while prioritizing execution efficiency to perform well on the benchmark environment.

### Key Components:

| Feature | Metric/Algorithm | Purpose |
| :--- | :--- | :--- |
| **Dissimilarity Metric** | **L2 Norm (Euclidean Distance)** | Measures the raw pixel distance between two frames to determine visual closeness. |
| **Optimization** | **Downsampling to 64x64 Grayscale** | Aggressively reduces the data size for the $\mathcal{O}(N^2)$ comparison stage, significantly reducing runtime. |
| **Ordering Algorithm** | **Greedy Shortest-Path Search** | Builds the final sequence by iteratively selecting the *least distant* (most similar) unvisited frame. |
| **Output Enhancement** | **Final Frame Fade-out** | Applies a smooth, alpha-blended transition to the last 10 frames to ensure a clean, seamless video loop or end. |

---

##  Requirements & Setup

This project requires Python 3.8+ and the following libraries.

### Option 1: Install via `requirements.txt` (Recommended)

Navigate to the project root (`Tecdia`) and install the dependencies within a virtual environment.

```bash
# 1. Create a virtual environment
python3 -m venv .venv

# 2. Activate the environment (macOS/Linux)
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

```
## Manual Installation

```bash
pip install opencv-python scikit-image tqdm numpy
```
## How to Run the Solution

### 1. Place Video  
Ensure your jumbled video file (`jumbled_video.mp4`) is located in the **root directory** of the project, alongside the `reconstruct_alternative.py` file.

### 2. Execute Script  
Run the reconstruction script using the Python 3 interpreter.

```bash
python reconstruct_alternative.py
```
## Outputs Generated

| File / Folder | Description |
| :--- | :--- |
| `frames/` | Folder containing all extracted raw frames from the input video. |
| `ordered_frames_alt/` | Folder containing frames arranged in the final predicted order. |
| `reconstructed_video_alt.mp4` | The final unjumbled video output (**Primary Deliverable**). |
| `execution_time_log_alt.txt` | Text file logging the total runtime of the algorithm (**Deliverable 4**). |

---

## Algorithm Explanation and Thought Process

### 1. Principle: Minimizing Pixel Distance  
The foundation of this solution lies in the principle that **two consecutive frames in a continuous video sequence exhibit minimal visual change** in their pixel values.  
Our algorithm’s main objective is to find the **unique sequence** through the jumbled frames that **minimizes the cumulative Euclidean distance** between adjacent frame pairs.

---

### 2. Dissimilarity Metric: L2 Norm (Euclidean Distance)  
We chose the **L2 Norm** due to its computational efficiency and straightforward pixel-level comparison — unlike perceptual metrics such as SSIM.

**Process:**  
- Each frame, after preprocessing, is represented as a single vector in a high-dimensional space.  
- The **L2 Norm** computes the straight-line distance between any two such vectors.

**Interpretation:**  
A smaller L2 Norm value implies the frames are **more visually continuous**, making them **more likely to be consecutive** in the original sequence.

---

### 3. Efficiency Optimization: Aggressive Dimensionality Reduction  
To overcome the **O(N²)** computational complexity of pairwise frame comparisons, the algorithm performs aggressive preprocessing to reduce dimensionality:

- Frames are converted to **grayscale arrays**.  
- Frames are **downsampled to 64×64 pixels**.  

This significantly reduces the vector size for L2 Norm calculations, **eliminating the main performance bottleneck** while maintaining essential temporal details.

---

### 4. Ordering Algorithm: Greedy Shortest-Path Search  
The final sequence reconstruction leverages a **Greedy Nearest-Neighbor** search applied to the **Dissimilarity Matrix** derived from L2 Norm values.

**Steps:**

1. **Start Node Selection:**  
   Identify the frame with the **highest average L2 distance** from all other frames.  
   This statistical outlier is assumed to be the **starting frame** of the original sequence.

2. **Path Extension:**  
   From the starting frame, **greedily select the unvisited frame** that has the **minimum L2 distance** (most visually similar) to the current frame.  
   This ensures each transition represents the **smallest possible visual jump**.

3. **Video Reconstruction:**  
   - Frames are reordered and written into a final `.mp4` video at **30 FPS**.  
   - A **smooth fade-out** effect is applied to the final 10 frames to deliver a **clean, cinematic ending**.

---

## Submission Details

- **Author:** Amar Jaglan  
- **Repository:** **  
- **Reconstructed Video Link:** *https://drive.google.com/file/d/1KttS1l9o3LJOyNUZinuwv0LG3BuKEcm0/view?usp=sharing*  


