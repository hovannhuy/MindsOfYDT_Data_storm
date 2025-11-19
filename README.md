# 🏌️‍♂️ Golf AI Coach — Golf Swing and Pose Analysis (DataStorm 2025)

Our project for participation in **DataStorm 2025** in the following topic:” Sports behaviours analysis”
A Computer Vision & Biomechanics system that analyzes a golf swing based on provided videos.

---

## 📜 Introduction

**Golf AI Coach** is a supportive AI system capable of:

- Pose estimation
- Biomechanics parameters calculation
- A feature dataset built for training ML
- Creating a standard swing profile for estimation

This project ultimately supports coaches and players to improve their technique based on subjective analysis.

---

## ✨ Features

### 📐 1. Biomechanical features extraction  
Using MediaPipe to extract **keypoints**, then calculate:

- Elbow angles (left/right)  
- Knee angles (left/right)  
- Shoulder inclination  
- Hip inclination  
- Body inclination/sway  
- Backswing – downswing’s amplitude

---

### ⚙️ 2. Video processing
-   **Input:** A video file(`.mp4`)with the swing taken.
-   **Process:**
    1. Automatically identify and extract 33 joints on the human body in every frame.
    2. Calculate crucial biomechanical features(angle, inclination).
    3. (Optional)Compare it with our swing profile made from the given data.
-   **Output:** A compiled video, visualizing the whole analysis process.
  
---

### 📊 3. Ideal Swing Profile  
Automatically calculate parameters from “good” swings as a subjective benchmark.

---

### 🎥 4. Analysis video 4-in-1  
Dashboard includes:

1. Original video
2. Skeletal frame video 
3. Real-time spreadsheet    

### 🚀 Demo Product (Round 1)
Preliminary results are shown in the following video on GitHub:

![simple_analyzed_Untitled00014096 (2)](https://github.com/user-attachments/assets/75fc495f-24c2-4796-89fc-be6d8ded0452)


### 🛠️ 5. Structure
```css
Data_storm_2025/
├── data/
│   └── Untitled00014096.mp4
│ 
├── notebooks/
│   └── cal_pose.ipynb
│ 
├── slide/
│   ├── Mindsofydt_data_storm_2025.pdf
│   └── Mindsofydt_data_storm_2025.pptx
│ 
├── src/
│   ├── pose_extractor.py      → Pose & keypoints extracting
│   ├── build_feature.py       → Creating features dataset
│   ├── swing_profile.py       → Creating ideal swing profile
│   └── video_analyzer.py      → Output video
│
├── video_results/
│   ├── results.mp4                      → End to end create output video                
│   └── simple_cal_Untitled00014096.mp4  → Output_video
├── requirements.txt
└── README.md
```
### 🔧 Installments & Trials
1️⃣ Clone repo from github:  
```bash
git clone https://github.com/hovannhuy/Data_storm_2025.git
cd Data_storm_2025
```
2️⃣ Use Google Colab and upload notebook cal_pose.ipynb
3️⃣  Connect real-time running, upload the desired video onto Google Colab, and sequentially run all the cells.
Results are shown in:
```bash
results/
```
### 🗺️ Development plan:
✔️ Round 1

- Skeletal frame extraction

- Angle calculation

- Analysis video

🔄 Round 2

- Classifying Good vs Bad Swing Machine Learning

- Swing Score

🏆 Hackathon

- Web app and Streamlit

- AI Coach for real-time suggestions
