# 🏗️ SafeSite AI

> **Field-Deployable AI Safety Monitoring for Small-Scale Construction Sites**  
> Real-time Upper-Body PPE Compliance (Helmets & Reflective Vests) with Multilingual Voice Feedback & Kiosk Check-In.

---

## 📌 Overview
Over 85% of construction activity in developing regions like India is managed by small contractors and local builders with tight budgets and diverse, non-literate workforces. **SafeSite AI** is a low-resource computer vision system designed to run on standard laptop hardware using a single USB webcam or IP camera feed.

Instead of complex supervisory dashboards, SafeSite AI communicates directly with laborers using:
- 🔊 **Sequential Regional Voice Prompts:** Hindi ➔ Bhojpuri ➔ Maithili
- 🪧 **Visual Onboarding Pamphlet Mode:** High-contrast iconography for workers
- 📦 **Spare PPE Kiosk Integration:** Directs workers to Spare Bin A instead of turning them away, preventing lost workdays
- 📄 **Contractor Safety Audit Reports:** One-click printable daily compliance reports with software uptime logs

---

## 🚀 Tech Stack
- **Backend:** Python 3.10+, FastAPI, Uvicorn
- **Computer Vision:** OpenCV (`cv2`), YOLOv8 Architecture
- **Audio Synthesis:** `gTTS` (pre-generated MP3 assets for low-latency playback)
- **Frontend:** HTML5, CSS3 (Cyber-Glassmorphism UI), JavaScript (ES6)

---

## 📂 Project Structure
```
safesite-ai/
├── server.py              # FastAPI application server & MJPEG camera streamer
├── vision_engine.py       # Vision pipeline, 2s stabilization timer & scan locking
├── audio_engine.py        # Multilingual voice playlist engine (Hindi, Bhojpuri, Maithili)
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore patterns
└── static/
    ├── index.html         # Kiosk UI with visual pamphlet & contractor panel
    ├── css/
    │   └── style.css      # Cyber-Glassmorphism dark theme styling
    ├── js/
    │   └── app.js         # Frontend controller & audio sequence player
    └── audio/             # Pre-synthesized MP3 voice files
```

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Shivansh-20/safesite.git
cd safesite
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Server
```bash
python server.py
```

### 4. Open in Browser
Visit **`http://localhost:8000`** in your browser.

---

## 🛠️ Features
- **Upper-Body AI Detection:** High-speed scanning for Safety Helmets and Reflective Vests.
- **2-Second Hold Timer:** Stabilizes detection when a worker steps in front of the camera before locking decisions.
- **True Hardware Release on Halt:** Pauses the vision engine and powers off the webcam sensor.
- **Zero-Wasted-Day Flow:** Automated gate lock and `🔁 Re-Scan` button after worker equips spare PPE.

---

## ⌨️ Operator Keyboard Shortcuts (Manual Override & Visitor Controls)

The kiosk interface operates cleanly without on-screen buttons to prevent accidental or unauthorized worker tampering. System operators or supervisors can control gate states, clear non-worker visitors, or trigger manual overrides using discreet keyboard shortcuts:

| Key | Alternative | Action | Use Case |
| :---: | :---: | :--- | :--- |
| **`C`** | **`2`** | **Force Clear (Green Pass)** | **Site Visitors & Exempt Personnel:** Instantly clears gate check-in for visitors, inspectors, clients, or office staff who do not require construction PPE, or manually overrides the gate if a sensor check requires bypass. |
| **`M`** | **`1`** | **Force Missing (Red Flag)** | Flags missing PPE, unlocks visual guidance pamphlet, and triggers regional audio alert directing the person to Spare Bin A. |
| **`R`** | — | **Re-Scan Current Worker** | Unlocks the scanner to re-evaluate the worker after they put on spare PPE. |
| **`N`** | — | **Next Worker** | Increments the worker ID counter and resets the gate for the next person in line. |
| **`H`** | — | **Halt / Resume** | Pauses the kiosk system and releases webcam hardware (powers off camera sensor). Pressing again immediately resumes scanning. |

