## Recognition and Tracking using AI Drones during Hajj

Autonomous drone surveillance system developed for the **University of Hafr Al Batin**. This project utilizes **YOLOv11** to provide real-time monitoring, safety analysis, 
and crowd management during the Hajj pilgrimage.

## Project Motivation :
Hajj is one of the world's largest human gatherings. Traditional fixed surveillance often suffers from blind spots. Our solution leverages:
**Mobility:** Drones cover areas CCTV cannot reach[cite: 1].
**Intelligence:** AI detects emergencies like falls or overcrowding in real-time[cite: 1].
**Rapid Response:** Provides GPS-linked alerts to emergency services[cite: 1].

##Features :
**Real-time Detection:** High-speed person and object tracking.
**Optimized Inference:** Powered by **ONNX Runtime** for smooth performance on standard laptops (~30 FPS).
**Clean UI:** Minimized overlapping boxes using Non-Maximum Suppression (NMS).

## Tech Stack:
* **Language:** Python 3.10+[cite: 4]
* **Model:** YOLOv11n (Exported to ONNX)
* **Libraries:** OpenCV, NumPy, ONNX Runtime[cite: 4]

## Project Structure
```text
Hajj-AI-Drone/
├── models/              # Contains yolo11n.onnx[cite: 5]
├── src/                 # Main detection.py script[cite: 5]
├── requirements.txt     # List of dependencies[cite: 4]
└── README.md            # Project documentation[cite: 5]

Installation & Usage 
1- Clone the repo:
bash
git clone [https://github.com/marahibalsowayih/Hajj-AI-Drone.git](https://github.com/marahibalsowayih/Hajj-AI-Drone.git)

2- Install dependencies:
bash
pip install -r requirements.txt

3- Run the system:
Bash
python src/detection.py


Results:
Accuracy: 91.2% detection rate.
Latency: < 30ms per frame.
Stability: Handles dense crowds with integrated NMS filtering.

Acknowledgments
Developed at the University of Hafr Al Batin,
 College of Computer Science and Engineering. Special thanks to our supervisors for their guidance.
