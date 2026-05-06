# Kinetiq: Real-Time Exercise Form Detection

This project is a real-time AI-powered exercise form detection system. It uses a webcam to analyze a user's movements, provide immediate feedback on their form, and count repetitions for various exercises.

## System Workflow

The application is composed of a web-based frontend that communicates with a Python backend via WebSockets for real-time analysis.

```mermaid
graph TD
    subgraph Frontend (Browser)
        A[Camera Capture] -->|Video Frames| B(WebSocket Client)
        B -->|Sends Frames| C{Backend}
        D[Canvas Overlay] <--|Receives JSON| B
    end

    subgraph Backend (Python Server)
        C --> E[WebSocket Server]
        E -->|Frames| F(Pose Estimation)
        F -->|Keypoints| G(Angle Calculation)
        G -->|Angles| H(Form Evaluation)
        H -->|Feedback & Reps| E
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#f9f,stroke:#333,stroke-width:2px
    style F fill:#ccf,stroke:#333,stroke-width:2px
    style G fill:#ccf,stroke:#333,stroke-width:2px
    style H fill:#ccf,stroke:#333,stroke-width:2px
```

### 1. Frontend (User's Browser)
- **Camera Access**: The web application requests access to the user's webcam.
- **Frame Capture**: JavaScript captures frames from the video feed at a consistent rate (e.g., 30 FPS).
- **WebSocket Communication**: Each captured frame is sent to the backend over a persistent WebSocket connection.
- **Rendering**: The frontend receives analysis data (keypoints, feedback, rep count) back from the server as a lightweight JSON object. It then uses the HTML5 Canvas to draw the skeleton, joint angles, and feedback directly over the live video feed. This avoids sending heavy video data back from the server, significantly reducing latency.

### 2. Backend (Python & FastAPI)
- **WebSocket Server**: A FastAPI server listens for incoming WebSocket connections.
- **Frame Processing**: When a video frame arrives, it's decoded into an image format that the AI model can understand.
- **AI Pipeline**:
    1.  **Pose Estimation (`pose_extractor.py`)**: The powerful **YOLOv8-Pose** model analyzes the image to detect a person and extract the coordinates of their 17 key body joints (e.g., shoulders, elbows, knees).
    2.  **Angle Calculation (`angle_engine.py`)**: Using the keypoint coordinates, the system calculates the angles of critical joints for the selected exercise (e.g., the elbow angle for a bicep curl).
    3.  **Form Evaluation (`form_evaluator.py`)**: Pre-defined rules for each exercise are used to evaluate the user's form based on the calculated joint angles. For example, it checks if the user is achieving the correct range of motion. It also includes a **Rep Counter** that tracks valid repetitions.
- **Response**: The backend packages the results—keypoints, form feedback, and rep count—into a JSON object and sends it back to the frontend.

### 3. Real-time Pipeline
The producer-consumer pattern with `asyncio.Queue` is used to manage the flow of frames. The WebSocket endpoint acts as the **producer**, receiving frames from the client and placing them in a queue. A separate asynchronous task acts as the **consumer**, pulling frames from the queue to be processed by the AI pipeline. This architecture prevents the backend from getting overwhelmed and ensures a smooth, low-latency experience by processing frames concurrently.

---

## Project Structure

The project is organized into several directories:

- `src/`: Contains the core logic for pose extraction, angle calculation, form evaluation, and rendering.
- `models/`: Stores the pre-trained YOLOv8 pose model.
- `web/`: Includes the backend server and frontend code for the web interface.
- `training_pipeline/`: Contains scripts and modules for training custom models or calibrating the system.
- `app.py`: The main entry point to run the application.
