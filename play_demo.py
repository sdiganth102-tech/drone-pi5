#!/usr/bin/env python3
"""
Interactive Video Player for Drone Flood Rescue System.
Plays the generated mission video with HUD overlays on top of all windows.
"""

import os
import sys
import cv2

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    demo_video = os.path.join(script_dir, "output_logs", "flood_rescue_demo.mp4")

    # If demo video doesn't exist, fallback to simulated output or run live
    if not os.path.exists(demo_video):
        alt_video = os.path.join(script_dir, "output_logs", "simulated_people_output.mp4")
        if os.path.exists(alt_video):
            demo_video = alt_video
        else:
            print("[INFO] Generating fresh demo video... Please wait 10 seconds.")
            from simulate_flood_mission import run_mission_simulation
            run_mission_simulation(
                video_path=os.path.join(script_dir, "test_videos", "people-detection.mp4"),
                output_video_path=demo_video,
                drone_alt_m=25.0,
                camera_tilt_deg=45.0,
                max_frames=120,
                conf_threshold=0.80,
                show_window=False
            )

    print("=" * 60)
    print("   DRONE FLOOD RESCUE - LIVE VIDEO DEMO PLAYER")
    print("=" * 60)
    print(f"Playing Video: {demo_video}")
    print(">> Controls: Press 'q' or 'ESC' to close, 'SPACE' to pause/resume <<")
    print("=" * 60)

    cap = cv2.VideoCapture(demo_video)
    if not cap.isOpened():
        print(f"Error opening video: {demo_video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    delay_ms = max(10, int(1000.0 / fps))

    window_name = "Drone Flood Rescue - Pythagorean Geolocation Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1020, 680)

    # Bring window to FRONT and keep on top
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    is_paused = False

    while True:
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                # Seamless loop
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            cv2.imshow(window_name, frame)

        key = cv2.waitKey(delay_ms if not is_paused else 50) & 0xFF
        if key == ord('q') or key == 27:
            print("Demo closed.")
            break
        elif key == ord(' '):
            is_paused = not is_paused
            print("Paused" if is_paused else "Playing")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
