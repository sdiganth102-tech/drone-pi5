#!/usr/bin/env python3
"""
Drone Flood Rescue: Mission Control Graphical User Interface (GUI).

An interactive desktop control station for demonstrating real-time human detection,
Pythagorean distance calculation, and GPS geolocalization on drone video feeds.
"""

import os
import sys
import time
import math
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# Project modules
from core.detector import DroneHumanDetector
from core.pythagoras_geolocator import PythagorasGeolocator
from core.telemetry import MockTelemetryProvider
from core.rescue_logger import RescueLogger
from utils.hud_visualizer import DroneHudVisualizer

class DroneRescueGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Drone Flood Rescue — Mission Control & Pythagorean Geolocation")
        self.root.geometry("1280x820")
        self.root.minsize(1050, 700)
        self.root.configure(bg="#1a1c23")

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.script_dir, "models")
        self.videos_dir = os.path.join(self.script_dir, "test_videos")
        self.logs_dir = os.path.join(self.script_dir, "output_logs")
        os.makedirs(self.logs_dir, exist_ok=True)

        # State variables
        self.is_running = False
        self.is_paused = False
        self.cap = None
        self.detector = None
        self.geolocator = None
        self.telemetry = None
        self.rescue_logger = None
        self.hud = None
        self.worker_thread = None

        # Config variables
        self.var_video_choice = tk.StringVar()
        self.var_custom_path = tk.StringVar(value="")
        self.var_alt = tk.DoubleVar(value=25.0)
        self.var_tilt = tk.DoubleVar(value=45.0)
        self.var_conf = tk.DoubleVar(value=0.80)
        self.var_lat = tk.DoubleVar(value=12.971599)
        self.var_lon = tk.DoubleVar(value=77.594566)
        self.var_heading = tk.DoubleVar(value=45.0)

        # Detection telemetry readouts
        self.readout_status = tk.StringVar(value="STANDBY")
        self.readout_count = tk.StringVar(value="0")
        self.readout_target_gps = tk.StringVar(value="-- , --")
        self.readout_ground_dist = tk.StringVar(value="-- m")
        self.readout_slant_dist = tk.StringVar(value="-- m")
        self.readout_bearing = tk.StringVar(value="-- °")
        self.readout_conf = tk.StringVar(value="-- %")

        self._build_style()
        self._build_ui()
        self._populate_video_list()

    def _build_style(self):
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.style.configure(".", background="#1a1c23", foreground="#ffffff", font=("Segoe UI", 10))
        self.style.configure("TFrame", background="#1a1c23")
        self.style.configure("Card.TFrame", background="#22252e", relief="flat")
        self.style.configure("TLabel", background="#1a1c23", foreground="#ffffff")
        self.style.configure("Card.TLabel", background="#22252e", foreground="#ffffff")
        self.style.configure("Header.TLabel", font=("Segoe UI", 13, "bold"), foreground="#00d2ff")
        self.style.configure("StatTitle.TLabel", font=("Segoe UI", 9), foreground="#94a3b8", background="#22252e")
        self.style.configure("StatVal.TLabel", font=("Segoe UI", 13, "bold"), foreground="#38bdf8", background="#22252e")
        self.style.configure("AlertVal.TLabel", font=("Segoe UI", 13, "bold"), foreground="#22c55e", background="#22252e")

    def _build_ui(self):
        # Top Header Bar
        header = tk.Frame(self.root, bg="#111318", height=50)
        header.pack(fill="x", side="top")
        
        lbl_title = tk.Label(
            header,
            text="🚁 DRONE FLOOD RESCUE — MISSION CONTROL STATION",
            font=("Segoe UI", 13, "bold"),
            bg="#111318",
            fg="#38bdf8"
        )
        lbl_title.pack(side="left", padx=20, pady=10)

        lbl_sub = tk.Label(
            header,
            text="Autonomous Human Detection & Pythagorean Geolocation",
            font=("Segoe UI", 10),
            bg="#111318",
            fg="#94a3b8"
        )
        lbl_sub.pack(side="left", padx=5, pady=10)

        # Main Container
        main_box = tk.Frame(self.root, bg="#1a1c23")
        main_box.pack(fill="both", expand=True, padx=12, pady=12)

        # LEFT SIDE: Controls & Stats Panel
        left_panel = tk.Frame(main_box, bg="#1a1c23", width=380)
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        left_panel.pack_propagate(False)

        # 1. Video Selection Card
        card_video = tk.Frame(left_panel, bg="#22252e", bd=1, relief="solid")
        card_video.pack(fill="x", pady=(0, 10), ipady=8, ipadx=8)

        tk.Label(card_video, text="📹 Video Feed Selection", font=("Segoe UI", 11, "bold"), bg="#22252e", fg="#38bdf8").pack(anchor="w", padx=10, pady=(4, 6))

        self.combo_videos = ttk.Combobox(card_video, textvariable=self.var_video_choice, state="readonly", font=("Segoe UI", 9))
        self.combo_videos.pack(fill="x", padx=10, pady=3)
        self.combo_videos.bind("<<ComboboxSelected>>", self._on_video_select)

        btn_browse = tk.Button(
            card_video,
            text="📁 Browse Custom Video File...",
            command=self._browse_file,
            bg="#334155",
            fg="#ffffff",
            activebackground="#475569",
            activeforeground="#ffffff",
            relief="flat",
            font=("Segoe UI", 9)
        )
        btn_browse.pack(fill="x", padx=10, pady=(5, 2))

        # 2. Mission Parameters Card
        card_params = tk.Frame(left_panel, bg="#22252e", bd=1, relief="solid")
        card_params.pack(fill="x", pady=(0, 10), ipady=8, ipadx=8)

        tk.Label(card_params, text="⚙️ Flight & Camera Parameters", font=("Segoe UI", 11, "bold"), bg="#22252e", fg="#38bdf8").pack(anchor="w", padx=10, pady=(4, 6))

        # Altitude slider
        f_alt = tk.Frame(card_params, bg="#22252e")
        f_alt.pack(fill="x", padx=10, pady=2)
        tk.Label(f_alt, text="Drone Altitude (h):", bg="#22252e", fg="#cbd5e1").pack(side="left")
        self.lbl_alt_val = tk.Label(f_alt, text=f"{self.var_alt.get():.0f} m", font=("Segoe UI", 9, "bold"), bg="#22252e", fg="#38bdf8")
        self.lbl_alt_val.pack(side="right")
        scale_alt = ttk.Scale(card_params, from_=5.0, to=80.0, variable=self.var_alt, command=lambda v: self.lbl_alt_val.config(text=f"{float(v):.0f} m"))
        scale_alt.pack(fill="x", padx=10, pady=(0, 4))

        # Tilt slider
        f_tilt = tk.Frame(card_params, bg="#22252e")
        f_tilt.pack(fill="x", padx=10, pady=2)
        tk.Label(f_tilt, text="Camera Tilt Angle:", bg="#22252e", fg="#cbd5e1").pack(side="left")
        self.lbl_tilt_val = tk.Label(f_tilt, text=f"{self.var_tilt.get():.0f}°", font=("Segoe UI", 9, "bold"), bg="#22252e", fg="#38bdf8")
        self.lbl_tilt_val.pack(side="right")
        scale_tilt = ttk.Scale(card_params, from_=15.0, to=85.0, variable=self.var_tilt, command=lambda v: self.lbl_tilt_val.config(text=f"{float(v):.0f}°"))
        scale_tilt.pack(fill="x", padx=10, pady=(0, 4))

        # Confidence slider
        f_conf = tk.Frame(card_params, bg="#22252e")
        f_conf.pack(fill="x", padx=10, pady=2)
        tk.Label(f_conf, text="Confidence Filter:", bg="#22252e", fg="#cbd5e1").pack(side="left")
        self.lbl_conf_val = tk.Label(f_conf, text=f"{int(self.var_conf.get()*100)}%", font=("Segoe UI", 9, "bold"), bg="#22252e", fg="#22c55e")
        self.lbl_conf_val.pack(side="right")
        scale_conf = ttk.Scale(card_params, from_=0.40, to=0.95, variable=self.var_conf, command=lambda v: self.lbl_conf_val.config(text=f"{int(float(v)*100)}%"))
        scale_conf.pack(fill="x", padx=10, pady=(0, 4))

        # 3. Mission Action Buttons
        f_actions = tk.Frame(left_panel, bg="#1a1c23")
        f_actions.pack(fill="x", pady=(0, 10))

        self.btn_start = tk.Button(
            f_actions,
            text="🚀 START MISSION",
            font=("Segoe UI", 10, "bold"),
            bg="#16a34a",
            fg="#ffffff",
            activebackground="#15803d",
            activeforeground="#ffffff",
            relief="flat",
            height=2,
            command=self.start_mission
        )
        self.btn_start.pack(fill="x", pady=2)

        f_sub_btns = tk.Frame(f_actions, bg="#1a1c23")
        f_sub_btns.pack(fill="x", pady=2)

        self.btn_pause = tk.Button(
            f_sub_btns,
            text="⏸ Pause",
            font=("Segoe UI", 9, "bold"),
            bg="#d97706",
            fg="#ffffff",
            activebackground="#b45309",
            activeforeground="#ffffff",
            relief="flat",
            state="disabled",
            command=self.toggle_pause
        )
        self.btn_pause.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.btn_stop = tk.Button(
            f_sub_btns,
            text="⏹ Stop",
            font=("Segoe UI", 9, "bold"),
            bg="#dc2626",
            fg="#ffffff",
            activebackground="#b91c1c",
            activeforeground="#ffffff",
            relief="flat",
            state="disabled",
            command=self.stop_mission
        )
        self.btn_stop.pack(side="right", fill="x", expand=True, padx=(2, 0))

        # 4. Live Telemetry & Geolocation Card
        card_telemetry = tk.Frame(left_panel, bg="#22252e", bd=1, relief="solid")
        card_telemetry.pack(fill="both", expand=True, ipady=4, ipadx=8)

        tk.Label(card_telemetry, text="📡 Real-Time Telemetry & Target", font=("Segoe UI", 11, "bold"), bg="#22252e", fg="#38bdf8").pack(anchor="w", padx=10, pady=(4, 6))

        # Status indicator
        f_stat = tk.Frame(card_telemetry, bg="#171920", padx=8, pady=6)
        f_stat.pack(fill="x", padx=10, pady=3)
        tk.Label(f_stat, text="STATUS:", font=("Segoe UI", 9), bg="#171920", fg="#94a3b8").pack(side="left")
        self.lbl_status = tk.Label(f_stat, textvariable=self.readout_status, font=("Segoe UI", 10, "bold"), bg="#171920", fg="#38bdf8")
        self.lbl_status.pack(side="right")

        # Telemetry metrics grid
        def add_metric_row(parent, label_text, str_var, val_color="#38bdf8"):
            row = tk.Frame(parent, bg="#22252e")
            row.pack(fill="x", padx=10, pady=3)
            tk.Label(row, text=label_text, font=("Segoe UI", 9), bg="#22252e", fg="#94a3b8").pack(side="left")
            tk.Label(row, textvariable=str_var, font=("Segoe UI", 10, "bold"), bg="#22252e", fg=val_color).pack(side="right")

        add_metric_row(card_telemetry, "Detected Survivors:", self.readout_count, "#22c55e")
        add_metric_row(card_telemetry, "Target GPS (Lat, Lon):", self.readout_target_gps, "#38bdf8")
        add_metric_row(card_telemetry, "Ground Distance (d_gnd):", self.readout_ground_dist, "#38bdf8")
        add_metric_row(card_telemetry, "Slant Range (Pythagoras):", self.readout_slant_dist, "#f59e0b")
        add_metric_row(card_telemetry, "Target True Bearing:", self.readout_bearing, "#38bdf8")
        add_metric_row(card_telemetry, "Peak Confidence:", self.readout_conf, "#22c55e")

        btn_open_folder = tk.Button(
            card_telemetry,
            text="📂 Open Evidence Logs & Snapshots",
            command=self._open_logs_dir,
            bg="#1e293b",
            fg="#94a3b8",
            activebackground="#334155",
            activeforeground="#ffffff",
            relief="flat",
            font=("Segoe UI", 9)
        )
        btn_open_folder.pack(fill="x", padx=10, pady=(8, 4), side="bottom")

        # RIGHT SIDE: Live Video Display & Log
        right_panel = tk.Frame(main_box, bg="#1a1c23")
        right_panel.pack(side="right", fill="both", expand=True)

        # Video Frame
        f_video = tk.Frame(right_panel, bg="#000000", bd=2, relief="solid")
        f_video.pack(fill="both", expand=True, pady=(0, 8))

        self.canvas_video = tk.Label(f_video, bg="#0a0a0c", text="[ Select a Video and Click 'START MISSION' to Begin ]", font=("Segoe UI", 12), fg="#64748b")
        self.canvas_video.pack(fill="both", expand=True)

        # Bottom Event Table
        f_table = tk.Frame(right_panel, bg="#22252e", height=150, bd=1, relief="solid")
        f_table.pack(fill="x")
        f_table.pack_propagate(False)

        cols = ("Time", "ID", "Conf", "Target GPS", "Ground Dist", "Slant (Pythagoras)", "Bearing")
        self.tree_log = ttk.Treeview(f_table, columns=cols, show="headings", height=5)
        
        self.tree_log.heading("Time", text="Time")
        self.tree_log.heading("ID", text="Track ID")
        self.tree_log.heading("Conf", text="Confidence")
        self.tree_log.heading("Target GPS", text="Survivor Coordinates")
        self.tree_log.heading("Ground Dist", text="Ground Dist")
        self.tree_log.heading("Slant (Pythagoras)", text="Slant Range")
        self.tree_log.heading("Bearing", text="Bearing")

        self.tree_log.column("Time", width=70, anchor="center")
        self.tree_log.column("ID", width=60, anchor="center")
        self.tree_log.column("Conf", width=75, anchor="center")
        self.tree_log.column("Target GPS", width=180, anchor="center")
        self.tree_log.column("Ground Dist", width=90, anchor="center")
        self.tree_log.column("Slant (Pythagoras)", width=120, anchor="center")
        self.tree_log.column("Bearing", width=70, anchor="center")

        scroll_y = ttk.Scrollbar(f_table, orient="vertical", command=self.tree_log.yview)
        self.tree_log.configure(yscrollcommand=scroll_y.set)

        scroll_y.pack(side="right", fill="y")
        self.tree_log.pack(side="left", fill="both", expand=True)

    def _populate_video_list(self):
        video_options = []
        if os.path.exists(self.videos_dir):
            files = [f for f in os.listdir(self.videos_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            for f in files:
                video_options.append(f)

        video_options.append("Live Drone Camera (Device 0)")
        self.combo_videos["values"] = video_options
        if video_options:
            # Default to people-detection if available, else first
            default_vid = "people-detection.mp4" if "people-detection.mp4" in video_options else video_options[0]
            self.combo_videos.set(default_vid)

    def _on_video_select(self, event=None):
        self.var_custom_path.set("")

    def _browse_file(self):
        f = filedialog.askopenfilename(
            title="Select Drone Video File",
            filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv"), ("All files", "*.*")]
        )
        if f:
            self.var_custom_path.set(f)
            self.combo_videos.set(f"[Custom] {os.path.basename(f)}")

    def _open_logs_dir(self):
        if sys.platform == "win32":
            os.startfile(self.logs_dir)
        elif sys.platform == "darwin":
            os.system(f"open '{self.logs_dir}'")
        else:
            os.system(f"xdg-open '{self.logs_dir}'")

    def start_mission(self):
        if self.is_running:
            return

        # Determine source
        custom = self.var_custom_path.get()
        if custom and os.path.exists(custom):
            video_source = custom
        else:
            chosen = self.var_video_choice.get()
            if "Live Drone Camera" in chosen:
                video_source = 0
            else:
                video_source = os.path.join(self.videos_dir, chosen)

        if isinstance(video_source, str) and not os.path.exists(video_source):
            messagebox.showerror("Error", f"Selected video file not found:\n{video_source}")
            return

        self.btn_start.config(state="disabled", bg="#334155")
        self.btn_pause.config(state="normal", text="⏸ Pause")
        self.btn_stop.config(state="normal")
        self.combo_videos.config(state="disabled")

        self.is_running = True
        self.is_paused = False
        self.readout_status.set("INITIALIZING MODEL...")
        self.lbl_status.config(fg="#f59e0b")

        # Run pipeline in background thread
        self.worker_thread = threading.Thread(target=self._mission_worker, args=(video_source,), daemon=True)
        self.worker_thread.start()

    def toggle_pause(self):
        if not self.is_running:
            return
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.config(text="▶ Resume", bg="#16a34a")
            self.readout_status.set("PAUSED")
            self.lbl_status.config(fg="#f59e0b")
        else:
            self.btn_pause.config(text="⏸ Pause", bg="#d97706")
            self.readout_status.set("SCANNING FLOOD ZONE")
            self.lbl_status.config(fg="#38bdf8")

    def stop_mission(self):
        self.is_running = False
        self.is_paused = False
        self.btn_start.config(state="normal", bg="#16a34a")
        self.btn_pause.config(state="disabled", text="⏸ Pause", bg="#d97706")
        self.btn_stop.config(state="disabled")
        self.combo_videos.config(state="readonly")
        self.readout_status.set("MISSION STOPPED")
        self.lbl_status.config(fg="#ef4444")

    def _mission_worker(self, video_source):
        try:
            model_path = os.path.join(self.models_dir, "yolov8n.pt")
            conf_val = float(self.var_conf.get())

            detector = DroneHumanDetector(
                model_path=model_path,
                confidence_threshold=conf_val,
                enable_tracking=True
            )

            geolocator = PythagorasGeolocator(
                horizontal_fov_deg=62.2,
                vertical_fov_deg=48.8,
                default_tilt_deg=float(self.var_tilt.get())
            )

            telemetry = MockTelemetryProvider(
                latitude=float(self.var_lat.get()),
                longitude=float(self.var_lon.get()),
                altitude_m=float(self.var_alt.get()),
                heading_deg=float(self.var_heading.get()),
                camera_tilt_deg=float(self.var_tilt.get()),
                simulate_drift=True
            )

            rescue_logger = RescueLogger(
                output_dir=self.logs_dir,
                save_snapshots=True,
                snapshot_cooldown_sec=1.5,
                export_csv=True,
                export_geojson=True
            )

            hud = DroneHudVisualizer(frame_width=640, frame_height=480)
            cap = cv2.VideoCapture(video_source)

            if not cap.isOpened():
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to open video source: {video_source}"))
                self.root.after(0, self.stop_mission)
                return

            self.root.after(0, lambda: (self.readout_status.set("SCANNING FLOOD ZONE"), self.lbl_status.config(fg="#38bdf8")))

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            delay_sec = 1.0 / max(10.0, min(60.0, fps))

            while self.is_running:
                if self.is_paused:
                    time.sleep(0.05)
                    continue

                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    if isinstance(video_source, str) and os.path.exists(video_source):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                fh, fw = frame.shape[:2]

                # Update live parameters from UI sliders
                telemetry.altitude_m = float(self.var_alt.get())
                telemetry.camera_tilt_deg = float(self.var_tilt.get())
                detector.confidence_threshold = float(self.var_conf.get())

                drone_state = telemetry.get_state()
                detections = detector.detect(frame)

                geo_results = []
                for det in detections:
                    geo = geolocator.locate_target(
                        drone_lat=drone_state.latitude,
                        drone_lon=drone_state.longitude,
                        drone_alt_m=drone_state.altitude_m,
                        drone_heading_deg=drone_state.heading_deg,
                        bbox=det["bbox"],
                        frame_width=fw,
                        frame_height=fh,
                        camera_tilt_deg=drone_state.camera_tilt_deg,
                        contact_point="bottom_center"
                    )
                    geo_results.append(geo)
                    snap = rescue_logger.log_detection(det, geo, frame)
                    if snap:
                        # Add row to UI log table
                        self._add_log_row(det, geo)

                # Render HUD
                annotated = hud.render(
                    frame=frame,
                    drone_state=drone_state,
                    detections=detections,
                    geo_results=geo_results,
                    fps=fps
                )

                # Update UI elements
                self.root.after(0, self._update_ui_frame, annotated, detections, geo_results)

                # Pace playback to video FPS
                elapsed = time.time() - t0
                rem = delay_sec - elapsed
                if rem > 0.002:
                    time.sleep(rem)

            cap.release()

        except Exception as e:
            self.root.after(0, lambda err=e: messagebox.showerror("Mission Error", f"Exception in mission worker:\n{err}"))
        finally:
            self.root.after(0, self.stop_mission)

    def _update_ui_frame(self, frame_bgr, detections, geo_results):
        if not self.is_running:
            return

        # Resize for canvas maintaining aspect ratio
        canvas_w = self.canvas_video.winfo_width()
        canvas_h = self.canvas_video.winfo_height()

        if canvas_w < 100 or canvas_h < 100:
            canvas_w, canvas_h = 720, 480

        h, w = frame_bgr.shape[:2]
        scale = min(canvas_w / w, canvas_h / h)
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))

        resized = cv2.resize(frame_bgr, (nw, nh))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        self.canvas_video.configure(image=imgtk, text="")
        self.canvas_video.image = imgtk

        # Update metrics
        num_det = len(detections)
        self.readout_count.set(str(num_det))

        if num_det > 0:
            self.readout_status.set(f"SURVIVOR DETECTED ({num_det})")
            self.lbl_status.config(fg="#22c55e")

            primary = geo_results[0]
            primary_det = detections[0]
            self.readout_target_gps.set(f"{primary['target_lat']:.6f}, {primary['target_lon']:.6f}")
            self.readout_ground_dist.set(f"{primary['ground_distance_m']} m")
            self.readout_slant_dist.set(f"{primary['slant_distance_m']} m")
            self.readout_bearing.set(f"{primary['target_bearing_deg']}°")
            self.readout_conf.set(f"{primary_det['confidence']*100:.1f} %")
        else:
            if not self.is_paused:
                self.readout_status.set("SCANNING FLOOD ZONE")
                self.lbl_status.config(fg="#38bdf8")

    def _add_log_row(self, det, geo):
        t_str = time.strftime("%H:%M:%S")
        tid = det.get("track_id", "N/A")
        conf_str = f"{det['confidence']*100:.1f}%"
        gps_str = f"{geo['target_lat']:.6f}, {geo['target_lon']:.6f}"
        g_dist = f"{geo['ground_distance_m']}m"
        s_dist = f"{geo['slant_distance_m']}m"
        brg = f"{geo['target_bearing_deg']}°"

        def insert():
            self.tree_log.insert("", 0, values=(t_str, tid, conf_str, gps_str, g_dist, s_dist, brg))
            # Keep table from overflowing
            items = self.tree_log.get_children()
            if len(items) > 50:
                self.tree_log.delete(items[-1])

        self.root.after(0, insert)

def main():
    root = tk.Tk()
    app = DroneRescueGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
