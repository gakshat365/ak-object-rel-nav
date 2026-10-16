import cv2
from pathlib import Path

video_path = Path("/scratch2/akshat.gupta/ak-object-rel-nav/visualisations/traversability_analysis/traversability_safe_4vwGX7U38Ux_0007799_tv_monitor_326.mp4")
out_dir = Path("/home2/akshat.gupta/.gemini/antigravity-ide/brain/baeea250-3184-4a77-a4c1-cfe181b1e5b0")

cap = cv2.VideoCapture(str(video_path))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Total frames: {total_frames}, FPS: {fps}")

# Intro slide is frames 0-30 (2.5s at 12fps = 30 frames).
# Step 1 is around frame 32.
# Turn in place is frames 32-35.
# Step 10 is around frame 42.
# Step 50 is around frame 82.
for f_idx in [32, 35, 50, 75, min(total_frames - 20, 95)]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if ret:
        out_file = out_dir / f"safe_step1_frame_{f_idx}.png"
        cv2.imwrite(str(out_file), frame)
        print(f"Saved: {out_file.name}")
cap.release()
print("Done extracting safe controller frames!")
