import cv2
from pathlib import Path

video_path = Path("/scratch2/akshat.gupta/ak-object-rel-nav/visualisations/traversability_analysis/traversability_4vwGX7U38Ux_0007799_tv_monitor_326.mp4")
out_dir = Path("/home2/akshat.gupta/.gemini/antigravity-ide/brain/baeea250-3184-4a77-a4c1-cfe181b1e5b0")

cap = cv2.VideoCapture(str(video_path))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Total frames: {total_frames}, FPS: {fps}")

# Extract: Intro slide (frame 15), Early step (frame 45), Mid step (frame 55)
for f_idx in [15, 45, 58]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if ret:
        out_file = out_dir / f"traversability_sample_frame_{f_idx}.png"
        cv2.imwrite(str(out_file), frame)
        print(f"Saved: {out_file.name}")
cap.release()
print("Done extracting sample frames!")
