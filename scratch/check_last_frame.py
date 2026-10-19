import cv2

video_path = "/scratch2/akshat.gupta/ak-object-rel-nav/visualisations/traversability_analysis/traversability_safe_4vwGX7U38Ux_0007799_tv_monitor_326.mp4"
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {total_frames}")

# Read final frame
cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 5))
ret, frame = cap.read()
if ret:
    out_path = "/home2/akshat.gupta/.gemini/antigravity-ide/brain/baeea250-3184-4a77-a4c1-cfe181b1e5b0/final_frame_ep0.png"
    cv2.imwrite(out_path, frame)
    print(f"Saved final frame to: {out_path}")
cap.release()
