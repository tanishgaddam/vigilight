"""Demo mode: loads real video files from videos/ directory (or Downloads)
and runs them through the behaviour engine, alert dispatcher, and dashboard.
No synthetic stick-figure scenes.
"""
import os
from worker import CameraWorker


def find_videos():
    """Return sorted list of all video files in videos/ directory (and Downloads)."""
    vid_dir = os.path.join(os.path.dirname(__file__), "videos")
    found = []
    if os.path.isdir(vid_dir):
        for f in sorted(os.listdir(vid_dir)):
            if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov", ".webm")):
                found.append(os.path.join(vid_dir, f))

    # Also check user's Downloads folder
    downloads_dirs = [
        os.path.expanduser(r"~\Downloads"),
        r"C:\Users\ASUS\Downloads",
    ]
    for d in downloads_dirs:
        if os.path.isdir(d):
            try:
                for f in sorted(os.listdir(d)):
                    if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov", ".webm")):
                        full = os.path.join(d, f)
                        if full not in found:
                            found.append(full)
            except OSError:
                pass
    return found


def find_demo_video():
    vids = find_videos()
    return vids[0] if vids else None


def create_demo_workers(dispatcher):
    """Create one CameraWorker per video file found (no synthetic scenes)."""
    workers = []
    for path in find_videos():
        base = os.path.splitext(os.path.basename(path))[0]
        clean = "".join(c if c.isalnum() or c in "_-" else "_" for c in base)[:30]
        name = "demo_" + clean.strip("_")
        workers.append(CameraWorker(name, path, "off", dispatcher, is_demo=True))
    return workers
