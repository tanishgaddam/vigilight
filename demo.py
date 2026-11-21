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


def demo_id(path):
    """Stable camera name for a video file, e.g. 'fight scene.mp4' -> 'demo_fight_scene'."""
    base = os.path.splitext(os.path.basename(path))[0]
    clean = "".join(c if c.isalnum() or c in "_-" else "_" for c in base)[:30]
    return "demo_" + clean.strip("_")


def list_demo_videos():
    """[{id, name, label, path}] for the dropdown. Same-named files get a numeric suffix."""
    out, seen = [], set()
    for path in find_videos():
        base = demo_id(path)
        vid, n = base, 2
        while vid in seen:
            vid, n = f"{base}_{n}", n + 1
        seen.add(vid)
        name = os.path.splitext(os.path.basename(path))[0]
        out.append({"id": vid, "name": name, "label": name.replace("_", " ").replace("-", " ").strip().title(), "path": path})
    return out


def create_demo_worker(dispatcher, video):
    """One demo CameraWorker for one entry of list_demo_videos()."""
    return CameraWorker(video["id"], video["path"], "off", dispatcher, is_demo=True)


def create_demo_workers(dispatcher):
    """Create one CameraWorker per video file found (no synthetic scenes)."""
    workers = []
    for path in find_videos():
        base = os.path.splitext(os.path.basename(path))[0]
        clean = "".join(c if c.isalnum() or c in "_-" else "_" for c in base)[:30]
        name = "demo_" + clean.strip("_")
        workers.append(CameraWorker(name, path, "off", dispatcher, is_demo=True))
    return workers
