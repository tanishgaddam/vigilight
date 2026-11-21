"""YOLO wrapper. Each camera gets its own Detector because ByteTrack keeps per-stream
state (track IDs) inside the model object."""
import config as cfg

VEHICLE_NAMES = {"car", "motorcycle", "bus", "truck"}
COCO_WEAPON_NAMES = {"knife", "scissors", "baseball bat"}
IGNORED_LABELS = {"person", "people", "human", "background"}


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class VehicleTracker:
    """Small IoU/centre-distance tracker for vehicles. Unlike ByteTrack it never drops a detection
    that has no track yet, and it keeps an ID alive for a while when the car is hidden (e.g. by a hand
    moving a toy car), so the collision logic always has a stable ID to work with."""

    def __init__(self, max_age=12):
        self.tracks, self.next_id, self.max_age = {}, 1, max_age

    @staticmethod
    def _score(a, b):
        iou = _iou(a, b)
        diag = max(1.0, ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5)
        dist = (((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2) ** 2 + ((a[1] + a[3]) / 2 - (b[1] + b[3]) / 2) ** 2) ** 0.5
        return max(iou, 0.3 * (1 - dist / (2 * diag)))     # still matches when a hand moves it quickly

    def update(self, dets):
        cand = sorted(((self._score(t["box"], d["box"]), tid, i)
                       for tid, t in self.tracks.items() for i, d in enumerate(dets)), reverse=True)
        used_t, used_d, ids = set(), set(), {}
        for sc, tid, i in cand:
            if sc < 0.05:
                break
            if tid in used_t or i in used_d:
                continue
            used_t.add(tid), used_d.add(i)
            ids[i] = tid
        for tid in list(self.tracks):
            if tid not in used_t:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]
        out = []
        for i, d in enumerate(dets):
            tid = ids.get(i)
            if tid is None:
                tid, self.next_id = self.next_id, self.next_id + 1
            self.tracks[tid] = {"box": d["box"], "age": 0}
            out.append(dict(d, id=tid))
        return out


class Detector:
    def __init__(self):
        from ultralytics import YOLO

        self.pose = YOLO(cfg.POSE_MODEL)
        self.obj = YOLO(cfg.OBJECT_MODEL)
        self.weapon_model = YOLO(cfg.WEAPON_MODEL) if cfg.WEAPON_MODEL else None
        wanted = VEHICLE_NAMES | COCO_WEAPON_NAMES | set(cfg.VEHICLE_EXTRA)
        self.vehicle_names = VEHICLE_NAMES | set(cfg.VEHICLE_EXTRA)
        self.vtracker = VehicleTracker()
        self.obj_classes = [i for i, n in self.obj.names.items() if n in wanted]
        self.kw = {"verbose": False, "imgsz": cfg.IMG_SIZE}
        if cfg.DEVICE:
            self.kw["device"] = cfg.DEVICE
        # With a trained weapon model, COCO's weak knife/scissors/bat classes are switched off by default
        self.use_coco_weapons = self.weapon_model is None or cfg.WEAPON_KEEP_COCO

    def detect(self, frame):
        persons, weapons, vehicles = [], [], []

        pr = self.pose.track(frame, persist=True, conf=cfg.PERSON_CONF, tracker="bytetrack.yaml", **self.kw)[0]
        if pr.boxes is not None and pr.boxes.id is not None and pr.keypoints is not None:
            ids = pr.boxes.id.int().cpu().tolist()
            boxes = pr.boxes.xyxy.cpu().numpy()
            kpts = pr.keypoints.data.cpu().numpy()
            for pid, box, kp in zip(ids, boxes, kpts):
                persons.append({"id": pid, "box": tuple(float(v) for v in box), "kpts": kp})

        # one pass for vehicles + COCO weapons at a low confidence; weapons are filtered back to OBJECT_CONF below
        okw = dict(self.kw, imgsz=max(cfg.IMG_SIZE, cfg.VEHICLE_IMGSZ))
        ob = self.obj.predict(frame, conf=min(cfg.VEHICLE_CONF, cfg.OBJECT_CONF), classes=self.obj_classes,
                              agnostic_nms=True, **okw)[0]
        raw_vehicles = []
        if ob.boxes is not None and len(ob.boxes):
            for box, cls, conf in zip(ob.boxes.xyxy.cpu().numpy(), ob.boxes.cls.int().cpu().tolist(),
                                      ob.boxes.conf.cpu().tolist()):
                label = self.obj.names[cls]
                box = tuple(float(v) for v in box)
                if label in COCO_WEAPON_NAMES:
                    if self.use_coco_weapons and conf >= cfg.OBJECT_CONF:
                        weapons.append({"label": label, "box": box, "conf": float(conf)})
                elif label in self.vehicle_names and conf >= cfg.VEHICLE_CONF:
                    raw_vehicles.append({"label": label, "box": box, "conf": float(conf)})
        vehicles = self.vtracker.update(raw_vehicles)

        if self.weapon_model is not None:
            kw = dict(self.kw, imgsz=cfg.WEAPON_IMGSZ)
            wr = self.weapon_model.predict(frame, conf=cfg.WEAPON_CONF, **kw)[0]
            if wr.boxes is not None:
                for box, cls, conf in zip(wr.boxes.xyxy.cpu().numpy(), wr.boxes.cls.int().cpu().tolist(),
                                          wr.boxes.conf.cpu().tolist()):
                    label = wr.names[cls]
                    if label.lower() in IGNORED_LABELS:   # some datasets also label people
                        continue
                    weapons.append({"label": label, "box": tuple(float(v) for v in box), "conf": float(conf)})

        return {"persons": persons, "weapons": weapons, "vehicles": vehicles}
