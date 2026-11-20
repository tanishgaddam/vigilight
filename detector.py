"""YOLO wrapper. Each camera gets its own Detector because ByteTrack keeps per-stream
state (track IDs) inside the model object."""
import config as cfg

VEHICLE_NAMES = {"car", "motorcycle", "bus", "truck"}
COCO_WEAPON_NAMES = {"knife", "scissors", "baseball bat"}
IGNORED_LABELS = {"person", "people", "human", "background"}


class Detector:
    def __init__(self):
        from ultralytics import YOLO

        self.pose = YOLO(cfg.POSE_MODEL)
        self.obj = YOLO(cfg.OBJECT_MODEL)
        self.weapon_model = YOLO(cfg.WEAPON_MODEL) if cfg.WEAPON_MODEL else None
        wanted = VEHICLE_NAMES | COCO_WEAPON_NAMES
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

        ob = self.obj.track(frame, persist=True, conf=cfg.OBJECT_CONF, classes=self.obj_classes,
                            tracker="bytetrack.yaml", **self.kw)[0]
        if ob.boxes is not None and len(ob.boxes):
            n = len(ob.boxes)
            ids = ob.boxes.id.int().cpu().tolist() if ob.boxes.id is not None else [None] * n
            for box, cls, conf, tid in zip(ob.boxes.xyxy.cpu().numpy(), ob.boxes.cls.int().cpu().tolist(),
                                           ob.boxes.conf.cpu().tolist(), ids):
                label = self.obj.names[cls]
                box = tuple(float(v) for v in box)
                if label in COCO_WEAPON_NAMES:
                    if self.use_coco_weapons:
                        weapons.append({"label": label, "box": box, "conf": float(conf)})
                elif tid is not None:
                    vehicles.append({"id": tid, "label": label, "box": box})

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
