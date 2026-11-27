import os
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

PERSON = 0
CONF_REPORT = 0.25
NMS_AP = 0.7


def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter + 1e-12)


def iou_matrix(boxes1, boxes2):
    a = np.asarray(boxes1, dtype=np.float64).reshape(-1, 4)[:, None, :]
    b = np.asarray(boxes2, dtype=np.float64).reshape(-1, 4)[None, :, :]
    w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = w * h
    area_a = np.clip(a[..., 2] - a[..., 0], 0, None) * np.clip(a[..., 3] - a[..., 1], 0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)
    return inter / (area_a + area_b - inter + 1e-12)


def greedy_nms(boxes, scores, thr):
    boxes = np.asarray(boxes, dtype=np.float64)
    order = np.argsort(-np.asarray(scores), kind="stable")
    keep = []
    while len(order) > 0:
        best = order[0]
        keep.append(int(best))
        rest = order[1:]
        ious = iou_matrix(boxes[best], boxes[rest])[0]
        order = rest[ious <= thr]
    return np.array(keep, dtype=np.int64)


def cumulative_pr(tp, n_gt):
    tp = np.asarray(tp, dtype=np.float64)
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    return ctp / (ctp + cfp), ctp / max(n_gt, 1)


def load_gt(label_path, w, h):
    if not label_path.exists():
        return np.zeros((0, 4)), np.zeros(0, dtype=int)
    data = np.loadtxt(label_path, ndmin=2)
    if data.size == 0:
        return np.zeros((0, 4)), np.zeros(0, dtype=int)
    cls = data[:, 0].astype(int)
    xc, yc, bw, bh = data[:, 1] * w, data[:, 2] * h, data[:, 3] * w, data[:, 4] * h
    boxes = np.stack([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], axis=1)
    return boxes, cls


def person_preds(images, nms_thr, conf_min=0.0):
    out = {}
    for name, d in images.items():
        m = (d["cls"] == PERSON) & (d["conf"] >= conf_min)
        boxes, conf = d["boxes"][m], d["conf"][m]
        keep = greedy_nms(boxes, conf, nms_thr) if len(boxes) else np.zeros(0, dtype=int)
        out[name] = (boxes[keep], conf[keep], len(boxes))
    return out


def match(preds, images, iou_thr):
    rows = []
    for name, (boxes, conf, _) in preds.items():
        for b, c in zip(boxes, conf):
            rows.append((c, name, b))
    rows.sort(key=lambda r: -r[0])
    used = {name: np.zeros(len(d["gt_person"]), dtype=bool) for name, d in images.items()}
    tp = np.zeros(len(rows))
    best_ious = np.zeros(len(rows))
    for i, (c, name, b) in enumerate(rows):
        gt = images[name]["gt_person"]
        if len(gt) == 0:
            continue
        ious = iou_matrix(b, gt)[0]
        best_ious[i] = ious.max()
        ious[used[name]] = -1
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr:
            tp[i] = 1
            used[name][j] = True
    n_gt = sum(len(d["gt_person"]) for d in images.values())
    return rows, tp, best_ious, used, n_gt


def ap_all_point(prec, rec):
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def draw_error(images, name, gt_boxes, pred_boxes, hl_box, hl_color, text, out_path):
    img = cv2.imread(str(images[name]["path"]))
    for b in gt_boxes:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 0), 2)
    for b in pred_boxes:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
    cv2.rectangle(img, (int(hl_box[0]), int(hl_box[1])), (int(hl_box[2]), int(hl_box[3])), hl_color, 4)
    cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
    cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.imwrite(str(out_path), img)


def main():
    print("=== 3. IoU на ручном примере ===")
    A = (10, 20, 70, 90)
    B = (40, 50, 100, 110)
    print(f"iou_xyxy(A, B)   = {iou_xyxy(A, B):.4f}")
    print(f"iou_matrix(A, B) = {iou_matrix([A], [B])[0, 0]:.4f}")

    print("\n=== 6. Вопрос 2: NMS для A, B, C и накопленные P/R ===")
    q_boxes = np.array([
        [10, 20, 110, 120],
        [10, 20, 163.846, 120],
        [79.231, 20, 163.846, 120],
    ])
    q_scores = np.array([0.95, 0.85, 0.70])
    m = iou_matrix(q_boxes, q_boxes)
    print(f"IoU(A,B) = {m[0, 1]:.4f}, IoU(A,C) = {m[0, 2]:.4f}, IoU(B,C) = {m[1, 2]:.4f}")
    keep = greedy_nms(q_boxes, q_scores, 0.5)
    print("После NMS(thr=0.5) остались:", [ "ABC"[i] for i in keep])
    prec, rec = cumulative_pr([1, 0, 1, 1, 0], 4)
    print("Precision:", np.round(prec, 3).tolist())
    print("Recall:   ", np.round(rec, 3).tolist())

    print("\n=== 7. Predictions YOLOv8n на coco128 ===")
    data = check_det_dataset("coco128.yaml")
    img_dir = Path(data["val"])
    lbl_dir = Path(str(img_dir).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"))
    model = YOLO("yolov8n.pt")
    images = {}
    pred_rows = []
    for r in model.predict(source=str(img_dir), conf=0.001, iou=1.0, stream=True, verbose=False):
        path = Path(r.path)
        h, w = r.orig_shape
        boxes = r.boxes.xyxy.cpu().numpy().astype(np.float64)
        conf = r.boxes.conf.cpu().numpy().astype(np.float64)
        cls = r.boxes.cls.cpu().numpy().astype(int)
        gt_boxes, gt_cls = load_gt(lbl_dir / (path.stem + ".txt"), w, h)
        images[path.name] = {"path": path, "boxes": boxes, "conf": conf, "cls": cls,
                             "gt": gt_boxes, "gt_cls": gt_cls, "gt_person": gt_boxes[gt_cls == PERSON]}
        for b, c, k in zip(boxes, conf, cls):
            pred_rows.append([path.name, *np.round(b, 2), round(float(c), 5), int(k)])
    with open("predictions.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "x1", "y1", "x2", "y2", "conf", "cls"])
        wr.writerows(pred_rows)
    n_gt_person = sum(len(d["gt_person"]) for d in images.values())
    print(f"Изображений: {len(images)}, predictions: {len(pred_rows)}, GT person: {n_gt_person}")

    print("\n=== 5. Сравнение greedy_nms с torchvision.ops.nms ===")
    all_same = True
    for name, d in images.items():
        if len(d["boxes"]) == 0:
            continue
        mine = greedy_nms(d["boxes"], d["conf"], 0.5)
        tv = torchvision.ops.nms(torch.tensor(d["boxes"]), torch.tensor(d["conf"]), 0.5).numpy()
        if not np.array_equal(mine, tv):
            all_same = False
            print(f"  расхождение на {name}: {len(mine)} vs {len(tv)}")
    first = next(iter(images))
    d = images[first]
    mine = greedy_nms(d["boxes"], d["conf"], 0.5)
    tv = torchvision.ops.nms(torch.tensor(d["boxes"]), torch.tensor(d["conf"]), 0.5).numpy()
    print(f"{first}: boxes={len(d['boxes'])}, mine={len(mine)}, torchvision={len(tv)}, "
          f"индексы совпадают: {np.array_equal(mine, tv)}")
    print(f"Все {len(images)} изображений, thr=0.5, индексы совпадают: {all_same}")

    print(f"\n=== 8. PR для person (своя NMS thr={NMS_AP}) ===")
    preds = person_preds(images, NMS_AP)
    rows, tp, _, _, n_gt = match(preds, images, 0.5)
    prec50, rec50 = cumulative_pr(tp, n_gt)
    ap50 = ap_all_point(prec50, rec50)
    aps = []
    for t in np.arange(0.5, 0.96, 0.05):
        _, tp_t, _, _, _ = match(preds, images, t)
        p_t, r_t = cumulative_pr(tp_t, n_gt)
        aps.append(ap_all_point(p_t, r_t))
    ap5095 = float(np.mean(aps))
    print(f"Predictions person после NMS: {len(rows)}, GT person: {n_gt}")
    print(f"AP50 = {ap50:.4f}, AP50-95 = {ap5095:.4f}")

    print("\n=== 9. PR curve -> pr_curve.png ===")
    plt.figure(figsize=(6, 5))
    plt.plot(rec50, prec50, color="#2a6fdb", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR curve, person, IoU=0.5 (AP50={ap50:.3f})")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("pr_curve.png", dpi=150)
    plt.close()

    print("\n=== 10. Сравнение с model.val ===")
    metrics = YOLO("yolov8n.pt").val(data="coco128.yaml", plots=False, verbose=False)
    idx = list(metrics.box.ap_class_index).index(PERSON)
    lib_ap50 = float(metrics.box.ap50[idx])
    lib_ap = float(metrics.box.ap[idx])
    print(f"{'':12}{'AP50':>10}{'AP50-95':>10}")
    print(f"{'свой':12}{ap50:>10.4f}{ap5095:>10.4f}")
    print(f"{'ultralytics':12}{lib_ap50:>10.4f}{lib_ap:>10.4f}")

    print("\n=== 11. Таблица NMS thr (person, conf>=0.25, IoU match 0.5) ===")
    table = []
    for thr in [0.3, 0.5, 0.7, 0.9]:
        p = person_preds(images, thr, CONF_REPORT)
        before = sum(v[2] for v in p.values())
        after = sum(len(v[0]) for v in p.values())
        _, tp_t, _, _, n = match(p, images, 0.5)
        P = tp_t.sum() / max(len(tp_t), 1)
        R = tp_t.sum() / n
        table.append([thr, before, after, round(float(P), 4), round(float(R), 4)])
    header = ["NMS IoU thr", "Boxes before", "Boxes after", "Precision", "Recall"]
    print("".join(f"{h:>14}" for h in header))
    for row in table:
        print("".join(f"{v:>14}" for v in row))
    with open("nms_table.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerows(table)

    print("\n=== 12. Разбор ошибок (person, conf>=0.25, NMS 0.5) ===")
    os.makedirs("errors", exist_ok=True)
    p = person_preds(images, 0.5, CONF_REPORT)
    rows, tp_t, best_ious, used, _ = match(p, images, 0.5)
    fps = [(rows[i], best_ious[i]) for i in range(len(rows)) if tp_t[i] == 0]
    print("FP (по убыванию confidence):")
    for k, ((c, name, b), biou) in enumerate(fps[:5], 1):
        print(f"  FP{k}: {name} box={np.round(b, 1).tolist()} conf={c:.3f} best IoU с GT={biou:.3f}")
        draw_error(images, name, images[name]["gt_person"], p[name][0], b, (0, 0, 255),
                   f"FP{k} conf={c:.2f} IoU={biou:.2f}", Path("errors") / f"FP{k}_{name}")
    fns = []
    for name, d in images.items():
        for j in np.where(~used[name])[0]:
            g = d["gt_person"][j]
            biou = iou_matrix(g, p[name][0])[0].max() if len(p[name][0]) else 0.0
            area = (g[2] - g[0]) * (g[3] - g[1])
            fns.append((area, name, g, biou))
    fns.sort(key=lambda r: -r[0])
    print(f"FN (всего {len(fns)}, показаны самые крупные):")
    for k, (area, name, g, biou) in enumerate(fns[:5], 1):
        print(f"  FN{k}: {name} box={np.round(g, 1).tolist()} best IoU с pred={biou:.3f}")
        draw_error(images, name, images[name]["gt_person"], p[name][0], g, (0, 255, 0),
                   f"FN{k} best IoU={biou:.2f}", Path("errors") / f"FN{k}_{name}")
    print(f"Всего FP: {len(fps)}, FN: {len(fns)}. Картинки сохранены в errors/")


if __name__ == "__main__":
    main()
