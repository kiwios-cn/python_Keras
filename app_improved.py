import cv2
import mediapipe as mp
import numpy as np
import os
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, url_for
from ultralytics import YOLO
from insightface.app import FaceAnalysis

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['FACES_FOLDER'] = 'static/faces'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs(app.config['FACES_FOLDER'], exist_ok=True)

LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)


def setup_logger(video_name):
    log_path = os.path.join(LOG_DIR, f"{video_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = logging.getLogger(video_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, log_path


mp_pose   = mp.solutions.pose
pose      = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
yolo_face = YOLO('weights/yolov8n-face.pt')

insight_app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
insight_app.prepare(ctx_id=0, det_size=(320, 320))

SKIP_FRAMES          = 5
MAX_ENCODINGS        = 10
MIN_ENCODING_DIST    = 0.20
DIST_THRESHOLD       = 0.40
POST_DEDUP_THRESHOLD = 0.65
CONF_THRESHOLD       = 0.60
MIN_FACE_AREA        = 4000
MAX_FACES_PER_FRAME  = 5
BLUR_THRESHOLD       = 25
FRONTAL_EYE_DX_RATIO = 0.20
FRONTAL_EYE_DY_RATIO = 0.12


def enhance_face_patch(face_img, blur_val):
    h, w  = face_img.shape[:2]
    scale = 200 / max(h, w, 1)
    up    = (cv2.resize(face_img, (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_LANCZOS4)
             if scale > 1.0 else face_img.copy())
    amount  = 3.0 if blur_val < 5 else (2.0 if blur_val < 20 else 1.2)
    blurred = cv2.GaussianBlur(up, (0, 0), 1.2)
    sharp   = np.clip(cv2.addWeighted(up, 1 + amount, blurred, -amount, 0), 0, 255).astype(np.uint8)
    lab     = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l       = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return cv2.bilateralFilter(enhanced, d=3, sigmaColor=30, sigmaSpace=30)


def enhance_frame_global(frame):
    lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    blurred  = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return np.clip(cv2.addWeighted(enhanced, 1.8, blurred, -0.8, 0), 0, 255).astype(np.uint8)


def cosine_distance(a, b):
    return 1.0 - float(np.dot(a, b))


def get_face_encoding(face_img, blur_val, logger, tag=""):
    proc = enhance_face_patch(face_img, blur_val) if blur_val < 100 else face_img.copy()
    proc = cv2.resize(proc, (160, 160), interpolation=cv2.INTER_LANCZOS4)
    try:
        faces = insight_app.get(proc)
    except Exception as e:
        logger.debug(f"    [{tag}] ✗ InsightFace 异常: {e}")
        return None

    if not faces:
        logger.debug(f"    [{tag}] ✗ InsightFace 未检测到人脸")
        return None

    best  = max(faces, key=lambda f: f.det_score)
    kps   = best.kps
    if kps is None or len(kps) < 2:
        logger.debug(f"    [{tag}] ✗ 关键点缺失")
        return None

    face_w = proc.shape[1]
    eye_dx = abs(float(kps[0][0]) - float(kps[1][0]))
    eye_dy = abs(float(kps[0][1]) - float(kps[1][1]))

    if eye_dx < face_w * FRONTAL_EYE_DX_RATIO:
        logger.debug(f"    [{tag}] ✗ 侧脸: eye_dx={eye_dx:.1f}")
        return None
    if eye_dy > face_w * FRONTAL_EYE_DY_RATIO:
        logger.debug(f"    [{tag}] ✗ 斜脸: eye_dy={eye_dy:.1f}")
        return None

    logger.debug(f"    [{tag}] ✓ 编码成功 eye_dx={eye_dx:.1f} eye_dy={eye_dy:.1f}")
    return best.normed_embedding


def is_duplicate(encoding, saved_encodings_list, logger, tag=""):
    for i, person_encodings in enumerate(saved_encodings_list):
        dists    = [cosine_distance(encoding, e) for e in person_encodings]
        avg_dist = float(np.mean(dists))
        min_dist = float(np.min(dists))
        if avg_dist < DIST_THRESHOLD:
            logger.debug(f"    [{tag}] → 匹配人员#{i+1} avg={avg_dist:.3f}")
            if len(person_encodings) < MAX_ENCODINGS and min_dist > MIN_ENCODING_DIST:
                person_encodings.append(encoding)
            return True
    return False


def post_dedup(saved_encodings_list, faces, logger, threshold=POST_DEDUP_THRESHOLD):
    keep   = []
    merged = set()

    for i in range(len(saved_encodings_list)):
        if i in merged:
            continue
        keep.append(i)
        for j in range(i + 1, len(saved_encodings_list)):
            if j in merged:
                continue
            dists = [cosine_distance(a, b)
                     for a in saved_encodings_list[i]
                     for b in saved_encodings_list[j]]
            avg = float(np.mean(dists))
            logger.info(f"  人员#{i+1} vs 人员#{j+1}: avg_dist={avg:.3f}")
            if avg < threshold:
                logger.info(f"  后处理去重: 人员#{j+1} 合并入 人员#{i+1} (avg_dist={avg:.3f})")
                merged.add(j)

    logger.info(f"  后处理去重: {len(faces)} 人 → {len(keep)} 人")
    return [faces[i] for i in keep]


def is_blurry(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    var  = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var < BLUR_THRESHOLD, var


def extract_faces(frame, frame_idx, video_name, h, w, logger):
    proc_frame = enhance_frame_global(frame)
    face_list  = []
    results    = yolo_face(proc_frame, verbose=False)

    if not results or len(results[0].boxes) == 0:
        return face_list

    video_face_dir = os.path.join(app.config['FACES_FOLDER'], video_name)
    os.makedirs(video_face_dir, exist_ok=True)

    for face_idx, box in enumerate(results[0].boxes):
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        area   = (x2 - x1) * (y2 - y1)

        if conf < CONF_THRESHOLD:
            continue
        if area < MIN_FACE_AREA:
            continue

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        blurry, blur_val = is_blurry(face_crop)
        if blurry:
            continue

        face_filename = f"face_{frame_idx:04d}_{face_idx}.jpg"
        face_path     = os.path.join(video_face_dir, face_filename)
        cv2.imwrite(face_path, cv2.resize(face_crop, (150, 150), interpolation=cv2.INTER_LANCZOS4))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        face_list.append({
            "area":       area,
            "bbox":       [x1, y1, x2, y2],
            "confidence": conf,
            "blur_val":   blur_val,
            "crop":       face_crop,
            "image_url":  url_for('static', filename=f'faces/{video_name}/{face_filename}'),
            "frame_idx":  frame_idx,
            "face_idx":   face_idx,
        })

    face_list.sort(key=lambda f: f["area"], reverse=True)
    return face_list[:MAX_FACES_PER_FRAME]


def diagnose_video(input_path, logger):
    cap   = cv2.VideoCapture(input_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("=" * 60)
    logger.info(f"  分辨率: {w}×{h} | FPS: {fps:.1f} | 总帧数: {total} | 时长: {total/max(fps,1):.1f}s")
    brightness_list, blur_list = [], []
    for idx in np.linspace(0, total - 1, min(10, total), dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_list.append(gray.mean())
        blur_list.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    cap.release()
    avg_b    = float(np.mean(brightness_list)) if brightness_list else 0
    avg_blur = float(np.mean(blur_list)) if blur_list else 0
    label    = ('极度模糊' if avg_blur < 10 else '严重模糊' if avg_blur < 50 else
                '轻度模糊' if avg_blur < 100 else '正常')
    logger.info(f"  平均亮度: {avg_b:.1f} | 平均清晰度: {avg_blur:.1f} ({label})")
    logger.info("=" * 60)


def analyze_video(input_path, output_name):
    video_base_name  = os.path.splitext(os.path.basename(input_path))[0]
    logger, log_path = setup_logger(video_base_name)
    diagnose_video(input_path, logger)

    cap          = cv2.VideoCapture(input_path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = os.path.join(app.config['RESULT_FOLDER'], output_name)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))
    stats = {
        "max_people":    0,
        "up_frames":     0,
        "down_frames":   0,
        "fall_detected": False,
        "run_detected":  False,
        "total_faces":   0,
        "faces":         [],
        "log_path":      log_path,
    }

    prev_hip_y = prev_ankle_x = None
    run_counter = fall_counter = frame_idx = 0
    saved_encodings_list = []

    logger.info(f"参数: CONF={CONF_THRESHOLD} AREA={MIN_FACE_AREA} BLUR={BLUR_THRESHOLD} "
                f"DIST={DIST_THRESHOLD} POST_DEDUP={POST_DEDUP_THRESHOLD} "
                f"EYE_DX={FRONTAL_EYE_DX_RATIO} EYE_DY={FRONTAL_EYE_DY_RATIO}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if frame_idx % 50 == 0:
            logger.info(f"进度: {frame_idx}/{total_frames} | 已识别: {stats['total_faces']} 人")

        if frame_idx % SKIP_FRAMES == 0:
            faces_in_frame = extract_faces(frame, frame_idx, video_base_name, h, w, logger)
            for face in faces_in_frame:
                tag      = f"帧{face['frame_idx']:04d}_脸{face['face_idx']}"
                encoding = get_face_encoding(face["crop"], face["blur_val"], logger, tag)
                if encoding is None:
                    continue
                if is_duplicate(encoding, saved_encodings_list, logger, tag):
                    continue
                saved_encodings_list.append([encoding])
                stats["faces"].append({
                    "frame_idx":  face["frame_idx"],
                    "face_idx":   face["face_idx"],
                    "bbox":       face["bbox"],
                    "confidence": face["confidence"],
                    "image_url":  face["image_url"]
                })
                stats["total_faces"] += 1
                logger.info(f"  ★ 新人员 #{stats['total_faces']} [{tag}] conf={face['confidence']:.3f}")

        pose_results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if pose_results.pose_landmarks:
            lm     = pose_results.pose_landmarks.landmark
            nose_y = lm[0].y
            sh_y   = (lm[11].y + lm[12].y) / 2
            if nose_y > sh_y - 0.05:
                stats["down_frames"] += 1
            else:
                stats["up_frames"] += 1

            hip_y = (lm[23].y + lm[24].y) / 2
            if prev_hip_y is not None and (hip_y - prev_hip_y) > 0.15:
                fall_counter += 1
            prev_hip_y = hip_y

            ankle_x = (lm[31].x + lm[32].x) / 2
            if prev_ankle_x is not None and abs(ankle_x - prev_ankle_x) > 0.05:
                run_counter += 1
            prev_ankle_x = ankle_x

            mp.solutions.drawing_utils.draw_landmarks(
                frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    logger.info("开始后处理全局去重...")
    stats["faces"] = post_dedup(saved_encodings_list, stats["faces"], logger)
    stats["total_faces"] = len(stats["faces"])

    stats["fall_detected"] = "是" if fall_counter > 3 else "否"
    stats["run_detected"]  = "是" if run_counter > 10 else "否"
    stats["max_people"]    = stats["total_faces"]

    logger.info(f"最终识别人数: {stats['total_faces']}")

    meta_path = os.path.join(app.config['FACES_FOLDER'], video_base_name, "faces_meta.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(stats["faces"], f, ensure_ascii=False, indent=2)

    return stats


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('video')
    if not file:
        return jsonify({"error": "未上传文件"}), 400
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)
    res_stats = analyze_video(path, "res_" + file.filename)
    res_stats["video_url"] = url_for('static', filename='results/res_' + file.filename)
    return jsonify(res_stats)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
