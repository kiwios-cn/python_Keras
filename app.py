import cv2
import mediapipe as mp
import face_recognition
import os
import json
from flask import Flask, render_template, request, jsonify, url_for

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['FACES_FOLDER'] = 'static/faces'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs(app.config['FACES_FOLDER'], exist_ok=True)

mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)


def get_face_encoding(face_img):
    """放大人脸后提取128维特征向量，提升小图/模糊图的提取成功率"""
    h, w = face_img.shape[:2]
    if h < 100 or w < 100:
        scale = max(100 / h, 100 / w)
        face_img = cv2.resize(face_img, (int(w * scale), int(h * scale)))
    rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    return encodings[0] if encodings else None


def is_duplicate(encoding, saved_encodings, threshold=0.7):
    """与已保存的特征向量比对，距离小于阈值则视为同一人"""
    if not saved_encodings:
        return False
    distances = face_recognition.face_distance(saved_encodings, encoding)
    return bool(distances.min() < threshold)


def extract_faces(frame, frame_idx, video_name, h, w):
    """从单帧中检测并截取所有人脸，返回人脸信息列表"""
    face_list = []
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection_results = face_detector.process(rgb_frame)

    if not detection_results.detections:
        return face_list

    video_face_dir = os.path.join(app.config['FACES_FOLDER'], video_name)
    os.makedirs(video_face_dir, exist_ok=True)

    for face_idx, detection in enumerate(detection_results.detections):
        bbox = detection.location_data.relative_bounding_box
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = x1 + int(bbox.width * w)
        y2 = y1 + int(bbox.height * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        face_filename = f"face_{frame_idx:04d}_{face_idx}.jpg"
        face_path = os.path.join(video_face_dir, face_filename)
        cv2.imwrite(face_path, face_crop)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        face_list.append({
            "frame_idx": frame_idx,
            "face_idx": face_idx,
            "bbox": [x1, y1, x2, y2],
            "confidence": float(detection.score[0]),
            "image_url": url_for('static', filename=f'faces/{video_name}/{face_filename}'),
            "crop": face_crop
        })

    return face_list


def analyze_video(input_path, output_name):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = os.path.join(app.config['RESULT_FOLDER'], output_name)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))

    stats = {
        "max_people": 0,
        "up_frames": 0,
        "down_frames": 0,
        "fall_detected": False,
        "run_detected": False,
        "total_faces": 0,
        "faces": []
    }

    prev_hip_y, prev_ankle_x = None, None
    run_counter, fall_counter = 0, 0
    frame_idx = 0
    saved_encodings = []   # 特征提取成功的人脸向量
    saved_face_boxes = []  # 特征提取失败时IoU兜底用

    video_base_name = os.path.splitext(os.path.basename(input_path))[0]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 人脸截取 + 双重去重
        faces_in_frame = extract_faces(frame, frame_idx, video_base_name, h, w)
        for face in faces_in_frame:
            encoding = get_face_encoding(face["crop"])
            face.pop("crop")

            if encoding is not None:
                # 特征提取成功：向量相似度判重
                if is_duplicate(encoding, saved_encodings):
                    continue
                saved_encodings.append(encoding)
            else:
                # 特征提取失败（人脸模糊/太小）：降级用IoU判重
                if any(iou(face["bbox"], box) > 0.3 for box in saved_face_boxes):
                    continue
                saved_face_boxes.append(face["bbox"])

            stats["faces"].append(face)
            stats["total_faces"] += 1

        # 姿态检测
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # 抬头/低头
            nose_y = lm[0].y
            sh_y = (lm[11].y + lm[12].y) / 2
            if nose_y > sh_y - 0.05:
                stats["down_frames"] += 1
            else:
                stats["up_frames"] += 1

            # 摔倒检测
            hip_y = (lm[23].y + lm[24].y) / 2
            if prev_hip_y is not None and (hip_y - prev_hip_y) > 0.15:
                fall_counter += 1
            prev_hip_y = hip_y

            # 奔跑检测
            ankle_x = (lm[31].x + lm[32].x) / 2
            if prev_ankle_x is not None and abs(ankle_x - prev_ankle_x) > 0.05:
                run_counter += 1
            prev_ankle_x = ankle_x

            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    stats["fall_detected"] = "是" if fall_counter > 3 else "否"
    stats["run_detected"] = "是" if run_counter > 10 else "否"
    stats["max_people"] = stats["total_faces"]

    meta_path = os.path.join(app.config['FACES_FOLDER'], video_base_name, "faces_meta.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(stats["faces"], f, ensure_ascii=False, indent=2)

    return stats


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['video']
    if file:
        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)
        res_stats = analyze_video(path, "res_" + file.filename)
        res_stats["video_url"] = url_for('static', filename='results/res_' + file.filename)
        return jsonify(res_stats)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
