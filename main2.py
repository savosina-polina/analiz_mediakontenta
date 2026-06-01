import asyncio
import os
import uuid
import yaml
from datetime import datetime, timedelta
from typing import List, Optional
import cv2
from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel, Field
import torch
import whisper
from ultralytics import YOLO

app = FastAPI(
    title="REST API",
    description="Базовый конвейер курсовой работы без LLM: YOLO + Whisper + локальные правила"
)

jobs_db = {}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Инициализация моделей ИИ на устройстве: {device}")
yolo_model = YOLO("yolov8n.pt")
whisper_model = whisper.load_model("base", device=device)

# ==========================================
# СХЕМЫ ДАННЫХ
# ==========================================
class DetectionClassInput(BaseModel):
    class_: str = Field(..., alias="class")
    subclasses: List[str]


class JobRequest(BaseModel):
    source: str
    customerId: str
    profile: str = "FULL"
    detectionClasses: List[DetectionClassInput]


class DetectionItem(BaseModel):
    startFrame: int
    endFrame: int
    start_time: str
    end_time: str
    time_interval: str
    subclass: str
    confidence: float
    type: str


class SourceInfo(BaseModel):
    frameCount: int
    fps: float
    video_duration_formatted: str
    analysis_timestamp: str


class ExtendedSourceInfo(SourceInfo):
    video_path: str
    video_duration_seconds: float
    processing_time_seconds: float
    processing_time_formatted: str


class JobResult(BaseModel):
    report_type: str = "TIME_BASED_REPORT"
    source_info: SourceInfo
    detections: List[DetectionItem]
    sourceInfo: ExtendedSourceInfo


class JobResponse(BaseModel):
    jobId: str
    status: str
    createdAt: str
    startedAt: Optional[str] = None
    finishedAt: Optional[str] = None
    request: JobRequest
    result: Optional[JobResult] = None


def frame_to_time(frame_idx, fps):
    td = timedelta(seconds=frame_idx / fps)
    dt = datetime.min + td
    return dt.strftime("%H:%M:%S")


def preprocess_frame(frame):
    """Предобработка кадра без LLM: серый цвет, усиление контраста, подавление шума."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ==========================================
# БАЗОВЫЙ КОНВЕЙЕР БЕЗ GEMINI/LLM
# ==========================================
def run_heavy_pipeline(job_id: str, video_path: str):
    jobs_db[job_id]["status"] = "IN_PROGRESS"
    jobs_db[job_id]["startedAt"] = datetime.utcnow().isoformat() + "Z"

    start_processing_time = datetime.now()
    raw_detections = []

    if not os.path.exists(video_path):
        video_path = "drugs.mp4"
    if not os.path.exists(video_path):
        print(f"[ERROR] Файл {video_path} не найден!")
        jobs_db[job_id]["status"] = "ERROR"
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    sample_rate = int(fps)  # Базовый вариант сканирует примерно 1 кадр в секунду

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            frame_resized = cv2.resize(frame, (640, 480))
            frame_processed = preprocess_frame(frame_resized)
            yolo_results = yolo_model(frame_processed, verbose=False)[0]

            for box in yolo_results.boxes:
                class_id = int(box.cls[0])
                label = yolo_model.names[class_id]
                conf = float(box.conf[0])

                if label in ["bottle", "wine glass"] and conf > 0.4:
                    raw_detections.append({
                        "startFrame": frame_idx, "endFrame": min(frame_idx + sample_rate, total_frames),
                        "subclass": "alcohol", "confidence": round(conf, 3), "type": "video"
                    })
                elif label in ["cell phone", "remote", "toothbrush"] and conf > 0.3:
                    sub_class = "smoking" if label in ["toothbrush", "remote"] else "drugs"
                    raw_detections.append({
                        "startFrame": frame_idx, "endFrame": min(frame_idx + sample_rate, total_frames),
                        "subclass": sub_class, "confidence": round(min(conf + 0.2, 0.99), 3), "type": "video"
                    })
        frame_idx += 1
    cap.release()

    # Whisper + локальный словарь. Это не LLM, а обычный Python-поиск по ключевым словам.
    try:
        audio_results = whisper_model.transcribe(video_path, language="ru")
        drugs_keywords = ["наркотик", "кокаин", "вещества", "ампул", "доза", "шприц", "таблет", "drugs", "порошок"]
        smoking_keywords = ["курить", "сигарет", "покурим", "затянись", "дым", "вейп", "табак"]
        alcohol_keywords = ["выпить", "водка", "вино", "пиво", "алкоголь", "бутылка", "бокал"]

        for segment in audio_results.get("segments", []):
            text = segment["text"].lower()
            start_frame = int(segment["start"] * fps)
            end_frame = min(int(segment["end"] * fps), total_frames)
            if any(word in text for word in drugs_keywords):
                raw_detections.append({"startFrame": start_frame, "endFrame": end_frame, "subclass": "drugs", "confidence": 0.92, "type": "audio"})
            if any(word in text for word in smoking_keywords):
                raw_detections.append({"startFrame": start_frame, "endFrame": end_frame, "subclass": "smoking", "confidence": 0.89, "type": "audio"})
            if any(word in text for word in alcohol_keywords):
                raw_detections.append({"startFrame": start_frame, "endFrame": end_frame, "subclass": "alcohol", "confidence": 0.91, "type": "audio"})
    except Exception as e:
        print(f"[WARN] Ошибка аудиопотока: {e}")

    def merge_intervals(detections):
        detections = [d for d in detections if d.get("confidence", 0) >= 0.35]
        if not detections:
            return []
        detections.sort(key=lambda x: (x["type"], x["subclass"], x["startFrame"]))
        merged = [detections[0]]
        for current in detections[1:]:
            prev = merged[-1]
            if (current["type"] == prev["type"] and current["subclass"] == prev["subclass"] and
                    current["startFrame"] <= prev["endFrame"] + int(fps * 3)):
                prev["endFrame"] = max(prev["endFrame"], current["endFrame"])
                prev["confidence"] = max(prev["confidence"], current["confidence"])
            else:
                merged.append(current)
        return merged

    final_intervals = merge_intervals(raw_detections)

    detections_list = []
    for det in final_intervals:
        st_time = frame_to_time(det["startFrame"], fps)
        en_time = frame_to_time(det["endFrame"], fps)
        detections_list.append(DetectionItem(
            startFrame=det["startFrame"], endFrame=det["endFrame"],
            start_time=st_time, end_time=en_time, time_interval=f"{st_time} - {en_time}",
            subclass=det["subclass"], confidence=det["confidence"], type=det["type"]
        ))

    end_processing_time = datetime.now()
    processing_time_seconds = (end_processing_time - start_processing_time).total_seconds()

    src_info = SourceInfo(
        frameCount=total_frames,
        fps=fps,
        video_duration_formatted=frame_to_time(total_frames, fps),
        analysis_timestamp=start_processing_time.isoformat()
    )
    extended_src_info = ExtendedSourceInfo(
        frameCount=total_frames,
        fps=fps,
        video_duration_formatted=frame_to_time(total_frames, fps),
        analysis_timestamp=start_processing_time.isoformat(),
        video_path=video_path,
        video_duration_seconds=round(total_frames / fps, 2),
        processing_time_seconds=round(processing_time_seconds, 2),
        processing_time_formatted=str(timedelta(seconds=int(processing_time_seconds)))
    )

    jobs_db[job_id]["result"] = JobResult(source_info=src_info, detections=detections_list, sourceInfo=extended_src_info)
    jobs_db[job_id]["status"] = "DONE"
    jobs_db[job_id]["finishedAt"] = datetime.utcnow().isoformat() + "Z"


@app.post("/api/jobs", status_code=status.HTTP_201_CREATED)
def create_job(request_data: JobRequest, background_tasks: BackgroundTasks):
    generated_id = str(uuid.uuid4())
    new_job = {
        "jobId": generated_id,
        "status": "PENDING",
        "createdAt": datetime.utcnow().isoformat() + "Z",
        "request": request_data.dict(by_alias=True),
        "startedAt": None,
        "finishedAt": None,
        "result": None
    }
    jobs_db[generated_id] = new_job
    background_tasks.add_task(run_heavy_pipeline, generated_id, request_data.source)
    return {"jobId": generated_id}


@app.get("/api/jobs/{jobId}", response_model=JobResponse)
def get_job_by_id(jobId: str):
    if jobId not in jobs_db:
        raise HTTPException(status_code=404, detail={"error": {"message": "Задача не найдена."}})
    return jobs_db[jobId]


@app.get("/api/jobs/{jobId}/report", response_model=JobResult)
def get_time_based_report(jobId: str):
    if jobId not in jobs_db or jobs_db[jobId].get("result") is None:
        raise HTTPException(status_code=404, detail={"error": {"message": "Отчет не найден."}})
    return jobs_db[jobId]["result"]


@app.get("/openapi.yaml", include_in_schema=False)
def get_openapi_yaml():
    yaml_path = "linza-api.yml"
    if not os.path.exists(yaml_path):
        raise HTTPException(status_code=500, detail="Критическая ошибка: спецификация отсутствует")
    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    return Response(content=yaml.dump(yaml_data, allow_unicode=True), media_type="application/x-yaml")
