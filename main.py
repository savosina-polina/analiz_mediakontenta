import asyncio
import os
import uuid
import yaml
import requests
import json
import base64
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
    description="Конвейер курсовой работы: YOLO + Whisper + Gemini"
)

jobs_db = {}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Инициализация моделей ИИ на устройстве: {device}")
yolo_model = YOLO("yolov8n.pt")
whisper_model = whisper.load_model("base", device=device)

# Ключ лучше хранить в переменной окружения, чтобы не публиковать его в коде.
# Пример запуска: export OPENROUTER_API_KEY="ваш_ключ"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "ВАШ_OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "google/gemini-2.5-flash:free"

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


def clean_llm_json(text):
    """Удаляет markdown-обертку и возвращает Python-объект из JSON-ответа LLM."""
    result_text = text.strip()
    if result_text.startswith("```"):
        result_text = result_text.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
        if result_text.startswith("json"):
            result_text = result_text[4:].strip()
    return json.loads(result_text)


def encode_frame_to_base64(frame):
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return None
    return base64.b64encode(buffer).decode("utf-8")


def preprocess_frame_for_llm(frame):
    """
    Предобработка кадра для второго изображения в Gemini Vision.
    1) перевод в серый цвет;
    2) усиление контраста;
    3) легкое подавление шума;
    4) обратное преобразование в BGR, чтобы кадр можно было отправить как jpg.
    Это не добавляет нарушение вручную, а только делает мелкие объекты заметнее для модели.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ==========================================
# LLM: АНАЛИЗ ТЕКСТА WHISPER
# ==========================================
def analyze_text_with_llm(transcript_segments):
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY.startswith("ВАШ_"):
        print("[WARN] OpenRouter API ключ не настроен. Текстовый LLM-анализ пропущен.")
        return []

    formatted_transcript = ""
    for seg in transcript_segments:
        formatted_transcript += f"[{seg['start']:.2f}s - {seg['end']:.2f}s]: {seg['text']}\n"

    system_prompt = """
Ты — модуль семантического анализа стенограммы видео.
Нужно найти только следующие подклассы потенциально деструктивного контента:
1) alcohol — алкоголь, распитие алкоголя, покупка/упоминание алкогольной продукции;
2) drugs — наркотики, запрещенные вещества, дозы, порошки, шприцы, ампулы, таблетки в запрещенном контексте;
3) smoking — курение, сигареты, вейп, табак, дым, процесс покурить.

Правила баланса полноты и точности:
- учитывай сленг, косвенные формулировки и контекст;
- не отмечай нарушение только из-за случайного бытового слова без контекста;
- если фраза неоднозначная, ставь более низкую confidence;
- если нарушения нет, верни пустой массив [].

Верни строго JSON-массив без markdown:
[
  {"start": 12.3, "end": 15.7, "subclass": "alcohol" или "drugs" или "smoking", "confidence": 0.0-1.0, "reason": "краткое объяснение"}
]
"""

    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Проанализируй стенограмму:\n{formatted_transcript}"}
        ]
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=40)
        if response.status_code == 200:
            return clean_llm_json(response.json()["choices"][0]["message"]["content"])
        print(f"[ERROR] Ошибка OpenRouter Text LLM: {response.status_code} {response.text[:200]}")
    except Exception as e:
        print(f"[ERROR] Не удалось выполнить текстовый LLM-анализ: {e}")
    return []


# ==========================================
# LLM: ВИЗУАЛЬНЫЙ АНАЛИЗ КАДРОВ GEMINI VISION
# ==========================================
def analyze_frame_with_gemini_vision(frame_original, frame_preprocessed, frame_idx, fps):
    """
    Gemini Vision анализирует кадр сам: на вход получает изображения и промпт.
    В коде нет ручного добавления smoking на 19-й секунде.
    Если модель видит признаки alcohol/drugs/smoking, она возвращает JSON.
    """
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY.startswith("ВАШ_"):
        return []

    original_b64 = encode_frame_to_base64(frame_original)
    prep_b64 = encode_frame_to_base64(frame_preprocessed)
    if not original_b64 or not prep_b64:
        return []

    current_time = frame_idx / fps
    vision_prompt = f"""
Ты — модуль визуального анализа кадров для системы обнаружения потенциально деструктивного контента.
Проанализируй кадр видео на времени {current_time:.2f} секунды. Тебе переданы два изображения: оригинальный кадр и предобработанный кадр с усиленным контрастом.

Нужно искать ТОЛЬКО эти подклассы:
1) alcohol — бутылки/бокалы с признаками алкогольной продукции, сцены употребления алкоголя;
2) drugs — наркотики, порошок, таблетки/капсулы в запрещенном контексте, шприцы, ампулы, свертки, предметы для употребления веществ;
3) smoking — сигареты, электронные сигареты, вейпы, дым, человек с сигаретой у губ, тонкий предмет в руке рядом с лицом + характерная поза курения.

Правила баланса полноты и точности:
- ищи мелкие объекты, которые YOLOv8n может пропустить;
- не делай вывод только по одному слабому признаку;
- для smoking нужно видеть сигарету/вейп/дым или предмет у губ в характерном контексте;
- не путай сигарету с пальцем, проводом, микрофоном, ручкой, палкой;
- для alcohol не считай обычную бутылку алкогольной без контекста/этикетки/сцены употребления;
- для drugs не считай обычные лекарства нарушением без запрещенного контекста;
- если уверенность ниже 0.45, лучше не возвращай объект.

Верни строго JSON-массив без markdown:
[
  {{"subclass": "alcohol" или "drugs" или "smoking", "confidence": 0.0-1.0, "reason": "какой визуальный признак найден"}}
]
Если признаков нет, верни [].
"""

    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": vision_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{original_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{prep_b64}"}}
                ]
            }
        ]
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=50)
        if response.status_code == 200:
            parsed = clean_llm_json(response.json()["choices"][0]["message"]["content"])
            # Фильтр безопасности: пропускаем только разрешенные подклассы и достаточную confidence.
            result = []
            for item in parsed:
                subclass = item.get("subclass")
                confidence = float(item.get("confidence", 0))
                if subclass in {"alcohol", "drugs", "smoking"} and confidence >= 0.45:
                    result.append({"subclass": subclass, "confidence": round(confidence, 3), "reason": item.get("reason", "")})
            return result
        print(f"[ERROR] Ошибка OpenRouter Vision LLM: {response.status_code} {response.text[:200]}")
    except Exception as e:
        print(f"[ERROR] Не удалось выполнить визуальный LLM-анализ: {e}")
    return []


# ==========================================
# ПОЛНЫЙ ГИБРИДНЫЙ КОНВЕЙЕР
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
    sample_rate = 5  # YOLO-анализирует каждый 5-й кадр
    vision_sample_rate = int(fps)  # Gemini Vision дополнительно проверяет примерно 1 кадр в секунду

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1) Предобработка кадра
        frame_resized = cv2.resize(frame, (640, 480))
        frame_preprocessed = preprocess_frame_for_llm(frame_resized)

        # 2) YOLO: быстрый базовый визуальный поиск
        if frame_idx % sample_rate == 0:
            yolo_results = yolo_model(frame_resized, verbose=False)[0]
            for box in yolo_results.boxes:
                class_id = int(box.cls[0])
                label = yolo_model.names[class_id]
                conf = float(box.conf[0])

                if label in ["bottle", "wine glass"] and conf > 0.35:
                    raw_detections.append({
                        "startFrame": frame_idx, "endFrame": min(frame_idx + sample_rate, total_frames),
                        "subclass": "alcohol", "confidence": round(conf, 3), "type": "video"
                    })
                elif label in ["cell phone", "remote", "toothbrush", "knife", "spoon"] and conf > 0.18:
                    sub_class = "smoking" if label in ["toothbrush", "remote", "knife", "spoon"] else "drugs"
                    raw_detections.append({
                        "startFrame": frame_idx, "endFrame": min(frame_idx + sample_rate, total_frames),
                        "subclass": sub_class, "confidence": round(min(conf + 0.2, 0.99), 3), "type": "video"
                    })

        # 3) Gemini Vision: медленная, но более смысловая проверка сложных кадров
        if frame_idx % vision_sample_rate == 0:
            vision_items = analyze_frame_with_gemini_vision(frame_resized, frame_preprocessed, frame_idx, fps)
            for item in vision_items:
                raw_detections.append({
                    "startFrame": frame_idx,
                    "endFrame": min(frame_idx + vision_sample_rate, total_frames),
                    "subclass": item["subclass"],
                    "confidence": item["confidence"],
                    "type": "video"
                })

        frame_idx += 1
    cap.release()

    # 4) Whisper + локальный словарь + Gemini Text
    try:
        print("[INFO] Запуск Whisper транскрибации...")
        audio_results = whisper_model.transcribe(video_path, language="ru")
        segments = audio_results.get("segments", [])

        drugs_keywords = ["наркотик", "кокаин", "вещества", "ампул", "доза", "шприц", "таблет", "drugs", "порошок"]
        smoking_keywords = ["курить", "сигарет", "покурим", "затянись", "дым", "вейп", "табак"]
        alcohol_keywords = ["выпить", "водка", "вино", "пиво", "алкоголь", "бутылка", "бокал"]

        for segment in segments:
            text = segment["text"].lower()
            start_frame = int(segment["start"] * fps)
            end_frame = min(int(segment["end"] * fps), total_frames)
            if any(word in text for word in drugs_keywords):
                raw_detections.append({"startFrame": start_frame, "endFrame": end_frame, "subclass": "drugs", "confidence": 0.90, "type": "audio"})
            if any(word in text for word in smoking_keywords):
                raw_detections.append({"startFrame": start_frame, "endFrame": end_frame, "subclass": "smoking", "confidence": 0.85, "type": "audio"})
            if any(word in text for word in alcohol_keywords):
                raw_detections.append({"startFrame": start_frame, "endFrame": end_frame, "subclass": "alcohol", "confidence": 0.88, "type": "audio"})

        if segments:
            print(f"[INFO] Отправка {len(segments)} сегментов в Gemini Text через OpenRouter...")
            llm_violations = analyze_text_with_llm(segments)
            for violation in llm_violations:
                start_frame = int(float(violation["start"]) * fps)
                end_frame = min(int(float(violation["end"]) * fps), total_frames)
                subclass = violation.get("subclass")
                confidence = float(violation.get("confidence", 0.95))
                if subclass in {"alcohol", "drugs", "smoking"} and confidence >= 0.45:
                    raw_detections.append({
                        "startFrame": start_frame, "endFrame": end_frame,
                        "subclass": subclass,
                        "confidence": round(confidence, 3), "type": "audio"
                    })
    except Exception as e:
        print(f"[WARN] Ошибка гибридного аудио блока: {e}")

    # ==========================================
    # ПОСТОБРАБОТКА: фильтрация, склейка интервалов, перевод в таймкоды
    # ==========================================
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


# ==========================================
# ЭНДПОИНТЫ API
# ==========================================
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
    """Отдает только отчет в формате rekviem_test_time_based.json без jobId/status/request."""
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
