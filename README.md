# Анализ медиаконтента

Запускается код:

```bash
uvicorn main:app --reload
```

Далее переходим по ссылке:

```text
http://127.0.0.1:8000/docs
```

В Swagger есть эндпоинты:

`POST /api/jobs` — создание задачи на анализ видео.

`GET /api/jobs/{jobId}` — получение результата анализа.

Переходим в `POST /api/jobs`, нажимаем `Try it out` и заменяем шаблон JSON на:

```json
{
  "source": "drugs.mp4",
  "customerId": "student_user_1",
  "profile": "FULL",
  "detectionClasses": [
    {
      "class": "DRUGS",
      "subclasses": ["alcohol", "drugs", "smoking"]
    }
  ]
}
```

Далее нажимаем `Execute`, получаем `jobId` и копируем его.

После этого переходим в `GET /api/jobs/{jobId}`, нажимаем `Try it out`, вставляем полученный `jobId` и нажимаем `Execute`.

Если обработка видео завершилась, вернется JSON-файл со всей информацией по результатам анализа.
