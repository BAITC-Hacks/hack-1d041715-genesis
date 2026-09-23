# Система автопротоколирования совещаний

Первый модуль MVP преобразует один аудиофайл в текст. Он использует OpenAI
`whisper-1` через endpoint `/audio/transcriptions`. В официальной документации
OpenAI модель описана как мультиязычная система распознавания речи; endpoint
принимает `flac`, `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `ogg`, `wav` и `webm`.

> На момент реализации в материалах репозитория не было отдельной модели для
> казахской/тюркской речи, поэтому выбран предусмотренный проектным контекстом
> fallback — OpenAI Whisper API.

## Установка

Требуется Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Для API-режима задайте ключ в переменной окружения; `.env.example` содержит
шаблон имени переменной, но файл `.env` модуль специально не читает, чтобы не
добавлять ещё одну зависимость. Ключ не добавляется в репозиторий.

```powershell
$env:OPENAI_API_KEY = "ваш_ключ"
```

## Использование

```python
from meeting_minutes.transcription import transcribe_audio

text = transcribe_audio("recording.wav")
print(text)
```

Функция возвращает строку. Отсутствующий файл, неподдерживаемый формат, пустой
или повреждённый файл приводят к понятным исключениям из
`meeting_minutes.transcription`.

## Проверка без аккаунта OpenAI

Demo-режим нужен только для проверки интерфейса: он валидирует путь и формат,
но возвращает встроенный демонстрационный текст вместо обращения к API.

```powershell
$env:PYTHONPATH = "src"
$env:TRANSCRIPTION_MODE = "demo"
python -c "from meeting_minutes.transcription import transcribe_audio; print(transcribe_audio('recording.wav'))"
```

Для запуска модульных тестов не нужны ни ключ, ни сеть:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

Официальная спецификация: [OpenAI Speech-to-Text](https://developers.openai.com/api/docs/guides/speech-to-text).
