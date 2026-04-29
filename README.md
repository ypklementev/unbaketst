# Lyrics Eval — WhisperX Pipeline

Оценивает качество транскрипции и синхронизации текста к вокалу (htdemucs v4, m4a).

## Setup

```bash
pip install -r requirements.txt

# ffmpeg нужен для конвертации m4a
brew install ffmpeg          # macOS
apt-get install -y ffmpeg    # Ubuntu
```

## Структура файлов

```
lyrics_eval/
  eval.py          ← главный скрипт
  manifest.json    ← список треков (отредактировать под свой датасет)
  requirements.txt
  audio/           ← положить сюда .m4a файлы из Яндекс.Диска
  lyrics_cache/    ← создаётся автоматически, кэш LRC/txt
  eval_results.json ← результаты
```

## Запуск

```bash
# GPU (рекомендуется)
python eval.py --device cuda --compute-type float16

# CPU / MacBook (медленнее, но работает)
python eval.py --device cpu --compute-type int8

```

## Метрики

| Метрика | Что значит | Цель |
|---------|-----------|------|
| WER | Word Error Rate. 0 = идеально, 1 = полный провал. Считается через substitutions+deletions+insertions / total words | < 0.15 хорошо, < 0.25 приемлемо |
| CER | Character Error Rate. Первичная метрика для JP, RU. Менее чувствителен к пробелам/токенизации | < 0.10 хорошо |
| Timestamp MAE | Средняя абсолютная ошибка в секундах (предсказанная строка vs GT LRC). Считается после выравнивания по текстовой схожести | < 0.5s хорошо, < 1s приемлемо |

### Особые случаи
- **Hallucination** — флаг ставится когда одна фраза повторяется 3+ раз подряд (известный баг Whisper на тишине/инструментальных паузах). Такие треки исключаются из средних.
- **Язык авто-определяется** если не указан в manifest, но лучше указать явно для JP/RU.
- **CER primary lang**: ja, zh, ko — для них CER важнее WER.

## Как заполнить manifest.json

Открываешь файлы из датасета, находишь оригинал на Spotify/YouTube, вбиваешь artist + title. Syncedlyrics автоматически подтянет LRC из Musixmatch/NetEase. LRC кэшируется в `lyrics_cache/`.

Если syncedlyrics не находит — скрипт делает fallback на lrclib.net API (тоже автоматически).

## Интерпретация результатов

```json
{
  "summary": {
    "avg_wer": 0.142,            ← средний WER по всем трекам с GT
    "avg_cer": 0.089,
    "avg_timestamp_mae_s": 0.31, ← средняя ошибка синхронизации в секундах
    "hallucination_rate": 0.05   ← 5% треков с галлюцинациями
  }
}
```

WER 0.14 на вокале из htdemucs (с артефактами) — хороший результат. На чистом вокале Whisper large-v3 даёт ~0.05-0.08.
