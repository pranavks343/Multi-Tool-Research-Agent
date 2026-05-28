"""YouTube transcript fetcher — via `youtube-transcript-api` (v1.0+ instance API)."""
from __future__ import annotations

import json
import re
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timezone

from app.config import _EXECUTOR, YOUTUBE_TIMEOUT
from app.schemas import FetchYoutubeTranscriptArgs, ToolResult, error_payload


def _extract_id(video: str) -> str:
    """Accept either a bare 11-char video ID or any common YouTube URL form."""
    if len(video) == 11 and "/" not in video and "." not in video:
        return video
    m = re.search(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})", video)
    if m:
        return m.group(1)
    raise ValueError(f"Could not extract video ID from: {video}")


def run_fetch_youtube_transcript(args: FetchYoutubeTranscriptArgs) -> str:
    """Return a JSON list[ToolResult] (single item) on success, or error JSON on failure."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return error_payload(
            "fetch_youtube_transcript", "youtube-transcript-api not installed (uv add youtube-transcript-api)"
        )

    def _fetch() -> str:
        try:
            fetched_at = datetime.now(timezone.utc)
            video_id = _extract_id(args.video)

            # v1.0+ instance-based API: YouTubeTranscriptApi().fetch(...) -> FetchedTranscript
            ytt_api = YouTubeTranscriptApi()
            fetched = ytt_api.fetch(video_id, languages=[args.language])
            transcript = fetched.to_raw_data()  # list of {text, start, duration}

            if args.include_timestamps:
                content = "\n".join(
                    f"[{int(t['start'] // 60):02d}:{int(t['start'] % 60):02d}] {t['text']}"
                    for t in transcript
                )
            else:
                content = " ".join(t["text"] for t in transcript)

            if len(content) > 15000:
                content = content[:15000] + " ... [truncated]"

            result = ToolResult(
                source="YouTube",
                title=f"YouTube video {video_id}",
                snippet=content,
                url=f"https://youtube.com/watch?v={video_id}",
                timestamp=fetched_at,
            )
            return json.dumps([result.model_dump(mode="json")])
        except Exception as e:
            return error_payload("fetch_youtube_transcript", f"{type(e).__name__}: {e}")

    future = _EXECUTOR.submit(_fetch)
    try:
        return future.result(timeout=YOUTUBE_TIMEOUT)
    except FutureTimeoutError:
        return error_payload("fetch_youtube_transcript", f"Timeout after {YOUTUBE_TIMEOUT}s")
