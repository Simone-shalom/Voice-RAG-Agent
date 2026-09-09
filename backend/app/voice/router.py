from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from ..core.rate_limit import limiter
from .stt import transcribe_audio
from .stream_ws import websocket_voice_stream
from .tts import synthesise

router = APIRouter(prefix="/voice", tags=["voice"])
router.add_api_websocket_route("/stream", websocket_voice_stream)


class STTResponse(BaseModel):
    text: str


class TTSRequest(BaseModel):
    text: str


@router.post("/stt", response_model=STTResponse)
@limiter.limit("20/minute")
async def speech_to_text(request: Request, audio: UploadFile = File(...)):
    try:
        file_bytes = await audio.read()
        text = transcribe_audio(file_bytes, audio.filename or "audio.webm")
        return STTResponse(text=text)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/tts")
@limiter.limit("20/minute")
def text_to_speech(request: Request, body: TTSRequest):
    try:
        audio_bytes = synthesise(body.text)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
