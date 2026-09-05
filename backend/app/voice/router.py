from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from .stt import transcribe_audio
from .tts import synthesise

router = APIRouter(prefix="/voice", tags=["voice"])


class STTResponse(BaseModel):
    text: str


class TTSRequest(BaseModel):
    text: str


@router.post("/stt", response_model=STTResponse)
async def speech_to_text(audio: UploadFile = File(...)):
    try:
        file_bytes = await audio.read()
        text = transcribe_audio(file_bytes, audio.filename or "audio.webm")
        return STTResponse(text=text)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/tts")
def text_to_speech(request: TTSRequest):
    try:
        audio_bytes = synthesise(request.text)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
