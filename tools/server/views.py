import io
import os
import re
import shutil
import tempfile
import time
from http import HTTPStatus
from pathlib import Path

import numpy as np
import ormsgpack
import soundfile as sf
import torch
import asyncio
import torchaudio

from kui.asgi import (
    Body,
    HTTPException,
    HttpView,
    JSONResponse,
    Routes,
    SocketView,
    StreamResponse,
    UploadFile,
    request,
    websocket,
)
from loguru import logger
from typing_extensions import Annotated

from fish_speech.utils.schema import (
    AddReferenceRequest,
    AddReferenceResponse,
    DeleteReferenceResponse,
    ListReferencesResponse,
    ServeTTSRequest,
    ServeVQGANDecodeRequest,
    ServeVQGANDecodeResponse,
    ServeVQGANEncodeRequest,
    ServeVQGANEncodeResponse,
    UpdateReferenceResponse,
)
from tools.server.api_utils import (
    buffer_to_async_generator,
    format_response,
    get_content_type,
    inference_async,
)
from tools.server.inference import inference_wrapper as inference
from tools.server.model_manager import ModelManager
from tools.server.model_utils import (
    batch_vqgan_decode,
    cached_vqgan_batch_encode,
)

MAX_NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", 1))

routes = Routes()


def _resample_if_needed(audio_np: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1D float audio to target_sr if different."""
    if target_sr is None or target_sr == orig_sr:
        return audio_np
    try:
        audio_t = torch.from_numpy(audio_np).unsqueeze(0)  # (1, T)
        resampled = torchaudio.functional.resample(audio_t, orig_sr, target_sr)
        return resampled.squeeze(0).numpy()
    except Exception as e:
        logger.warning(f"[RESAMPLE] failed {orig_sr}->{target_sr}, using original: {e}")
        return audio_np


@routes.http("/v1/health")
class Health(HttpView):
    @classmethod
    async def get(cls):
        return JSONResponse({"status": "ok"})

    @classmethod
    async def post(cls):
        return JSONResponse({"status": "ok"})


@routes.http.post("/v1/vqgan/encode")
async def vqgan_encode(req: Annotated[ServeVQGANEncodeRequest, Body(exclusive=True)]):
    """
    Encode audio using VQGAN model.
    """
    try:
        # Get the model from the app
        model_manager: ModelManager = request.app.state.model_manager
        decoder_model = model_manager.decoder_model

        # Encode the audio
        start_time = time.time()
        tokens = cached_vqgan_batch_encode(decoder_model, req.audios)
        logger.info(
            f"[EXEC] VQGAN encode time: {(time.time() - start_time) * 1000:.2f}ms"
        )

        # Return the response
        return ormsgpack.packb(
            ServeVQGANEncodeResponse(tokens=[i.tolist() for i in tokens]),
            option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
        )
    except Exception as e:
        logger.error(f"Error in VQGAN encode: {e}", exc_info=True)
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR, content="Failed to encode audio"
        )


@routes.http.post("/v1/vqgan/decode")
async def vqgan_decode(req: Annotated[ServeVQGANDecodeRequest, Body(exclusive=True)]):
    """
    Decode tokens to audio using VQGAN model.
    """
    try:
        # Get the model from the app
        model_manager: ModelManager = request.app.state.model_manager
        decoder_model = model_manager.decoder_model

        # Decode the audio
        tokens = [torch.tensor(token, dtype=torch.int) for token in req.tokens]
        start_time = time.time()
        audios = batch_vqgan_decode(decoder_model, tokens)
        logger.info(
            f"[EXEC] VQGAN decode time: {(time.time() - start_time) * 1000:.2f}ms"
        )
        audios = [audio.astype(np.float16).tobytes() for audio in audios]

        # Return the response
        return ormsgpack.packb(
            ServeVQGANDecodeResponse(audios=audios),
            option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
        )
    except Exception as e:
        logger.error(f"Error in VQGAN decode: {e}", exc_info=True)
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR, content="Failed to decode tokens to audio"
        )


@routes.http.post("/v1/tts")
async def tts(req: Annotated[ServeTTSRequest, Body(exclusive=True)]):
    """
    Generate speech from text using TTS model.
    """
    try:
        # Get the model from the app
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine
        sample_rate = engine.decoder_model.sample_rate

        # Check if the text is too long
        if app_state.max_text_length > 0 and len(req.text) > app_state.max_text_length:
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                content=f"Text is too long, max length is {app_state.max_text_length}",
            )

        # Check if streaming is enabled
        # Allow pcm/wav streaming (matches fish-audio cloud)
        if req.streaming and req.format not in ("wav", "pcm"):
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                content="Streaming only supports WAV or PCM format",
            )

        # Normalize alias fields from client (output_format/voice_id)
        req.format = req.format or "pcm"
        if req.reference_id is None:
            req.reference_id = getattr(req, "voice_id", None)

        # Perform TTS
        if req.streaming:
            return StreamResponse(
                iterable=inference_async(req, engine),
                headers={
                    "Content-Disposition": f"attachment; filename=audio.{req.format}",
                },
                content_type=get_content_type(req.format),
            )
        else:
            fake_audios = next(inference(req, engine))
            if req.format == "pcm":
                # Return raw little-endian int16 PCM
                pcm_bytes = (fake_audios * 32768).astype(np.int16).tobytes()
                return StreamResponse(
                    iterable=buffer_to_async_generator(pcm_bytes),
                    headers={"Content-Disposition": "attachment; filename=audio.pcm"},
                    content_type="audio/pcm",
                )
            else:
                buffer = io.BytesIO()
                sf.write(
                    buffer,
                    fake_audios,
                    sample_rate,
                    format=req.format,
                )

                return StreamResponse(
                    iterable=buffer_to_async_generator(buffer.getvalue()),
                    headers={
                        "Content-Disposition": f"attachment; filename=audio.{req.format}",
                    },
                    content_type=get_content_type(req.format),
                )
    except HTTPException:
        # Re-raise HTTP exceptions as they are already properly formatted
        raise
    except Exception as e:
        logger.error(f"Error in TTS generation: {e}", exc_info=True)
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR, content="Failed to generate speech"
        )


@routes.http.post("/v1/references/add")
async def add_reference(
    id: str = Body(...), audio: UploadFile = Body(...), text: str = Body(...)
):
    """
    Add a new reference voice with audio file and text.
    """
    temp_file_path = None

    try:
        # Validate input parameters
        if not id or not id.strip():
            raise ValueError("Reference ID cannot be empty")

        if not text or not text.strip():
            raise ValueError("Reference text cannot be empty")

        # Get the model manager to access the reference loader
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine

        # Read the uploaded audio file
        audio_content = audio.read()
        if not audio_content:
            raise ValueError("Audio file is empty or could not be read")

        # Create a temporary file for the audio data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_content)
            temp_file_path = temp_file.name

        # Add the reference using the engine's reference loader
        engine.add_reference(id, temp_file_path, text)

        response = AddReferenceResponse(
            success=True,
            message=f"Reference voice '{id}' added successfully",
            reference_id=id,
        )
        return format_response(response)

    except FileExistsError as e:
        logger.warning(f"Reference ID '{id}' already exists: {e}")
        response = AddReferenceResponse(
            success=False,
            message=f"Reference ID '{id}' already exists",
            reference_id=id,
        )
        return format_response(response, status_code=409)  # Conflict

    except ValueError as e:
        logger.warning(f"Invalid input for reference '{id}': {e}")
        response = AddReferenceResponse(success=False, message=str(e), reference_id=id)
        return format_response(response, status_code=400)

    except (FileNotFoundError, OSError) as e:
        logger.error(f"File system error for reference '{id}': {e}")
        response = AddReferenceResponse(
            success=False, message="File system error occurred", reference_id=id
        )
        return format_response(response, status_code=500)

    except Exception as e:
        logger.error(f"Unexpected error adding reference '{id}': {e}", exc_info=True)
        response = AddReferenceResponse(
            success=False, message="Internal server error occurred", reference_id=id
        )
        return format_response(response, status_code=500)

    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError as e:
                logger.warning(
                    f"Failed to clean up temporary file {temp_file_path}: {e}"
                )


@routes.http.get("/v1/references/list")
async def list_references():
    """
    Get a list of all available reference voice IDs.
    """
    try:
        # Get the model manager to access the reference loader
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine

        # Get the list of reference IDs
        reference_ids = engine.list_reference_ids()

        response = ListReferencesResponse(
            success=True,
            reference_ids=reference_ids,
            message=f"Found {len(reference_ids)} reference voices",
        )
        return format_response(response)

    except Exception as e:
        logger.error(f"Unexpected error listing references: {e}", exc_info=True)
        response = ListReferencesResponse(
            success=False, reference_ids=[], message="Internal server error occurred"
        )
        return format_response(response, status_code=500)


@routes.http.delete("/v1/references/delete")
async def delete_reference(reference_id: str = Body(...)):
    """
    Delete a reference voice by ID.
    """
    try:
        # Validate input parameters
        if not reference_id or not reference_id.strip():
            raise ValueError("Reference ID cannot be empty")

        # Get the model manager to access the reference loader
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine

        # Delete the reference using the engine's reference loader
        engine.delete_reference(reference_id)

        response = DeleteReferenceResponse(
            success=True,
            message=f"Reference voice '{reference_id}' deleted successfully",
            reference_id=reference_id,
        )
        return format_response(response)

    except FileNotFoundError as e:
        logger.warning(f"Reference ID '{reference_id}' not found: {e}")
        response = DeleteReferenceResponse(
            success=False,
            message=f"Reference ID '{reference_id}' not found",
            reference_id=reference_id,
        )
        return format_response(response, status_code=404)  # Not Found

    except ValueError as e:
        logger.warning(f"Invalid input for reference '{reference_id}': {e}")
        response = DeleteReferenceResponse(
            success=False, message=str(e), reference_id=reference_id
        )
        return format_response(response, status_code=400)

    except OSError as e:
        logger.error(f"File system error deleting reference '{reference_id}': {e}")
        response = DeleteReferenceResponse(
            success=False,
            message="File system error occurred",
            reference_id=reference_id,
        )
        return format_response(response, status_code=500)

    except Exception as e:
        logger.error(
            f"Unexpected error deleting reference '{reference_id}': {e}", exc_info=True
        )
        response = DeleteReferenceResponse(
            success=False,
            message="Internal server error occurred",
            reference_id=reference_id,
        )
        return format_response(response, status_code=500)


@routes.http.post("/v1/references/update")
async def update_reference(
    old_reference_id: str = Body(...), new_reference_id: str = Body(...)
):
    """
    Rename a reference voice directory from old_reference_id to new_reference_id.
    """
    try:
        # Validate input parameters
        if not old_reference_id or not old_reference_id.strip():
            raise ValueError("Old reference ID cannot be empty")
        if not new_reference_id or not new_reference_id.strip():
            raise ValueError("New reference ID cannot be empty")
        if old_reference_id == new_reference_id:
            raise ValueError("New reference ID must be different from old reference ID")

        # Validate ID format per ReferenceLoader rules
        id_pattern = r"^[a-zA-Z0-9\-_ ]+$"
        if not re.match(id_pattern, new_reference_id) or len(new_reference_id) > 255:
            raise ValueError(
                "New reference ID contains invalid characters or is too long"
            )

        # Access engine to update caches after renaming
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine

        refs_base = Path("references")
        old_dir = refs_base / old_reference_id
        new_dir = refs_base / new_reference_id

        # Existence checks
        if not old_dir.exists() or not old_dir.is_dir():
            raise FileNotFoundError(f"Reference ID '{old_reference_id}' not found")
        if new_dir.exists():
            # Conflict: destination already exists
            response = UpdateReferenceResponse(
                success=False,
                message=f"Reference ID '{new_reference_id}' already exists",
                old_reference_id=old_reference_id,
                new_reference_id=new_reference_id,
            )
            return format_response(response, status_code=409)

        # Perform rename
        old_dir.rename(new_dir)

        # Update in-memory cache key if present
        if old_reference_id in engine.ref_by_id:
            engine.ref_by_id[new_reference_id] = engine.ref_by_id.pop(old_reference_id)

        response = UpdateReferenceResponse(
            success=True,
            message=(
                f"Reference voice renamed from '{old_reference_id}' to '{new_reference_id}' successfully"
            ),
            old_reference_id=old_reference_id,
            new_reference_id=new_reference_id,
        )
        return format_response(response)

    except FileNotFoundError as e:
        logger.warning(str(e))
        response = UpdateReferenceResponse(
            success=False,
            message=str(e),
            old_reference_id=old_reference_id,
            new_reference_id=new_reference_id,
        )
        return format_response(response, status_code=404)

    except ValueError as e:
        logger.warning(f"Invalid input for update reference: {e}")
        response = UpdateReferenceResponse(
            success=False,
            message=str(e),
            old_reference_id=old_reference_id if "old_reference_id" in locals() else "",
            new_reference_id=new_reference_id if "new_reference_id" in locals() else "",
        )
        return format_response(response, status_code=400)

    except OSError as e:
        logger.error(f"File system error renaming reference: {e}")
        response = UpdateReferenceResponse(
            success=False,
            message="File system error occurred",
            old_reference_id=old_reference_id,
            new_reference_id=new_reference_id,
        )
        return format_response(response, status_code=500)

    except Exception as e:
        logger.error(f"Unexpected error updating reference: {e}", exc_info=True)
        response = UpdateReferenceResponse(
            success=False,
            message="Internal server error occurred",
            old_reference_id=old_reference_id if "old_reference_id" in locals() else "",
            new_reference_id=new_reference_id if "new_reference_id" in locals() else "",
        )
        return format_response(response, status_code=500)


@routes.websocket("/v1/tts/live")
class TTSLiveWebSocket(SocketView):
    """
    WebSocket endpoint for real-time TTS streaming.
    Compatible with Fish Audio WebSocket protocol.
    """

    encoding = "bytes"  # MessagePack binary format

    async def on_connect(self):
        await websocket.accept()
        self.text_buffer = []
        self.request_config = {}
        self.started = False
        logger.info("[WS] TTS WebSocket connection accepted")

    async def on_receive(self, data):
        try:
            msg = ormsgpack.unpackb(data)
        except Exception as e:
            logger.error(f"[WS] Failed to decode MessagePack message: {e}")
            logger.error(f"[WS] Raw data (first 200 bytes): {data[:200]}")
            return

        # Log ALL raw messages for debugging
        logger.info(f"[WS] Raw message: event={msg.get('event')}, keys={list(msg.keys())}")

        event = msg.get("event")

        if event == "start":
            self.request_config = msg.get("request", {})
            self.started = True
            logger.info(f"[WS] TTS session started with config: {self.request_config}")
            # Explicitly ack start so clients don't time out waiting
            await websocket.send(
                {
                    "type": "websocket.send",
                    "bytes": ormsgpack.packb(
                        {
                            "event": "start",
                            "status": "ok",
                            "sample_rate": self.request_config.get(
                                "sample_rate",
                                websocket.app.state.model_manager.tts_inference_engine.decoder_model.sample_rate,
                            ),
                            "format": self.request_config.get("output_format", self.request_config.get("format", "pcm")),
                        }
                    ),
                }
            )

        elif event == "text":
            text = msg.get("text", "")
            logger.info(f"[WS] Received text event: '{text[:50]}...' (total buffer: {len(self.text_buffer) + 1})")
            if text:
                # Some clients (e.g. livekit fishaudio plugin) expect synthesis per text event
                # so we synthesize immediately to avoid idle disconnects.
                self.text_buffer = [text]
                await self._synthesize_and_send(text)
                # Clear buffer to avoid duplicate synthesis on stop/flush
                self.text_buffer = []

        elif event == "flush":
            logger.info(f"[WS] Received flush event, buffer size: {len(self.text_buffer)}")
            if self.text_buffer:
                buffered_text = "".join(self.text_buffer)
                self.text_buffer = []
                await self._synthesize_and_send(buffered_text)

        elif event == "stop":
            logger.info(f"[WS] Received stop event, buffer size: {len(self.text_buffer)}")
            if self.text_buffer:
                buffered_text = "".join(self.text_buffer)
                self.text_buffer = []
                await self._synthesize_and_send(buffered_text)
            await websocket.send({"type": "websocket.send", "bytes": ormsgpack.packb({"event": "finish", "reason": "stop"})})

    async def _synthesize_and_send(self, text: str):
        # Get engine from app state
        model_manager: ModelManager = websocket.app.state.model_manager
        engine = model_manager.tts_inference_engine

        if not text.strip():
            return

        req = ServeTTSRequest(
            text=text,
            output_format=self.request_config.get("output_format", self.request_config.get("format", "pcm")),
            voice_id=self.request_config.get("reference_id") or self.request_config.get("voice_id"),
            chunk_length=self.request_config.get("chunk_length", 200),
            streaming=False,
            sample_rate=self.request_config.get("sample_rate"),
        )

        logger.info(f"[WS] Synthesizing: {text[:50]}...")

        # Run inference in thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        def run_inference():
            # Run TTS inference and collect only the segment audio data
            audio_segments = []
            for result in engine.inference(req):
                if result.code == "segment" and isinstance(result.audio, tuple):
                    # Convert float audio to int16 PCM bytes
                    audio_data = (result.audio[1] * 32768).astype(np.int16).tobytes()
                    audio_segments.append(audio_data)
            return audio_segments

        try:
            audio_chunks = await loop.run_in_executor(None, run_inference)
            for chunk in audio_chunks:
                await websocket.send(
                    {"type": "websocket.send", "bytes": ormsgpack.packb({"event": "audio", "audio": chunk})}
                )
        except Exception as e:
            logger.error(f"[WS] TTS inference error: {e}", exc_info=True)

    async def on_disconnect(self, close_code):
        logger.info(f"[WS] TTS session disconnected with code: {close_code}")
