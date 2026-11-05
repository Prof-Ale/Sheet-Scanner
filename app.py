import os
import io
import uuid
import base64
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import music21

from omr_engine import (
    get_system_status,
    load_document_images,
    preprocess_sheet_image,
    run_audiveris_omr,
    split_voices_to_parts,
    run_gemini_vision_omr,
    run_builtin_cv_omr,
    abc_to_musicxml,
    export_all_formats,
    generate_sample_sheet_image,
    find_musescore_executable,
)

BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "jobs_output"
STATIC_DIR = BASE_DIR / "static"
JOBS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Sheet Scanner OMR — Conversor de Partituras",
    description="Converte fotos e PDFs de partituras em MusicXML, MIDI e MuseScore (.mscz)",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _image_to_data_url(img, max_w: int = 1200) -> str:
    copy_img = img.copy()
    if copy_img.width > max_w:
        ratio = max_w / float(copy_img.width)
        copy_img = copy_img.resize((max_w, int(copy_img.height * ratio)))
    buf = io.BytesIO()
    copy_img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


@app.get("/", response_class=HTMLResponse)
async def index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Interface não encontrada em static/index.html</h1>", status_code=404)
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/api/status")
async def api_status():
    return get_system_status()


@app.post("/api/convert")
async def convert_sheet(
    file: UploadFile = File(...),
    engine: str = Form("auto"),
    gemini_key: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    layout: str = Form("original"),
):
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    orig_filename = file.filename or "partitura.png"
    ext = Path(orig_filename).suffix.lower()
    allowed_exts = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Formato '{ext}' não suportado. Envie PDF, PNG, JPG ou WEBP.",
        )

    input_path = job_dir / f"input{ext}"
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        images = load_document_images(str(input_path))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao abrir o arquivo enviado: {e}")

    if not images:
        raise HTTPException(status_code=400, detail="Nenhuma página encontrada no arquivo.")

    base_title = (title or Path(orig_filename).stem).strip() or "Partitura"
    used_engine = engine
    abc_notation: Optional[str] = None
    xml_content: Optional[str] = None
    engine_warnings = []

    status = get_system_status()
    effective_gemini_key = (gemini_key or os.environ.get("GEMINI_API_KEY", "")).strip()

    # Prioridade no modo automático: Audiveris > Gemini. O motor "builtin" só roda se escolhido.
    if engine == "auto":
        if status["audiveris"]["installed"]:
            used_engine = "audiveris"
        elif effective_gemini_key:
            used_engine = "gemini"
        else:
            raise HTTPException(422, "Nenhum motor disponível: instale o Audiveris ou informe uma chave Gemini.")

    try:
        if used_engine == "audiveris":
            try:
                xml_content = run_audiveris_omr(str(input_path), str(job_dir))
            except Exception as aud_err:
                if not effective_gemini_key:
                    raise
                engine_warnings.append(f"Audiveris falhou ({aud_err}); usando Gemini.")
                used_engine = "gemini"
        if used_engine == "gemini":
            xml_content, abc_notation = await run_gemini_vision_omr(images, effective_gemini_key)
        elif used_engine == "builtin":
            xml_content = run_builtin_cv_omr(images, title=base_title)
        if layout == "satb":
            xml_content = split_voices_to_parts(xml_content)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    export_data = export_all_formats(
        xml_content=xml_content,
        output_dir=str(job_dir),
        base_name=base_title,
    )

    # Generate preview images (original & preprocessed first page)
    orig_preview = _image_to_data_url(images[0])
    proc_preview = _image_to_data_url(preprocess_sheet_image(images[0]))

    files_urls = {}
    for fmt_key, fname in export_data["files"].items():
        if fname:
            files_urls[fmt_key] = f"/api/download/{job_id}/{fname}"

    return {
        "job_id": job_id,
        "engine_used": used_engine,
        "warnings": engine_warnings,
        "page_count": len(images),
        "orig_preview": orig_preview,
        "proc_preview": proc_preview,
        "abc_notation": abc_notation,
        "musescore_native": export_data["musescore_native"],
        "summary": export_data["summary"],
        "playback_notes": export_data["playback_notes"],
        "xml_content": export_data["xml_content"],
        "download_urls": files_urls,
        "filenames": export_data["files"],
    }


@app.post("/api/sample/{song_id}")
async def run_sample_sheet(song_id: str):
    """Generate a sample sheet music image and run it through the OMR pipeline."""
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    img, title = generate_sample_sheet_image(song_id)
    sample_path = job_dir / "sample_sheet.png"
    img.save(str(sample_path), "PNG")

    xml_content = run_builtin_cv_omr([img], title=title)
    export_data = export_all_formats(
        xml_content=xml_content,
        output_dir=str(job_dir),
        base_name=song_id,
    )

    files_urls = {}
    for fmt_key, fname in export_data["files"].items():
        if fname:
            files_urls[fmt_key] = f"/api/download/{job_id}/{fname}"

    return {
        "job_id": job_id,
        "engine_used": "builtin",
        "warnings": [],
        "page_count": 1,
        "orig_preview": _image_to_data_url(img),
        "proc_preview": _image_to_data_url(preprocess_sheet_image(img)),
        "abc_notation": None,
        "musescore_native": export_data["musescore_native"],
        "summary": export_data["summary"],
        "playback_notes": export_data["playback_notes"],
        "xml_content": export_data["xml_content"],
        "download_urls": files_urls,
        "filenames": export_data["files"],
    }


class TransposeRequest(BaseModel):
    job_id: str
    semitones: int
    base_name: str = "partitura"


@app.post("/api/transpose")
async def transpose_score(req: TransposeRequest):
    """Transpose an existing converted score by N semitones and regenerate all files."""
    job_dir = JOBS_DIR / req.job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Sessão da partitura não encontrada.")

    xml_files = list(job_dir.glob("*.musicxml"))
    if not xml_files:
        raise HTTPException(status_code=404, detail="Arquivo MusicXML não encontrado na sessão.")

    try:
        score = music21.converter.parse(str(xml_files[0]))
        transposed = score.transpose(req.semitones)
        tmp_xml = job_dir / "_transposed_tmp.musicxml"
        transposed.write("musicxml", fp=str(tmp_xml))
        xml_content = tmp_xml.read_text(encoding="utf-8", errors="replace")
        tmp_xml.unlink(missing_ok=True)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Erro ao transpor partitura: {e}")

    export_data = export_all_formats(
        xml_content=xml_content,
        output_dir=str(job_dir),
        base_name=req.base_name,
    )

    files_urls = {}
    for fmt_key, fname in export_data["files"].items():
        if fname:
            files_urls[fmt_key] = f"/api/download/{req.job_id}/{fname}"

    return {
        "job_id": req.job_id,
        "musescore_native": export_data["musescore_native"],
        "summary": export_data["summary"],
        "playback_notes": export_data["playback_notes"],
        "xml_content": export_data["xml_content"],
        "download_urls": files_urls,
        "filenames": export_data["files"],
    }


class AbcUpdateRequest(BaseModel):
    job_id: str
    abc_notation: str
    base_name: str = "partitura"


@app.post("/api/update-abc")
async def update_from_abc(req: AbcUpdateRequest):
    """Update score from edited ABC notation and regenerate .musicxml, .mxl, .mid, .mscz."""
    job_dir = JOBS_DIR / req.job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        xml_content = abc_to_musicxml(req.abc_notation)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Erro na sintaxe ABC: {e}")

    export_data = export_all_formats(
        xml_content=xml_content,
        output_dir=str(job_dir),
        base_name=req.base_name,
    )

    files_urls = {}
    for fmt_key, fname in export_data["files"].items():
        if fname:
            files_urls[fmt_key] = f"/api/download/{req.job_id}/{fname}"

    return {
        "job_id": req.job_id,
        "musescore_native": export_data["musescore_native"],
        "summary": export_data["summary"],
        "playback_notes": export_data["playback_notes"],
        "xml_content": export_data["xml_content"],
        "download_urls": files_urls,
        "filenames": export_data["files"],
    }


@app.post("/api/open-musescore/{job_id}")
async def open_in_musescore(job_id: str):
    """Launch MuseScore 4 on the local desktop with the generated .mscz or .musicxml file."""
    musescore_bin = find_musescore_executable()
    if not musescore_bin:
        raise HTTPException(
            status_code=404,
            detail="MuseScore 4 não foi encontrado neste computador. Baixe o arquivo .mscz ou .musicxml.",
        )

    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Sessão não encontrada.")

    mscz_files = list(job_dir.glob("*.mscz"))
    xml_files = list(job_dir.glob("*.musicxml"))
    target = mscz_files[0] if mscz_files else (xml_files[0] if xml_files else None)
    if not target:
        raise HTTPException(status_code=404, detail="Arquivo de partitura não encontrado.")

    try:
        subprocess.Popen([musescore_bin, str(target)])
        return {"opened": True, "file": target.name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao abrir o MuseScore: {e}")


@app.get("/api/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    safe_job = Path(job_id).name
    safe_file = Path(filename).name
    file_path = JOBS_DIR / safe_job / safe_file
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado.")

    media_types = {
        ".musicxml": "application/vnd.recordare.musicxml+xml",
        ".mxl": "application/vnd.recordare.musicxml",
        ".mid": "audio/midi",
        ".mscz": "application/x-musescore",
    }
    ext = file_path.suffix.lower()
    return FileResponse(
        path=str(file_path),
        filename=safe_file,
        media_type=media_types.get(ext, "application/octet-stream"),
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
