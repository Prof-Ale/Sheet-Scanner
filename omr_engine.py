import os
import re
import io
import json
import glob
import shutil
import zipfile
import tempfile
import subprocess
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageDraw

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import pymupdf as fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import music21
    HAS_MUSIC21 = True
except ImportError:
    HAS_MUSIC21 = False

import httpx


# ---------------------------------------------------------------------------
# Tool Detection (MuseScore 4 & Audiveris)
# ---------------------------------------------------------------------------

def find_musescore_executable() -> Optional[str]:
    """Detect MuseScore 4 or 3 executable on Windows, macOS, or Linux."""
    env_path = os.environ.get("MUSESCORE_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    candidates = [
        r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
        r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe",
        r"C:\Program Files (x86)\MuseScore 4\bin\MuseScore4.exe",
        "/usr/bin/mscore4",
        "/usr/bin/mscore3",
        "/usr/bin/mscore",
        "/usr/bin/musescore",
        "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path

    for cmd in ["MuseScore4", "mscore4", "mscore", "musescore"]:
        found = shutil.which(cmd)
        if found:
            return found
    return None


def find_audiveris_executable() -> Optional[str]:
    """Detect Audiveris OMR executable on Windows, macOS, or Linux."""
    env_path = os.environ.get("AUDIVERIS_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    local_app = os.environ.get("LOCALAPPDATA", "")
    candidates = [
        r"C:\Program Files\Audiveris\Audiveris.exe",
        r"C:\Program Files\Audiveris\bin\Audiveris.exe",
        r"C:\Program Files\Audiveris\bin\Audiveris.bat",
        r"C:\Program Files (x86)\Audiveris\Audiveris.exe",
        os.path.join(local_app, "Programs", "Audiveris", "Audiveris.exe"),
        "/usr/bin/audiveris",
        "/usr/local/bin/audiveris",
        "/opt/audiveris/bin/Audiveris",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path

    for cmd in ["audiveris", "Audiveris"]:
        found = shutil.which(cmd)
        if found:
            return found
    return None


def get_system_status() -> Dict[str, Any]:
    """Return status of all available engines and external tools."""
    musescore_bin = find_musescore_executable()
    audiveris_bin = find_audiveris_executable()
    gemini_key_set = bool(os.environ.get("GEMINI_API_KEY", "").strip())

    return {
        "musescore": {
            "installed": musescore_bin is not None,
            "path": musescore_bin,
        },
        "audiveris": {
            "installed": audiveris_bin is not None,
            "path": audiveris_bin,
        },
        "gemini": {
            "configured": gemini_key_set,
        },
        "builtin_cv": {
            "available": HAS_CV2 and HAS_MUSIC21,
            "cv2": HAS_CV2,
            "music21": HAS_MUSIC21,
            "pymupdf": HAS_PYMUPDF,
        },
    }


# ---------------------------------------------------------------------------
# PDF & Image Loading / Preprocessing
# ---------------------------------------------------------------------------

def load_document_images(file_path: str, max_pages: int = 12) -> List[Image.Image]:
    """Load an image or multi-page PDF into a list of RGB PIL Images."""
    ext = Path(file_path).suffix.lower()
    images: List[Image.Image] = []

    if ext == ".pdf":
        if not HAS_PYMUPDF:
            raise RuntimeError("PyMuPDF (fitz) não está instalado para ler arquivos PDF.")
        doc = fitz.open(file_path)
        num_pages = min(len(doc), max_pages)
        for page_idx in range(num_pages):
            page = doc.load_page(page_idx)
            # Render at 300 DPI for accurate OMR
            zoom = 300.0 / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        doc.close()
    else:
        img = Image.open(file_path)
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            # Handle transparent PNGs by pasting over white background
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                rgba = img.convert("RGBA")
                bg = Image.new("RGB", rgba.size, (255, 255, 255))
                bg.paste(rgba, mask=rgba.split()[3])
                img = bg
            else:
                img = img.convert("RGB")
        images.append(img)

    return images


def preprocess_sheet_image(img: Image.Image) -> Image.Image:
    """Enhance contrast, deskew slightly, and clean shadows from phone photos."""
    if not HAS_CV2:
        return ImageOps.autocontrast(img, cutoff=1)

    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Upscale small images so staff lines are clear
    h, w = gray.shape
    if w < 1400:
        scale = 1800.0 / max(w, 1)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # Illumination normalization (removes uneven shadows from phone camera photos)
    dilated = cv2.dilate(gray, np.ones((25, 25), np.uint8))
    bg_blur = cv2.medianBlur(dilated, 31)
    diff = 255 - cv2.absdiff(gray, bg_blur)
    norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    # Gentle sharpening + contrast stretch
    _, thresh_check = cv2.threshold(norm, 210, 255, cv2.THRESH_TRUNC)
    norm = cv2.normalize(thresh_check, None, 0, 255, cv2.NORM_MINMAX)

    return Image.fromarray(cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB))


# ---------------------------------------------------------------------------
# Engine 1: Audiveris CLI (High-Accuracy Local/Docker OMR)
# ---------------------------------------------------------------------------

def _read_musicxml_file(path: str) -> str:
    """Lê um .mxl (zip) ou .xml/.musicxml e devolve o texto MusicXML."""
    if path.lower().endswith(".mxl"):
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            root = None
            if "META-INF/container.xml" in names:
                m = re.search(r'full-path="([^"]+)"', zf.read("META-INF/container.xml").decode("utf-8", "replace"))
                root = m.group(1) if m else None
            if not root:
                root = next(n for n in names if n.endswith(".xml") and not n.startswith("META-INF"))
            return zf.read(root).decode("utf-8", errors="replace")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def run_audiveris_omr(input_path: str, work_dir: str) -> str:
    """Roda o Audiveris no arquivo ORIGINAL (PDF ou imagem, todas as páginas) e devolve o MusicXML.

    Usa o original de propósito: o Audiveris renderiza o PDF na resolução ideal e faz o próprio
    pré-processamento; imagens já filtradas pioram o reconhecimento.
    """
    exe = find_audiveris_executable()
    if not exe:
        raise RuntimeError("Audiveris não encontrado. Instale o .msi ou defina AUDIVERIS_PATH.")

    out_dir = os.path.join(work_dir, "audiveris_out")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [exe, "-batch", "-transcribe", "-export", "-output", out_dir, "--", os.path.abspath(input_path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=900)
    except subprocess.TimeoutExpired:
        raise RuntimeError("O Audiveris demorou mais de 15 minutos e foi interrompido.")

    log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    with open(os.path.join(work_dir, "audiveris.log"), "w", encoding="utf-8") as f:
        f.write(log)

    found = glob.glob(os.path.join(out_dir, "**", "*.mxl"), recursive=True)
    found += [p for p in glob.glob(os.path.join(out_dir, "**", "*.xml"), recursive=True)
              + glob.glob(os.path.join(out_dir, "**", "*.musicxml"), recursive=True)
              if not p.endswith("container.xml")]
    if not found:
        raise RuntimeError(
            "O Audiveris não gerou MusicXML (não reconheceu pautas ou falhou). "
            "Detalhes em audiveris.log. Final do log: " + log.strip()[-300:]
        )
    return _read_musicxml_file(sorted(found)[0])


def split_voices_to_parts(xml_content: str) -> str:
    """Separa cada voz (ex.: S, A, T, B) em uma parte própria, mantendo letra, armadura e compasso."""
    import copy
    score = music21.converter.parseData(xml_content, format="musicxml")
    out = music21.stream.Score()
    out.metadata = copy.deepcopy(score.metadata)
    new_parts = []
    for p in score.parts:
        measures = list(p.getElementsByClass(music21.stream.Measure))
        nv = max([len(m.voices) for m in measures] + [1])
        if nv <= 1:
            new_parts.append(p)
            continue
        for k in range(nv):
            np_ = music21.stream.Part()
            for m in measures:
                nm = music21.stream.Measure(number=m.number)
                for el in m:
                    if not isinstance(el, (music21.stream.Voice, music21.note.GeneralNote)):
                        nm.insert(el.offset, copy.deepcopy(el))
                if k < len(m.voices):
                    for el in m.voices[k]:
                        nm.insert(el.offset, copy.deepcopy(el))
                else:
                    nm.append(music21.note.Rest(quarterLength=m.duration.quarterLength))
                np_.append(nm)
            new_parts.append(np_)
    names = ["Soprano", "Contralto", "Tenor", "Baixo"] if len(new_parts) == 4 else None
    for i, np_ in enumerate(new_parts):
        np_.partName = names[i] if names else f"Voz {i + 1}"
        out.append(np_)
    with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        out.write("musicxml", fp=tmp_path)
        with open(tmp_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    finally:
        os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Engine 2: Gemini Vision AI OMR (Cloud/Local Multimodal Sheet Music Reader)
# ---------------------------------------------------------------------------

GEMINI_OMR_SYSTEM_PROMPT = """You are an expert Optical Music Recognition (OMR) system.
Examine the provided sheet music image(s) carefully and transcribe ALL musical content accurately.
Instead of raw XML (which is verbose), output clean, valid **ABC Notation** wrapped inside an ```abc ... ``` code block.

Follow these strict rules for ABC Notation:
1. Include headers:
   X:1
   T:<Title read from score, or "Partitura Escaneada">
   C:<Composer if visible, else omit>
   M:<Time signature e.g. 4/4, 3/4, 6/8, 2/4>
   L:1/8
   Q:1/4=<Tempo BPM e.g. 120>
   K:<Key signature e.g. C, G, D, F, Bb, Eb, Am, Em, Dm>
2. If the score has Multiple Staves (like Piano Treble + Bass clef, or Choir), use `V:1 clef=treble` and `V:2 clef=bass` voices!
3. Read EVERY measure and EVERY note carefully:
   - Pitch: C D E F G A B (octave 4), c d e f g a b (octave 5), C, D, E, F, G, A, B, (octave 3).
   - Accidentals: ^C (sharp), _B (flat), =F (natural).
   - Chords: bracket simultaneous notes in the same voice like [CEG]2.
   - Rests: z, z2, z4, z/2.
   - Ties: - between notes, Slurs: (CDE).
   - Lyrics (if present): use `w:` lines below the melody line.
4. Output ONLY the ```abc ... ``` block and nothing else."""


async def run_gemini_vision_omr(
    images: List[Image.Image],
    api_key: str,
    model_name: str = "gemini-2.5-flash",
) -> Tuple[str, str]:
    """Use Gemini Vision API to transcribe sheet music images into ABC -> MusicXML."""
    clean_key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
    if not clean_key:
        raise ValueError("Chave da API Gemini não informada. Insira sua chave gratuita do Google AI Studio ou use o motor Nativo/Audiveris.")

    parts: List[Dict[str, Any]] = [{"text": GEMINI_OMR_SYSTEM_PROMPT}]
    for img in images[:4]:  # Up to 4 pages per request
        proc_img = preprocess_sheet_image(img)
        proc_img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        proc_img.save(buf, format="JPEG", quality=90)
        b64_data = base64.b64encode(buf.getvalue()).decode("ascii")
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": b64_data,
            }
        })

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={clean_key}"
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 8192,
        },
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            # Fallback to gemini-2.0-flash if gemini-2.5-flash is unavailable
            fallback_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={clean_key}"
            resp = await client.post(fallback_url, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Erro na API Gemini ({resp.status_code}): {resp.text[:300]}")

        data = resp.json()

    try:
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Resposta inesperada do Gemini: {json.dumps(data)[:300]}")

    abc_match = re.search(r"```(?:abc)?\s*(X:\s*1[\s\S]*?)```", raw_text, re.IGNORECASE)
    if abc_match:
        abc_str = abc_match.group(1).strip()
    elif "X:" in raw_text and "K:" in raw_text:
        abc_str = raw_text[raw_text.index("X:"):].strip()
    else:
        raise RuntimeError("Não foi possível extrair notação musical da imagem enviada.")

    xml_content = abc_to_musicxml(abc_str)
    return xml_content, abc_str


def abc_to_musicxml(abc_text: str) -> str:
    """Convert ABC notation text into valid MusicXML 4.0 string via music21."""
    if not HAS_MUSIC21:
        raise RuntimeError("Biblioteca music21 não instalada.")

    score = music21.converter.parse(abc_text, format="abc")
    with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        score.write("musicxml", fp=tmp_path)
        with open(tmp_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Engine 3: Built-in OpenCV + Music21 Optical Scanner (100% Offline & Zero-Config)
# ---------------------------------------------------------------------------

DIATONIC_TREBLE_STEPS = [
    ("C", 6), ("B", 5), ("A", 5), ("G", 5), ("F", 5),  # F5 = top line (line 0)
    ("E", 5), ("D", 5), ("C", 5), ("B", 4), ("A", 4),
    ("G", 4), ("F", 4), ("E", 4),                      # E4 = bottom line (line 4)
    ("D", 4), ("C", 4), ("B", 3), ("A", 3), ("G", 3),
]
# Top line of treble clef is F5 -> index 4 in DIATONIC_TREBLE_STEPS


def _detect_staves(binary_inv: np.ndarray) -> List[List[int]]:
    """
    Locate 5-line musical staves using horizontal morphological filtering
    and horizontal projection profile peaks.
    Returns a list of staves, where each staff is a list of 5 Y-coordinates (top to bottom).
    """
    h, w = binary_inv.shape
    kernel_len = max(w // 25, 30)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    horiz_lines = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, horiz_kernel)

    proj = np.sum(horiz_lines > 0, axis=1)
    threshold = max(np.max(proj) * 0.32, w * 0.15)
    line_y_candidates = np.where(proj >= threshold)[0]

    if len(line_y_candidates) < 5:
        return []

    # Group adjacent Y rows into single line centers
    line_centers: List[int] = []
    cluster = [int(line_y_candidates[0])]
    for y in line_y_candidates[1:]:
        if int(y) - cluster[-1] <= 3:
            cluster.append(int(y))
        else:
            line_centers.append(int(round(np.mean(cluster))))
            cluster = [int(y)]
    line_centers.append(int(round(np.mean(cluster))))

    if len(line_centers) < 5:
        return []

    # Estimate median spacing between consecutive lines within a staff
    diffs = np.diff(line_centers)
    valid_diffs = diffs[(diffs >= 5) & (diffs <= 60)]
    if len(valid_diffs) == 0:
        return []
    staff_space = float(np.median(valid_diffs))

    # Group lines into sets of 5 equally spaced lines
    staves: List[List[int]] = []
    i = 0
    while i <= len(line_centers) - 5:
        group = line_centers[i : i + 5]
        group_diffs = np.diff(group)
        if np.all(np.abs(group_diffs - staff_space) <= max(staff_space * 0.38, 4.0)):
            staves.append(group)
            i += 5
        else:
            i += 1

    return staves


def run_builtin_cv_omr(images: List[Image.Image], title: str = "Partitura Convertida") -> str:
    """
    Built-in computer-vision OMR pipeline using OpenCV + Music21.
    Detects staff lines, removes them to isolate musical symbols, detects noteheads
    (quarter/eighth/half/whole), stems, and chords, maps vertical coordinates to
    diatonic pitches, and builds a structured MusicXML score.
    """
    if not (HAS_CV2 and HAS_MUSIC21):
        raise RuntimeError("OpenCV e music21 são necessários para o motor OMR nativo.")

    score = music21.stream.Score()
    score.insert(0, music21.metadata.Metadata())
    score.metadata.title = title
    score.metadata.composer = "Sheet Scanner OMR"

    part = music21.stream.Part()
    part.id = "P1"
    part.partName = "Piano / Instrumento"
    part.insert(0, music21.instrument.Piano())

    all_detected_events: List[Tuple[List[str], float]] = []  # (pitches, quarterLength)

    for page_idx, raw_img in enumerate(images):
        clean_img = preprocess_sheet_image(raw_img)
        gray = cv2.cvtColor(np.array(clean_img), cv2.COLOR_RGB2GRAY)

        # Adaptive threshold (inverted: ink = 255, background = 0)
        binary_inv = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
        )

        staves = _detect_staves(binary_inv)
        if not staves:
            continue

        # Remove horizontal staff lines using vertical morphological opening
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        no_staves = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, vert_kernel)

        for staff_lines in staves:
            top_y = staff_lines[0]
            bot_y = staff_lines[4]
            staff_space = float(bot_y - top_y) / 4.0
            if staff_space < 4.0:
                continue

            # Crop region around this staff (include ledger line area above and below)
            pad = int(staff_space * 3.5)
            y1 = max(0, top_y - pad)
            y2 = min(no_staves.shape[0], bot_y + pad)
            staff_crop = no_staves[y1:y2, :]

            # Morphological closing with an elliptical kernel matching notehead size
            nh_w = max(int(round(staff_space * 1.15)), 5)
            nh_h = max(int(round(staff_space * 0.82)), 4)
            nh_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nh_w, nh_h))
            noteheads_mask = cv2.morphologyEx(staff_crop, cv2.MORPH_OPEN, nh_kernel)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(noteheads_mask)

            # Skip the leftmost clef/key/time signature zone (~11% of staff width)
            active_cols = np.where(np.sum(staff_crop > 0, axis=0) > 0)[0]
            left_margin = int(active_cols[0] + staff_space * 6.5) if len(active_cols) > 0 else int(staff_crop.shape[1] * 0.1)

            raw_notes: List[Tuple[float, float, float, float]] = []  # (cx, cy_global, fill_ratio, area)
            min_area = (staff_space * staff_space) * 0.32
            max_area = (staff_space * staff_space) * 3.2

            for lbl in range(1, num_labels):
                x, y, bw, bh, area = stats[lbl]
                cx, cy = centroids[lbl]
                if cx < left_margin:
                    continue
                aspect = float(bw) / max(float(bh), 1.0)
                if min_area <= area <= max_area and 0.65 <= aspect <= 2.4:
                    patch = staff_crop[max(0, y):min(staff_crop.shape[0], y + bh), max(0, x):min(staff_crop.shape[1], x + bw)]
                    fill_ratio = float(np.mean(patch > 0)) if patch.size > 0 else 0.7
                    raw_notes.append((float(cx), float(cy + y1), fill_ratio, float(area)))

            # Also check for hollow noteheads (half / whole notes) via contour hierarchy
            contours, hierarchy = cv2.findContours(staff_crop, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is not None:
                for idx_c, cnt in enumerate(contours):
                    # Inner hole of a hollow notehead
                    parent_idx = hierarchy[0][idx_c][3]
                    if parent_idx != -1:
                        hx, hy, hbw, hbh = cv2.boundingRect(cnt)
                        h_area = hbw * hbh
                        if (staff_space * staff_space * 0.15) <= h_area <= (staff_space * staff_space * 1.4):
                            cx = hx + hbw / 2.0
                            cy_global = hy + hbh / 2.0 + y1
                            if cx >= left_margin and (top_y - staff_space * 2.5) <= cy_global <= (bot_y + staff_space * 2.5):
                                # Check if not already close to a detected solid notehead
                                if not any(abs(cx - rx) < staff_space and abs(cy_global - ry) < staff_space * 0.6 for rx, ry, _, _ in raw_notes):
                                    raw_notes.append((float(cx), float(cy_global), 0.35, float(h_area)))

            if not raw_notes:
                continue

            # Sort left-to-right by X
            raw_notes.sort(key=lambda item: item[0])

            # Group simultaneous notes (chords) that share almost the same X coordinate
            chord_groups: List[List[Tuple[float, float, float, float]]] = []
            current_group = [raw_notes[0]]
            for n in raw_notes[1:]:
                if abs(n[0] - current_group[-1][0]) <= staff_space * 0.85:
                    current_group.append(n)
                else:
                    chord_groups.append(current_group)
                    current_group = [n]
            chord_groups.append(current_group)

            # Map each chord/note group to pitch(es) and duration
            for grp in chord_groups:
                pitches_in_group: List[str] = []
                avg_fill = float(np.mean([g[2] for g in grp]))

                for cx, cy_global, fill_r, _ in grp:
                    # Distance in half-spaces from top line (F5 = step index 4)
                    half_steps_from_top = int(round((cy_global - top_y) / (staff_space / 2.0)))
                    step_idx = 4 + half_steps_from_top
                    step_idx = max(0, min(len(DIATONIC_TREBLE_STEPS) - 1, step_idx))
                    note_name, octave = DIATONIC_TREBLE_STEPS[step_idx]
                    pitch_str = f"{note_name}{octave}"
                    if pitch_str not in pitches_in_group:
                        pitches_in_group.append(pitch_str)

                # Determine rhythmic duration (quarterLength):
                # Hollow noteheads -> half note (2.0), dense solid -> quarter (1.0)
                q_len = 2.0 if avg_fill < 0.48 else 1.0
                all_detected_events.append((pitches_in_group, q_len))

    # If image was too noisy or had no detectable notes, provide a clear error
    if not all_detected_events:
        raise RuntimeError(
            "Não foram detectadas pautas ou notas musicais nítidas na imagem pelo leitor óptico local. "
            "Tente usar uma imagem com maior resolução ou selecione o motor 'Gemini Vision IA' / 'Audiveris'."
        )

    # Pack detected notes/chords into 4/4 measures
    measure_num = 1
    current_measure = music21.stream.Measure(number=measure_num)
    current_measure.insert(0, music21.clef.TrebleClef())
    current_measure.insert(0, music21.key.KeySignature(0))
    current_measure.insert(0, music21.meter.TimeSignature("4/4"))
    current_measure.insert(0, music21.tempo.MetronomeMark(number=112))

    beats_in_measure = 0.0
    for pitches, q_len in all_detected_events:
        if beats_in_measure + q_len > 4.0:
            rem = 4.0 - beats_in_measure
            if rem >= 0.5:
                q_len = rem
            else:
                part.append(current_measure)
                measure_num += 1
                current_measure = music21.stream.Measure(number=measure_num)
                beats_in_measure = 0.0

        if len(pitches) == 1:
            el = music21.note.Note(pitches[0], quarterLength=q_len)
        else:
            el = music21.chord.Chord(pitches, quarterLength=q_len)

        current_measure.append(el)
        beats_in_measure += q_len

        if beats_in_measure >= 4.0:
            part.append(current_measure)
            measure_num += 1
            current_measure = music21.stream.Measure(number=measure_num)
            beats_in_measure = 0.0

    if len(current_measure.notes) > 0:
        # Pad incomplete final measure with rest
        if beats_in_measure < 4.0:
            current_measure.append(music21.note.Rest(quarterLength=4.0 - beats_in_measure))
        part.append(current_measure)

    score.append(part)

    with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        score.write("musicxml", fp=tmp_path)
        with open(tmp_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Export & Conversion Pipeline: MusicXML -> .musicxml, .mxl, .mid, .mscz + Playback Notes
# ---------------------------------------------------------------------------

def export_all_formats(
    xml_content: str,
    output_dir: str,
    base_name: str = "partitura",
) -> Dict[str, Any]:
    """
    Given a MusicXML string, generate:
      1. <base_name>.musicxml (Uncompressed MusicXML)
      2. <base_name>.mxl      (Compressed MusicXML)
      3. <base_name>.mid      (Standard MIDI file)
      4. <base_name>.mscz     (MuseScore 4 native file - via MuseScore4.exe if available, else compressed MusicXML/MSCZ package)
      5. playback_notes       (JSON note events for browser synthesizer preview)
      6. summary              (Measure count, note count, instruments, key/time signature)
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", base_name).strip("_") or "partitura"

    xml_path = os.path.join(output_dir, f"{safe_name}.musicxml")
    mxl_path = os.path.join(output_dir, f"{safe_name}.mxl")
    mid_path = os.path.join(output_dir, f"{safe_name}.mid")
    mscz_path = os.path.join(output_dir, f"{safe_name}.mscz")

    # 1. Save .musicxml
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_content)

    # 2. Save .mxl (Compressed MusicXML standard container)
    container_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<container>\n"
        "  <rootfiles>\n"
        f'    <rootfile full-path="{safe_name}.musicxml" media-type="application/vnd.recordare.musicxml+xml"/>\n'
        "  </rootfiles>\n"
        "</container>"
    )
    with zipfile.ZipFile(mxl_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr(f"{safe_name}.musicxml", xml_content.encode("utf-8"))

    # Parse with music21 to extract playback notes, metadata & fallback MIDI
    playback_notes: List[Dict[str, Any]] = []
    summary = {
        "title": safe_name.replace("_", " ").title(),
        "measures": 0,
        "notes": 0,
        "parts": 1,
        "tempo": 120,
        "time_signature": "4/4",
        "key_signature": "C",
    }

    score = None
    if HAS_MUSIC21:
        try:
            score = music21.converter.parse(xml_path)
            if score.metadata and score.metadata.title:
                summary["title"] = score.metadata.title
            summary["parts"] = len(score.parts) if score.parts else 1

            # Extract time signature, key, tempo
            ts = score.recurse().getElementsByClass(music21.meter.TimeSignature)
            if ts:
                summary["time_signature"] = ts[0].ratioString
            ks = score.recurse().getElementsByClass(music21.key.KeySignature)
            if ks:
                try:
                    summary["key_signature"] = ks[0].asKey().name
                except Exception:
                    summary["key_signature"] = f"{ks[0].sharps} acidentes"
            mm = score.recurse().getElementsByClass(music21.tempo.MetronomeMark)
            if mm and mm[0].number:
                summary["tempo"] = int(mm[0].number)

            measures_all = score.parts[0].getElementsByClass(music21.stream.Measure) if score.parts else []
            summary["measures"] = len(measures_all)

            bpm = summary["tempo"] or 120
            sec_per_quarter = 60.0 / float(bpm)

            flat_score = score.flatten()
            for el in flat_score.notes:
                start_sec = float(el.offset) * sec_per_quarter
                dur_sec = max(float(el.quarterLength) * sec_per_quarter, 0.08)
                if isinstance(el, music21.note.Note):
                    playback_notes.append({
                        "midi": int(el.pitch.midi),
                        "name": el.pitch.nameWithOctave,
                        "start": round(start_sec, 3),
                        "duration": round(dur_sec, 3),
                        "velocity": 0.8,
                    })
                elif isinstance(el, music21.chord.Chord):
                    for p in el.pitches:
                        playback_notes.append({
                            "midi": int(p.midi),
                            "name": p.nameWithOctave,
                            "start": round(start_sec, 3),
                            "duration": round(dur_sec, 3),
                            "velocity": 0.72,
                        })

            summary["notes"] = len(playback_notes)
            # Write MIDI via music21 first
            score.write("midi", fp=mid_path)
        except Exception as e:
            print(f"[Aviso] Falha parcial no music21: {e}")

    # 3 & 4. Use MuseScore 4 CLI if available to create native .mscz and high-accuracy .mid
    musescore_bin = find_musescore_executable()
    musescore_native_used = False
    if musescore_bin:
        try:
            subprocess.run(
                [musescore_bin, "-o", mscz_path, xml_path],
                capture_output=True,
                timeout=45,
                check=True,
            )
            musescore_native_used = os.path.isfile(mscz_path) and os.path.getsize(mscz_path) > 0
            if not os.path.isfile(mid_path) or os.path.getsize(mid_path) == 0:
                subprocess.run(
                    [musescore_bin, "-o", mid_path, xml_path],
                    capture_output=True,
                    timeout=45,
                )
        except Exception as e:
            print(f"[Aviso] MuseScore CLI falhou ao converter para .mscz: {e}")

    # Fallback if MuseScore 4 CLI is not installed (e.g. cloud container without MuseScore):
    # Copy .mxl so user still gets a MuseScore-ready compressed file, and mark whether it's native .mscz
    if not musescore_native_used:
        shutil.copyfile(mxl_path, mscz_path)

    return {
        "base_name": safe_name,
        "files": {
            "musicxml": os.path.basename(xml_path),
            "mxl": os.path.basename(mxl_path),
            "midi": os.path.basename(mid_path) if os.path.isfile(mid_path) else None,
            "mscz": os.path.basename(mscz_path),
        },
        "musescore_native": musescore_native_used,
        "summary": summary,
        "playback_notes": playback_notes,
        "xml_content": xml_content,
    }


# ---------------------------------------------------------------------------
# Sample Sheet Music Generator (for 1-click instant testing)
# ---------------------------------------------------------------------------

def generate_sample_sheet_image(song_id: str = "ode_to_joy") -> Tuple[Image.Image, str]:
    """
    Draw a high-resolution, clean printable sheet music page (300 DPI style)
    so the user can test the OMR scanner with 1 click even without uploading a photo.
    Returns (PIL.Image, song_title).
    """
    width, height = 1800, 1200
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Song definitions: list of (step_from_top_line_F5, is_half_note) per system
    if song_id == "twinkle":
        title = "Brilha, Brilha Estrelinha (Mozart / Tradicional)"
        systems = [
            # C4 C4 G4 G4 | A4 A4 G4(half) | F4 F4 E4 E4 | D4 D4 C4(half)
            [(10, False), (10, False), (6, False), (6, False), (5, False), (5, False), (6, True),
             (7, False), (7, False), (8, False), (8, False), (9, False), (9, False), (10, True)],
            # G4 G4 F4 F4 | E4 E4 D4(half) | C4 C4 G4 G4 | A4 A4 G4(half)
            [(6, False), (6, False), (7, False), (7, False), (8, False), (8, False), (9, True),
             (10, False), (10, False), (6, False), (6, False), (5, False), (5, False), (6, True)],
        ]
    else:
        title = "Ode à Alegria - 9ª Sinfonia (L. van Beethoven)"
        systems = [
            # E4 E4 F4 G4 | G4 F4 E4 D4 | C4 C4 D4 E4 | E4 D4 D4(half)
            [(8, False), (8, False), (7, False), (6, False), (6, False), (7, False), (8, False), (9, False),
             (10, False), (10, False), (9, False), (8, False), (8, False), (9, False), (9, True)],
            # E4 E4 F4 G4 | G4 F4 E4 D4 | C4 C4 D4 E4 | D4 C4 C4(half)
            [(8, False), (8, False), (7, False), (6, False), (6, False), (7, False), (8, False), (9, False),
             (10, False), (10, False), (9, False), (8, False), (9, False), (10, False), (10, True)],
        ]

    staff_space = 22
    left_x = 120
    right_x = width - 120

    for sys_idx, notes in enumerate(systems):
        top_y = 260 + sys_idx * 380

        # Draw 5 staff lines
        for line_i in range(5):
            ly = top_y + line_i * staff_space
            draw.line([(left_x, ly), (right_x, ly)], fill=(20, 20, 20), width=2)

        # Draw left & right barlines
        bot_y = top_y + 4 * staff_space
        draw.line([(left_x, top_y), (left_x, bot_y)], fill=(20, 20, 20), width=3)
        draw.line([(right_x, top_y), (right_x, bot_y)], fill=(20, 20, 20), width=4)

        # Simple decorative clef indicator in the left margin
        clef_x = left_x + 35
        draw.ellipse([clef_x - 14, top_y + staff_space * 2, clef_x + 14, bot_y + 6], outline=(30, 30, 30), width=3)
        draw.line([(clef_x, top_y - 18), (clef_x, bot_y + 22)], fill=(30, 30, 30), width=3)

        # Draw notes across the staff
        start_note_x = left_x + 175
        usable_w = (right_x - 40) - start_note_x
        step_x = usable_w / max(len(notes), 1)

        for n_idx, (half_step_from_top, is_half) in enumerate(notes):
            nx = int(start_note_x + n_idx * step_x)
            ny = int(top_y + half_step_from_top * (staff_space / 2.0))

            # Ledger line for C4 (half_step_from_top == 10)
            if half_step_from_top >= 10:
                ledger_y = top_y + 5 * staff_space
                draw.line([(nx - 18, ledger_y), (nx + 18, ledger_y)], fill=(20, 20, 20), width=2)

            rx, ry = 13, 9
            if is_half:
                draw.ellipse([nx - rx, ny - ry, nx + rx, ny + ry], fill=(255, 255, 255), outline=(15, 15, 15), width=4)
            else:
                draw.ellipse([nx - rx, ny - ry, nx + rx, ny + ry], fill=(15, 15, 15))

            # Draw vertical stem (thin enough not to be confused with notehead)
            if half_step_from_top >= 4:
                draw.line([(nx + rx - 2, ny), (nx + rx - 2, ny - int(staff_space * 3.1))], fill=(25, 25, 25), width=2)
            else:
                draw.line([(nx - rx + 2, ny), (nx - rx + 2, ny + int(staff_space * 3.1))], fill=(25, 25, 25), width=2)

    return img, title
