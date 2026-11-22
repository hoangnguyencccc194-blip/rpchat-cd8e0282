# RP Chat TTS server: a small FastAPI wrapper around GPT-SoVITS (TTS_infer_pack).
# Run from the GPT-SoVITS repo root:  python rp_tts_server.py --voices voices.json --port 9880
# API:  GET /health   GET /voices   GET|POST /tts  (returns audio/wav or audio/ogg)
import os, sys, io, re, json, time, argparse, threading, traceback
now_dir = os.getcwd()
sys.path.insert(0, now_dir); sys.path.insert(0, os.path.join(now_dir, "GPT_SoVITS"))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np, soundfile as sf, torch, torchaudio

# torchaudio>=2.9 needs torchcodec for load(); use soundfile instead so a torch/torchcodec mismatch can't break us.
def _sf_load(path, *a, **k):
    d, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(np.ascontiguousarray(d.T)), sr
torchaudio.load = _sf_load

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool
import uvicorn
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

ap = argparse.ArgumentParser()
ap.add_argument("--voices", default="voices.json")
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=9880)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--half", default="auto")
ap.add_argument("--max-chars", type=int, default=400)
args = ap.parse_args()

CFG = json.load(open(args.voices, encoding="utf-8"))
VOICES = CFG["voices"]
DEFAULT_VOICE = CFG.get("default") or next(iter(VOICES))
half = (args.device.startswith("cuda")) if args.half == "auto" else args.half.lower() in ("1", "true", "yes")
v0 = VOICES[DEFAULT_VOICE]
PM = "GPT_SoVITS/pretrained_models"
tts_config = TTS_Config({"custom": {
    "device": args.device, "is_half": half, "version": CFG.get("version", "v2Pro"),
    "t2s_weights_path": v0["gpt"], "vits_weights_path": v0["sovits"],
    "bert_base_path": f"{PM}/chinese-roberta-wwm-ext-large", "cnhuhbert_base_path": f"{PM}/chinese-hubert-base"}})
pipe = TTS(tts_config)
LOADED = {"gpt": v0["gpt"], "sovits": v0["sovits"]}
LOCK = threading.Lock()
STATS = {"requests": 0, "errors": 0, "started": time.time(), "last_ms": None}
LANGS = list(tts_config.languages)
print(f"[rp-tts] version={tts_config.version} device={tts_config.device} half={tts_config.is_half} langs={LANGS}", flush=True)

ALLOW_RE = os.environ.get("CORS_ORIGIN_REGEX",
    r"https://hoangnguyencccc194-blip\.github\.io|https?://(localhost|127\.0\.0\.1)(:\d+)?")
app = FastAPI(title="RP Chat TTS (GPT-SoVITS)")
app.add_middleware(CORSMiddleware, allow_origin_regex=ALLOW_RE if os.environ.get("CORS_ALLOW_ALL") != "1" else ".*",
                   allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"],
                   expose_headers=["X-Voice", "X-Lang", "X-Duration", "X-Gen-Ms", "X-Text"], max_age=86400)

def clean_text(t: str) -> str:
    t = str(t or "")
    t = re.sub(r"\(\(.*?\)\)|\[\s*OOC[^\]]*\]|\(\s*OOC[^)]*\)|^\s*OOC\b.*$", " ", t, flags=re.I | re.M | re.S)
    t = re.sub(r"\*[^*\n]+\*", " ", t)          # *action*
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"[*_~`#>|]", " ", t)
    t = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]", " ", t)  # emoji
    return re.sub(r"\s+", " ", t).strip()

VI_CHARS = re.compile(r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]", re.I)
def latinize_vi(t: str) -> str:
    import unicodedata
    t = t.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")

def map_lang(l: str, v: dict) -> str:
    l = (l or "auto").strip().lower()
    alias = {"jp": "ja", "japanese": "ja", "english": "en", "chinese": "zh", "cn": "zh", "korean": "ko", "kr": "ko", "cantonese": "yue"}
    l = alias.get(l, l)
    return l if l in LANGS else "auto"          # e.g. "vi" is not supported by GPT-SoVITS -> auto

def ensure_weights(v):
    if v["gpt"] != LOADED["gpt"]:
        pipe.init_t2s_weights(v["gpt"]); LOADED["gpt"] = v["gpt"]
    if v["sovits"] != LOADED["sovits"]:
        pipe.init_vits_weights(v["sovits"]); LOADED["sovits"] = v["sovits"]

def pack(audio: np.ndarray, sr: int, fmt: str) -> bytes:
    buf = io.BytesIO()
    if audio.dtype != np.int16:
        audio = np.clip(audio, -1, 1) if audio.dtype.kind == "f" else audio
        audio = (audio * 32767).astype(np.int16) if audio.dtype.kind == "f" else audio.astype(np.int16)
    if fmt == "ogg": sf.write(buf, audio, sr, format="OGG", subtype="VORBIS")
    else: sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()

def synth(text, voice, lang, speed, fmt, seed):
    v = VOICES[voice]
    with LOCK:
        ensure_weights(v)
        t0 = time.time()
        req = {"text": text, "text_lang": lang, "ref_audio_path": v["ref"], "prompt_text": v.get("text", ""),
               "prompt_lang": v.get("lang", "ja"), "top_k": 15, "top_p": 1, "temperature": 1,
               "text_split_method": "cut5", "batch_size": 1, "speed_factor": float(speed),
               "fragment_interval": 0.25, "seed": int(seed), "parallel_infer": True,
               "repetition_penalty": 1.35, "split_bucket": False}
        chunks, sr = [], 32000
        for sr, a in pipe.run(req): chunks.append(a)
        audio = np.concatenate(chunks) if chunks else np.zeros(1, np.int16)
        ms = int((time.time() - t0) * 1000)
    return pack(audio, sr, fmt), sr, len(audio) / sr, ms

@app.get("/health")
@app.get("/")
def health():
    return {"ok": True, "service": "rp-tts", "engine": "GPT-SoVITS", "model_version": tts_config.version,
            "device": str(tts_config.device), "half": bool(tts_config.is_half), "languages": LANGS,
            "voices": list(VOICES), "default_voice": DEFAULT_VOICE, "busy": LOCK.locked(),
            "formats": ["wav", "ogg"], "max_chars": args.max_chars,
            "uptime_s": int(time.time() - STATS["started"]), "requests": STATS["requests"], "last_ms": STATS["last_ms"]}

@app.get("/voices")
def voices():
    return {"default": DEFAULT_VOICE, "voices": [{"id": k, "name": v.get("name", k), "prompt_lang": v.get("lang", "ja"),
            "has_prompt_text": bool(v.get("text"))} for k, v in VOICES.items()]}

@app.api_route("/tts", methods=["GET", "POST"])
async def tts(request: Request):
    p = dict(request.query_params)
    if request.method == "POST":
        try:
            body = await request.json()
            if isinstance(body, dict): p.update(body)
        except Exception:
            return JSONResponse({"ok": False, "error": "body must be JSON"}, 400)
    text = clean_text(p.get("text", ""))[: args.max_chars]
    if not re.search(r"\w", text): return JSONResponse({"ok": False, "error": "empty text after cleaning"}, 400)
    voice = p.get("voice") or DEFAULT_VOICE
    if voice not in VOICES: voice = DEFAULT_VOICE
    raw_lang = str(p.get("lang") or p.get("text_lang") or "auto").lower()
    lang = map_lang(raw_lang, VOICES[voice])
    note = ""
    # GPT-SoVITS has no Vietnamese front-end: Vietnamese text is dropped or crashes. Strip the tone marks and
    # read it with the English front-end instead (accented, "foreigner reading Vietnamese"), unless the client
    # asked for a CJK language explicitly.
    if (raw_lang.startswith("vi") or VI_CHARS.search(text)) and lang in ("auto", "en"):
        text, lang, note = latinize_vi(text), "en", "vi->latin/en"
    fmt = "ogg" if str(p.get("format") or p.get("media_type") or "wav").lower() == "ogg" else "wav"
    try: speed = min(2.0, max(0.5, float(p.get("speed", 1.0))))
    except Exception: speed = 1.0
    try: seed = int(p.get("seed", -1))
    except Exception: seed = -1
    STATS["requests"] += 1
    try:
        data, sr, dur, ms = await run_in_threadpool(synth, text, voice, lang, speed, fmt, seed)
    except Exception as e:
        STATS["errors"] += 1; traceback.print_exc()
        if "non-empty list" in str(e):
            return JSONResponse({"ok": False, "error": f"no speakable text for lang '{lang}' (GPT-SoVITS supports {', '.join(LANGS)})"}, 422)
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)
    STATS["last_ms"] = ms
    print(f"[rp-tts] {voice}/{lang}{' '+note if note else ''} {len(text)} chars -> {dur:.1f}s audio in {ms} ms", flush=True)
    return Response(data, media_type="audio/ogg" if fmt == "ogg" else "audio/wav",
                    headers={"X-Voice": voice, "X-Lang": lang + (f" ({note})" if note else ""), "X-Duration": f"{dur:.2f}", "X-Gen-Ms": str(ms),
                             "Cache-Control": "no-store"})

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
