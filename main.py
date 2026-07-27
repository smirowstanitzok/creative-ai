#!/usr/bin/env python3
"""Creative Cloud Infrastructure Suite - On-Demand-GPU-Pods auf Runpod.

Startet für ein gewähltes Produktionsprofil einen Runpod-GPU-Pod, richtet darin
den passenden KI-Stack ein (Modelle/Pakete werden automatisch geladen), wartet
bis die Oberfläche über den Runpod-Proxy antwortet, gibt den Link aus und löscht
den Pod danach *garantiert* wieder - auch bei Fehlern, Ctrl-C oder SIGTERM.
Ohne persistentes Volume entstehen so nur Kosten für die reine Laufzeit.

API: Runpod REST v1 (``https://rest.runpod.io/v1``). Der GraphQL-Endpunkt, den
das Python-SDK ``runpod`` intern verwendet, ist laut Runpod-Release-Notes vom
23.07.2026 abgekündigt; REST v1 ist der dokumentierte Weg. API v2 ist noch
öffentliche Beta ("Endpoints and behavior may change") und wird hier bewusst
noch nicht genutzt - ein Wechsel betrifft nur ``API_BASE_URL`` und
``RunpodClient``.

Benötigt Python 3.10+ und ``requests``.
"""

from __future__ import annotations

import argparse
import atexit
import os
import secrets
import signal
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping, NoReturn, Sequence

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --------------------------------------------------------------------------- #
# API-Konstanten
# --------------------------------------------------------------------------- #

API_BASE_URL = os.environ.get("RUNPOD_API_BASE_URL", "https://rest.runpod.io/v1")
CONSOLE_URL = "https://www.console.runpod.io/pods"

#: Runpod terminiert HTTP-Ports über einen Cloudflare-Proxy unter diesem Schema.
PROXY_URL_TEMPLATE = "https://{pod_id}-{port}.proxy.runpod.net"

#: Statuscodes, die der Proxy liefert, solange im Container niemand lauscht.
#: Alles andere (inkl. 401/403 hinter einem Login) gilt als "Oberfläche steht".
PROXY_NOT_READY_CODES = frozenset({502, 503, 504})

#: Mehrere Runpod-Images stellen ein Nginx vor die Anwendung, das den 502 des
#: Backends per ``error_page 502 =200 @502`` in ein HTTP **200** mit
#: Warteseite umschreibt (geprüft in /etc/nginx/nginx.conf von runpod/forge und
#: runpod/pytorch). Der Statuscode allein taugt dort nicht als
#: Bereitschaftssignal - deshalb wird zusätzlich der Seitenanfang geprüft.
PROXY_NOT_READY_TEXTS = ("Port Not Up Yet", "The port is not up yet")

#: So viele Bytes des Antwortkörpers werden für diese Prüfung gelesen.
PROBE_BODY_BYTES = 4096

BOOT_TIMEOUT_SECONDS = 1500  # inkl. Modell-Download beim ersten Start
POLL_INTERVAL_SECONDS = 5.0
PROBE_TIMEOUT_SECONDS = 6.0
TERMINATE_ATTEMPTS = 5
BOOTSTRAP_LOG = "/workspace/bootstrap.log"

#: Erlaubte Werte für ``gpuTypeIds`` gemäß OpenAPI-Spec von REST v1
#: (``https://rest.runpod.io/v1/openapi.json``, Stand 26.07.2026). Dient nur der
#: Vorabprüfung, damit ein Tippfehler nicht erst als HTTP 400 auffällt.
KNOWN_GPU_TYPE_IDS = frozenset({
    "AMD Instinct MI300X OAM",
    "NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-40GB", "NVIDIA A100-SXM4-80GB",
    "NVIDIA A40", "NVIDIA B200", "NVIDIA B300 SXM6 AC",
    "NVIDIA GeForce RTX 3070", "NVIDIA GeForce RTX 3080",
    "NVIDIA GeForce RTX 3080 Ti", "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 3090 Ti", "NVIDIA GeForce RTX 4070 Ti",
    "NVIDIA GeForce RTX 4080", "NVIDIA GeForce RTX 4080 SUPER",
    "NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 5080",
    "NVIDIA GeForce RTX 5090",
    "NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL", "NVIDIA H100 PCIe",
    "NVIDIA H200", "NVIDIA H200 NVL",
    "NVIDIA L4", "NVIDIA L40", "NVIDIA L40S",
    "NVIDIA RTX 2000 Ada Generation", "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA RTX 4000 SFF Ada Generation", "NVIDIA RTX 5000 Ada Generation",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA RTX A2000", "NVIDIA RTX A4000", "NVIDIA RTX A4500",
    "NVIDIA RTX A5000", "NVIDIA RTX A6000",
    "NVIDIA RTX PRO 4000 Blackwell", "NVIDIA RTX PRO 4500 Blackwell",
    "NVIDIA RTX PRO 5000 Blackwell",
    "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
    "Tesla V100-PCIE-16GB", "Tesla V100-SXM2-16GB",
})

# GPU-Pools statt pro Profil wiederholter Listen. Die Reihenfolge ist die
# Wunschreihenfolge; Runpod arbeitet sie mit ``gpuTypePriority="custom"``
# serverseitig ab - ein clientseitiger Fallback-Loop ist dadurch überflüssig.
#
# Wichtig: Images mit CUDA-12-Wheels (Forge = cu121, ComfyUI = cu128) laufen
# NICHT auf Blackwell-Karten (RTX 5090, RTX PRO 6000). Diese Profile bekommen
# deshalb den Ada/Ampere-Pool.
GPU_POOL_ADA = (
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA L40S",
    "NVIDIA RTX A6000",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A5000",
)
GPU_POOL_MODERN = (
    "NVIDIA GeForce RTX 5090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
    "NVIDIA L40S",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A5000",
)

#: ``allowedCudaVersions`` für Images mit CUDA-12-Wheels.
CUDA_12 = ("12.8", "12.7", "12.6", "12.5", "12.4", "12.3", "12.2", "12.1")

# --------------------------------------------------------------------------- #
# Bootstrap: Provisionierung im Container
# --------------------------------------------------------------------------- #

# POSIX-sh-Hilfsfunktionen, die jedem Bootstrap-Skript vorangestellt werden.
# ``say`` schreibt auf stderr, damit die Rückgabewerte von ``find_dir`` nicht
# verunreinigt werden (beides landet ohnehin im Bootstrap-Log).
BOOTSTRAP_HELPERS = r"""
say() { echo "[bootstrap $(date -u +%H:%M:%S)] $*" >&2; }

wait_dir() {
    deadline=$(( $(date +%s) + ${2:-1200} ))
    while [ ! -d "$1" ]; do
        if [ "$(date +%s)" -ge "$deadline" ]; then
            say "Timeout: $1 wurde nicht angelegt"
            return 1
        fi
        sleep 5
    done
    say "gefunden: $1"
}

find_dir() {
    deadline=$(( $(date +%s) + ${2:-1200} ))
    while :; do
        hit=$(find /workspace -maxdepth 6 -type d -name "$1" 2>/dev/null | head -1)
        if [ -n "$hit" ]; then
            say "gefunden: $hit"
            echo "$hit"
            return 0
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            say "Timeout: Verzeichnis '$1' nicht gefunden"
            return 1
        fi
        sleep 5
    done
}

fetch() {
    if [ -s "$2" ]; then say "vorhanden: $2"; return 0; fi
    mkdir -p "$(dirname "$2")"
    say "lade $(basename "$2") ..."
    if curl -fL --retry 5 --retry-delay 5 --retry-connrefused -o "$2.part" "$1"; then
        mv "$2.part" "$2"
        say "fertig: $2"
    else
        rm -f "$2.part"
        say "FEHLER beim Laden von $1"
        return 1
    fi
}
"""

# Verifizierte Download-Quellen (HTTP-Status und Größe geprüft am 26.07.2026).
URL_SDXL_TURBO = (
    "https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/"
    "sd_xl_turbo_1.0_fp16.safetensors"  # 6.9 GB
)
URL_LTX_VIDEO = (
    "https://huggingface.co/Lightricks/LTX-Video/resolve/main/"
    "ltxv-2b-0.9.6-distilled-04-25.safetensors"  # 6.3 GB
)
URL_T5XXL_FP8 = (
    "https://huggingface.co/Comfy-Org/mochi_preview_repackaged/resolve/main/"
    "split_files/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors"  # 5.2 GB
)
URL_ACE_STEP = (
    "https://huggingface.co/Comfy-Org/ACE-Step_ComfyUI_repackaged/resolve/main/"
    "all_in_one/ace_step_v1_3.5b.safetensors"  # 7.7 GB
)
URL_TRIPOSR = (
    "https://huggingface.co/stabilityai/TripoSR/resolve/main/model.ckpt"  # 1.7 GB
)

#: Installationspfad von ComfyUI im offiziellen Runpod-Image (siehe start.sh in
#: runpod-workers/comfyui-base).
COMFY_DIR = "/workspace/runpod-slim/ComfyUI"

BOOTSTRAP_FORGE = f"""
say "Warte auf die Forge-Installation ..."
dir=$(find_dir Stable-diffusion) || exit 1
fetch "{URL_SDXL_TURBO}" "$dir/sd_xl_turbo_1.0_fp16.safetensors" || exit 1
say "SDXL-Turbo liegt bereit - im Forge-UI oben links auf das Reload-Symbol."
"""

BOOTSTRAP_COMFY_VIDEO = f"""
wait_dir "{COMFY_DIR}/models" || exit 1
fetch "{URL_LTX_VIDEO}" "{COMFY_DIR}/models/checkpoints/ltxv-2b-0.9.6-distilled.safetensors" || exit 1
fetch "{URL_T5XXL_FP8}" "{COMFY_DIR}/models/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors" || exit 1
say "LTX-Video bereit - in ComfyUI 'Workflow > Browse Templates > Video' laden."
"""

BOOTSTRAP_COMFY_MUSIC = f"""
wait_dir "{COMFY_DIR}/models" || exit 1
fetch "{URL_ACE_STEP}" "{COMFY_DIR}/models/checkpoints/ace_step_v1_3.5b.safetensors" || exit 1
say "ACE-Step bereit - in ComfyUI 'Workflow > Browse Templates > Audio' laden."
"""

BOOTSTRAP_COMFY_3D = f"""
wait_dir "{COMFY_DIR}/custom_nodes" || exit 1
node="{COMFY_DIR}/custom_nodes/ComfyUI-Flowty-TripoSR"
if [ ! -d "$node" ]; then
    say "installiere TripoSR-Node ..."
    git clone --depth 1 https://github.com/flowtyone/ComfyUI-Flowty-TripoSR "$node" || exit 1
fi
fetch "{URL_TRIPOSR}" "{COMFY_DIR}/models/checkpoints/TripoSR-model.ckpt" || exit 1
say "TripoSR bereit - im ComfyUI-Manager auf 'Restart' klicken, dann Node laden."
"""

BOOTSTRAP_F5_TTS = """
say "installiere F5-TTS (dauert einige Minuten) ..."
pip install --no-cache-dir f5-tts || exit 1
say "starte Gradio-UI auf Port 7860 ..."
exec f5-tts_infer-gradio --host 0.0.0.0 --port 7860
"""

BOOTSTRAP_VIDEO_TOOLS = """
say "installiere Werkzeuge (ffmpeg, faster-whisper, TTS-Basis) ..."
apt-get update -qq && apt-get install -y -qq ffmpeg git-lfs
pip install --no-cache-dir faster-whisper f5-tts || exit 1
say "Basis fertig: Transkription/Übersetzung via faster-whisper, Stimme via f5-tts."
say "Lippensync-Modelle (LatentSync/Wav2Lip) bei Bedarf im JupyterLab klonen."
"""


def _ollama_bootstrap(model: str) -> str:
    """Wartet auf den Ollama-Server und lädt danach das Modell."""
    return f"""
say "warte auf den Ollama-Server ..."
i=0
while [ $i -lt 90 ]; do
    /bin/ollama list >/dev/null 2>&1 && break
    i=$((i + 1))
    sleep 2
done
say "lade Modell {model} ..."
/bin/ollama pull {model} || exit 1
say "Modell {model} ist einsatzbereit."
"""


# --------------------------------------------------------------------------- #
# Profile
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Profile:
    """Ein Produktionsprofil: Image, Ports, Hardware und Provisionierung."""

    key: str
    name: str
    image: str
    #: Port, auf dem die eigentliche Anwendung lauscht - er entscheidet über
    #: die Bereitschaft. Erst wenn er antwortet, ist das Profil nutzbar.
    primary_port: int
    disk_gb: int
    gpu_type_ids: tuple[str, ...]
    #: Zusätzlich freigeschaltete HTTP-Ports (JupyterLab, FileBrowser, ...).
    extra_ports: tuple[int, ...] = ()
    #: sh-Skript, das den KI-Stack im Container einrichtet.
    bootstrap: str = ""
    #: ``cmd`` überschreibt CMD (Images mit ENTRYPOINT + CMD /start.sh),
    #: ``entrypoint`` überschreibt ENTRYPOINT (ComfyUI, Ollama).
    inject: Literal["cmd", "entrypoint"] = "cmd"
    #: Prozess, der nach dem Bootstrap im Vordergrund laufen muss.
    start_target: str = "/start.sh"
    cuda_versions: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    #: Name der Umgebungsvariable, mit der das Image das JupyterLab-Token setzt.
    #: Die älteren Runpod-Images (Forge, oobabooga) erwarten
    #: ``JUPYTER_LAB_PASSWORD``, die neueren ``JUPYTER_PASSWORD``. Ein falscher
    #: Name führt dazu, dass JupyterLab ungeschützt oder gar nicht startet.
    jupyter_env: str = "JUPYTER_PASSWORD"
    provides: str = ""
    hint: str = ""

    @property
    def http_ports(self) -> tuple[int, ...]:
        return (self.primary_port, *self.extra_ports)


COMFYUI_IMAGE = "runpod/comfyui:1.4.4-cuda12.8"
PYTORCH_IMAGE = "runpod/pytorch:1.1.0-cu1290-torch291-ubuntu2404"
OLLAMA_IMAGE = "ollama/ollama:0.32.4"

#: Alle Images, Ports, Entrypoints und Download-URLs wurden gegen Docker Hub,
#: die Container-Registry und die Runpod-Doku geprüft (26.07.2026). Die früher
#: verwendeten Tags ``runpod/stable-diffusion:webui-3.10.0``,
#: ``runpod/stable-diffusion:comfy-ui-4.1.0`` und
#: ``runpod/text-generation-webui:1.1`` existieren auf Docker Hub nicht.
PROFILES: tuple[Profile, ...] = (
    Profile(
        key="1",
        name="Bildgenerierung (Stable Diffusion WebUI Forge)",
        image="runpod/forge:3.3.0",
        # Forge selbst lauscht auf 3001 (COMMANDLINE_ARGS="--port 3001"),
        # erreichbar ist aber nur der Nginx davor auf 3000. Nachgeprüft in
        # /etc/nginx/nginx.conf und webui-user.sh des Images.
        primary_port=3000,
        extra_ports=(8888,),
        disk_gb=60,
        gpu_type_ids=GPU_POOL_ADA,
        cuda_versions=CUDA_12,  # Image bringt Torch 2.4 + cu121 mit
        bootstrap=BOOTSTRAP_FORGE,
        jupyter_env="JUPYTER_LAB_PASSWORD",  # ältere Image-Generation
        provides="Forge-UI + vorinstalliertes Modell, dazu SDXL-Turbo",
        hint=(
            "Das Image bringt realisticVision schon mit - du kannst sofort "
            "arbeiten. SDXL-Turbo erscheint nach dem Download im Dropdown "
            "(Reload-Symbol oben links)."
        ),
    ),
    Profile(
        key="2",
        name="Cinematic Video AI & Workflows (ComfyUI + LTX-Video)",
        image=COMFYUI_IMAGE,
        primary_port=8188,
        extra_ports=(8888, 8080),
        disk_gb=100,
        gpu_type_ids=GPU_POOL_ADA,
        cuda_versions=CUDA_12,
        inject="entrypoint",
        bootstrap=BOOTSTRAP_COMFY_VIDEO,
        provides="ComfyUI + LTX-Video 2B + T5-Encoder",
    ),
    Profile(
        key="3",
        name="Social-Media Video & Schnitt (ComfyUI + LTX-Video)",
        image=COMFYUI_IMAGE,
        primary_port=8188,
        extra_ports=(8888, 8080),
        disk_gb=70,
        gpu_type_ids=GPU_POOL_ADA,
        cuda_versions=CUDA_12,
        inject="entrypoint",
        bootstrap=BOOTSTRAP_COMFY_VIDEO,
        provides="ComfyUI + LTX-Video 2B + T5-Encoder",
        hint="Weitere Nodes über den ComfyUI-Manager nachinstallieren.",
    ),
    Profile(
        key="4",
        name="Radio-Jingles & Musikproduktion (ComfyUI + ACE-Step)",
        image=COMFYUI_IMAGE,
        primary_port=8188,
        extra_ports=(8888, 8080),
        disk_gb=70,
        gpu_type_ids=GPU_POOL_ADA,
        cuda_versions=CUDA_12,
        inject="entrypoint",
        bootstrap=BOOTSTRAP_COMFY_MUSIC,
        provides="ComfyUI + ACE-Step 3.5B (Musik/Gesang aus Text)",
    ),
    Profile(
        key="5",
        name="Sprachsynthese & Voice-Cloning (F5-TTS)",
        image=PYTORCH_IMAGE,
        primary_port=7860,
        extra_ports=(8888,),
        disk_gb=50,
        gpu_type_ids=GPU_POOL_MODERN,
        bootstrap=BOOTSTRAP_F5_TTS,
        provides="F5-TTS mit Gradio-UI (Modelle laden beim ersten Lauf)",
    ),
    Profile(
        key="6",
        name="Automatisierte Präsentations-Erstellung (Ollama + Llama 3.2)",
        image=OLLAMA_IMAGE,
        primary_port=11434,
        disk_gb=50,
        gpu_type_ids=GPU_POOL_MODERN,
        inject="entrypoint",
        start_target="/bin/ollama serve",
        bootstrap=_ollama_bootstrap("llama3.2:3b"),
        provides="Ollama-API mit llama3.2:3b",
        hint="Kein UI - die HTTP-API (/api/chat) aus dem eigenen Agenten nutzen.",
    ),
    Profile(
        key="7",
        name="Interaktiver Text-Chat (Ollama + Qwen3 8B)",
        image=OLLAMA_IMAGE,
        primary_port=11434,
        disk_gb=70,
        gpu_type_ids=GPU_POOL_MODERN,
        inject="entrypoint",
        start_target="/bin/ollama serve",
        bootstrap=_ollama_bootstrap("qwen3:8b"),
        provides="Ollama-API mit qwen3:8b",
        hint="Chat z. B. per: curl <URL>/api/chat -d '{\"model\":\"qwen3:8b\",...}'",
    ),
    Profile(
        key="8",
        name="3D-Modellierung (ComfyUI + TripoSR)",
        image=COMFYUI_IMAGE,
        primary_port=8188,
        extra_ports=(8888, 8080),
        disk_gb=70,
        gpu_type_ids=GPU_POOL_ADA,
        cuda_versions=CUDA_12,
        inject="entrypoint",
        bootstrap=BOOTSTRAP_COMFY_3D,
        provides="ComfyUI + TripoSR-Node + TripoSR-Modell",
    ),
    Profile(
        key="9",
        name="Video-Übersetzung & Lippensynchronisation (Werkzeugkasten)",
        image=PYTORCH_IMAGE,
        primary_port=8888,
        disk_gb=80,
        gpu_type_ids=GPU_POOL_MODERN,
        bootstrap=BOOTSTRAP_VIDEO_TOOLS,
        provides="JupyterLab + ffmpeg + faster-whisper + F5-TTS",
        hint="Bewusst ohne Ein-Klick-UI: Lippensync-Modelle sind projektspezifisch.",
    ),
)

PROFILES_BY_KEY: Mapping[str, Profile] = {p.key: p for p in PROFILES}

#: Ports, auf denen die Runpod-Basisimages JupyterLab bzw. FileBrowser starten -
#: beide werden mit einem Zufallspasswort abgesichert, weil die Proxy-URL
#: öffentlich erreichbar ist.
JUPYTER_PORT = 8888
FILEBROWSER_PORT = 8080


# --------------------------------------------------------------------------- #
# Fehler
# --------------------------------------------------------------------------- #


class RunpodError(RuntimeError):
    """Fehler bei der Kommunikation mit der Runpod-API."""


class PodFailedError(RunpodError):
    """Der Pod hat den Start nicht überlebt oder wurde nicht bereit."""


# --------------------------------------------------------------------------- #
# Container-Start zusammenbauen
# --------------------------------------------------------------------------- #


def build_start_override(profile: Profile) -> dict[str, Any]:
    """Erzeugt ``dockerEntrypoint``/``dockerStartCmd`` für die Provisionierung.

    Muster aus ``runpod-workers/pod-template``: das Bootstrap-Skript läuft im
    Hintergrund und protokolliert in eine Datei, der eigentliche Dienst läuft
    per ``exec`` im Vordergrund und bleibt damit PID-Hauptprozess.

    Welches Feld überschrieben werden muss, hängt vom Image ab:
    ``runpod/pytorch`` und ``runpod/forge`` haben ``CMD ["/start.sh"]``,
    ``runpod/comfyui`` hat ``ENTRYPOINT ["/start.sh"]`` und ``ollama/ollama``
    hat ``ENTRYPOINT ["/bin/ollama"]``.
    """
    if not profile.bootstrap:
        return {}

    script = (
        "set -u\n"
        f"mkdir -p $(dirname {BOOTSTRAP_LOG})\n"
        f"{BOOTSTRAP_HELPERS}\n"
        f"{{\n{profile.bootstrap}\n}} >>{BOOTSTRAP_LOG} 2>&1 &\n"
        f"exec {profile.start_target}\n"
    )
    command = ["/bin/sh", "-c", script]
    if profile.inject == "entrypoint":
        return {"dockerEntrypoint": command, "dockerStartCmd": []}
    return {"dockerStartCmd": command}


def build_credentials(profile: Profile) -> dict[str, str]:
    """Zufallspasswörter für die mitgelieferten Zusatzdienste.

    Die Proxy-URLs sind öffentlich erreichbar; JupyterLab läuft in den
    Runpod-Images ohne Token und der FileBrowser mit dem dokumentierten
    Standardpasswort ``adminadmin12``. Beides wird hier ersetzt.

    ``runpod/pytorch`` startet JupyterLab überhaupt nur, wenn die Variable
    gesetzt ist (``if [[ $JUPYTER_PASSWORD ]]`` in dessen start.sh) - der
    Name muss deshalb zum Image passen, siehe ``Profile.jupyter_env``.
    """
    credentials: dict[str, str] = {}
    if JUPYTER_PORT in profile.http_ports:
        credentials[profile.jupyter_env] = secrets.token_urlsafe(18)
    if FILEBROWSER_PORT in profile.http_ports:
        credentials["FILEBROWSER_PASSWORD"] = secrets.token_urlsafe(18)
    return credentials


# --------------------------------------------------------------------------- #
# API-Client
# --------------------------------------------------------------------------- #


class RunpodClient:
    """Dünner Wrapper um die Pod-Endpunkte der Runpod-REST-API."""

    def __init__(self, api_key: str, base_url: str = API_BASE_URL) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "creative-ai-suite/2.0",
            }
        )
        # Retries bewusst nur für idempotente Methoden: ein wiederholtes
        # POST /pods würde einen zweiten Pod erzeugen und doppelt kosten.
        retry = Retry(
            total=4,
            connect=4,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "DELETE"}),
            raise_on_status=False,
        )
        self._session.mount("https://", HTTPAdapter(max_retries=retry))

    def close(self) -> None:
        self._session.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = f"{self._base_url}{path}"
        try:
            response = self._session.request(method, url, timeout=30, **kwargs)
        except requests.RequestException as exc:  # Netzwerk-/DNS-/TLS-Fehler
            raise RunpodError(f"{method} {path} fehlgeschlagen: {exc}") from exc

        if response.status_code == 401:
            raise RunpodError(
                "Runpod lehnt den API-Key ab (401). Ist RUNPOD_API_KEY aktuell "
                "und hat er Schreibrechte?"
            )
        if response.status_code >= 400:
            raise RunpodError(
                f"{method} {path} -> HTTP {response.status_code}: "
                f"{_error_detail(response)}"
            )
        return response

    def create_pod(
        self,
        *,
        profile: Profile,
        gpu_type_ids: Sequence[str],
        disk_gb: int,
        cloud_type: str,
        interruptible: bool,
        env: Mapping[str, str],
        network_volume_id: str | None = None,
    ) -> dict[str, Any]:
        """Erzeugt einen Pod und liefert das Pod-Objekt der API zurück."""
        payload: dict[str, Any] = {
            "name": f"creative-suite-{profile.key}"[:191],
            "imageName": profile.image,
            "computeType": "GPU",
            "gpuCount": 1,
            "gpuTypeIds": list(gpu_type_ids),
            # "custom" = Wunschreihenfolge strikt abarbeiten; "availability"
            # würde Runpod frei wählen lassen.
            "gpuTypePriority": "custom",
            "cloudType": cloud_type,
            "containerDiskInGb": disk_gb,
            # Kein Pod-Volume: Runpod legt sonst standardmäßig 20 GB an, die
            # auch im gestoppten Zustand abgerechnet würden.
            "volumeInGb": 0,
            "ports": [f"{port}/http" for port in profile.http_ports] + ["22/tcp"],
            "interruptible": interruptible,
            "env": dict(env),
            **build_start_override(profile),
        }
        if profile.cuda_versions:
            payload["allowedCudaVersions"] = list(profile.cuda_versions)
        if network_volume_id:
            # Das Netzwerk-Volume wird auf /workspace gemountet - genau dort
            # liegen ComfyUI, die Modelle und die Pip-Caches der Images. Damit
            # entfällt der mehrere GB große Download bei jedem weiteren Start;
            # `fetch` im Bootstrap überspringt vorhandene Dateien.
            payload["networkVolumeId"] = network_volume_id
            payload["volumeMountPath"] = "/workspace"
        return self._request("POST", "/pods", json=payload).json()

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        return self._request("GET", f"/pods/{pod_id}").json()

    def delete_pod(self, pod_id: str) -> None:
        """Löscht den Pod endgültig (HTTP 204). 404 gilt als Erfolg."""
        try:
            self._request("DELETE", f"/pods/{pod_id}")
        except RunpodError as exc:
            if "HTTP 404" in str(exc):
                return
            raise


def _error_detail(response: requests.Response) -> str:
    """Extrahiert eine lesbare Fehlermeldung aus einer API-Antwort."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:300].strip() or "(leere Antwort)"
    if isinstance(body, dict):
        for key in ("error", "message", "detail", "errors"):
            if key in body:
                return str(body[key])[:300]
    return str(body)[:300]


# --------------------------------------------------------------------------- #
# Lebenszyklus / Kostengarantie
# --------------------------------------------------------------------------- #


class PodLease:
    """Besitzt einen laufenden Pod und garantiert dessen Löschung.

    Das ist der kostenrelevante Kern des Skripts: sobald ein Pod existiert,
    wird er beim Verlassen des ``with``-Blocks gelöscht - egal ob regulär, per
    Exception, Ctrl-C oder SIGTERM. Zusätzlich hängt ein ``atexit``-Netz
    darunter, falls ein Pfad den Kontextmanager umgeht.
    """

    def __init__(self, client: RunpodClient, pod_id: str) -> None:
        self._client = client
        self.pod_id = pod_id
        self._lock = threading.Lock()
        self._released = False
        self._keep = False
        atexit.register(self._atexit_safety_net)

    def keep_alive(self) -> None:
        """Pod bewusst behalten - die Abrechnung läuft dann weiter."""
        self._keep = True

    def __enter__(self) -> "PodLease":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if self._keep:
            self._released = True
            print(
                f"\n⚠️  Pod {self.pod_id} bleibt bestehen und kostet weiter Geld.\n"
                f"    Löschen: {CONSOLE_URL}"
            )
            return False
        self.terminate()
        return False

    def terminate(self) -> None:
        """Löscht den Pod idempotent und mit Wiederholungen."""
        with self._lock:
            if self._released:
                return
            self._released = True

        print("\n🛑 Fahre Instanz herunter...")
        last_error: Exception | None = None
        for attempt in range(1, TERMINATE_ATTEMPTS + 1):
            try:
                self._client.delete_pod(self.pod_id)
            except RunpodError as exc:
                last_error = exc
                print(f"⚠️  Löschversuch {attempt}/{TERMINATE_ATTEMPTS}: {exc}")
                time.sleep(min(2**attempt, 15))
                continue
            print("💀 Instanz gelöscht. Keine weiteren Gebühren.")
            return

        # Nicht stillschweigend scheitern - hier hängt echtes Geld dran.
        print(
            "\n" + "!" * 64 + "\n"
            f"❌ POD {self.pod_id} KONNTE NICHT GELÖSCHT WERDEN - ER KOSTET WEITER!\n"
            f"   Letzter Fehler: {last_error}\n"
            f"   Sofort manuell beenden: {CONSOLE_URL}\n"
            + "!" * 64,
            file=sys.stderr,
        )

    def _atexit_safety_net(self) -> None:
        if not self._released and not self._keep:
            self.terminate()


def install_signal_handlers() -> None:
    """SIGINT/SIGTERM in eine Exception übersetzen, damit ``finally`` greift."""

    def handler(signum: int, _frame: object) -> NoReturn:
        raise KeyboardInterrupt(f"Signal {signal.Signals(signum).name} empfangen")

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handler)


# --------------------------------------------------------------------------- #
# Bereitschaftsprüfung
# --------------------------------------------------------------------------- #


def proxy_url(pod_id: str, port: int) -> str:
    return PROXY_URL_TEMPLATE.format(pod_id=pod_id, port=port)


def probe_http(session: requests.Session, url: str) -> bool:
    """True, sobald hinter dem Proxy ein *echter* Dienst antwortet.

    Zwei Stufen, weil der Statuscode allein nicht reicht:

    1. 502/503/504 und Verbindungsfehler heißen "noch nicht da" - der
       Runpod-Proxy antwortet mit 502, solange im Container niemand lauscht.
       401/403 zählen dagegen als bereit (Oberfläche steht, verlangt Login).
    2. Liefert ein Image-internes Nginx seine Warteseite mit Status 200 aus,
       erkennt nur der Seiteninhalt den Unterschied - siehe
       ``PROXY_NOT_READY_TEXTS``.
    """
    try:
        response = session.get(
            url,
            timeout=PROBE_TIMEOUT_SECONDS,
            allow_redirects=False,
            stream=True,
            headers={"User-Agent": "creative-ai-suite/healthcheck"},
        )
    except requests.RequestException:
        return False
    try:
        if response.status_code in PROXY_NOT_READY_CODES:
            return False
        if "html" not in response.headers.get("Content-Type", ""):
            return True
        head = next(response.iter_content(PROBE_BODY_BYTES), b"")
        text = head.decode("utf-8", "replace")
        return not any(marker in text for marker in PROXY_NOT_READY_TEXTS)
    except requests.RequestException:
        return False  # Abbruch beim Lesen = Dienst noch nicht stabil
    finally:
        response.close()


def wait_for_service(
    client: RunpodClient,
    lease: PodLease,
    profile: Profile,
    timeout: float,
) -> tuple[str, float]:
    """Wartet darauf, dass der Anwendungsport des Profils antwortet.

    Geprüft wird ausschließlich ``profile.primary_port``: JupyterLab wäre schon
    nach Sekunden erreichbar und würde eine Bereitschaft melden, die es noch
    nicht gibt, solange der eigentliche Stack noch installiert wird.

    Returns:
        (URL der Oberfläche, Kosten pro Stunde).

    Raises:
        PodFailedError: Pod beendet/terminiert oder Zeitlimit erreicht.
    """
    probe_session = requests.Session()  # ohne Retry-Adapter: Fehler sind hier normal
    url = proxy_url(lease.pod_id, profile.primary_port)
    start = time.monotonic()
    cost_per_hour = 0.0
    try:
        while True:
            elapsed = time.monotonic() - start
            if elapsed > timeout:
                raise PodFailedError(
                    f"Port {profile.primary_port} antwortete nach {int(elapsed)}s "
                    f"nicht. Log im Pod: {BOOTSTRAP_LOG} "
                    f"(--timeout erhöhen oder Logs in der Console prüfen)."
                )

            pod = client.get_pod(lease.pod_id)
            status = str(pod.get("desiredStatus", "UNKNOWN"))
            cost_per_hour = float(
                pod.get("adjustedCostPerHr") or pod.get("costPerHr") or 0.0
            )
            if status in ("EXITED", "TERMINATED"):
                raise PodFailedError(
                    f"Pod-Status {status} - der Container ist beendet. "
                    f"Letztes Ereignis: {pod.get('lastStatusChange') or 'unbekannt'}"
                )

            if probe_http(probe_session, url):
                print(f"\r{' ' * 78}\r", end="")
                return url, cost_per_hour

            print(
                f"\r⏳ {status} | warte auf Port {profile.primary_port} "
                f"| {int(elapsed)}s | {cost_per_hour:.3f} $/h ",
                end="",
                flush=True,
            )
            time.sleep(POLL_INTERVAL_SECONDS)
    finally:
        probe_session.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def print_menu() -> None:
    print("=" * 72)
    print("               Ultimate Creative Cloud Infrastructure")
    print("=" * 72)
    print("Bitte wähle das gewünschte Produktions-Profil:")
    for profile in PROFILES:
        print(f" [{profile.key}] {profile.name}")
        print(f"     → {profile.provides}")
    print(" [Q] Beenden")
    print("-" * 72)


def choose_profile() -> Profile:
    """Interaktive Profilauswahl; wiederholt bei Fehleingabe."""
    print_menu()
    valid = "/".join(p.key for p in PROFILES)
    while True:
        try:
            choice = input(f"Auswahl eingeben ({valid} / Q): ").strip()
        except EOFError:
            raise SystemExit("Keine Eingabe möglich - nutze --profile.")
        if choice.upper() == "Q":
            raise SystemExit("Vorgang abgebrochen.")
        if choice in PROFILES_BY_KEY:
            return PROFILES_BY_KEY[choice]
        print(f"❌ Ungültige Auswahl: {choice!r}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Startet einen Runpod-GPU-Pod für ein Kreativ-Profil, richtet den "
            "passenden KI-Stack ein und löscht den Pod danach garantiert wieder."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p", "--profile", choices=sorted(PROFILES_BY_KEY), help="Profil ohne Menü wählen"
    )
    parser.add_argument("-l", "--list", action="store_true", help="Profile anzeigen und beenden")
    parser.add_argument(
        "--cloud",
        choices=("SECURE", "COMMUNITY"),
        default="SECURE",
        help="COMMUNITY ist günstiger, SECURE zuverlässiger",
    )
    parser.add_argument(
        "--gpu",
        action="append",
        metavar="GPU_TYPE_ID",
        help="GPU-Wunsch überschreiben (mehrfach = Fallback-Reihenfolge)",
    )
    parser.add_argument("--disk", type=int, metavar="GB", help="Container-Disk überschreiben")
    parser.add_argument(
        "--spot",
        action="store_true",
        help="Interruptible/Spot-Pod: günstiger, kann jederzeit gestoppt werden",
    )
    parser.add_argument(
        "--network-volume-id",
        metavar="ID",
        default=os.environ.get("RUNPOD_NETWORK_VOLUME_ID") or None,
        help=(
            "Netzwerk-Volume auf /workspace mounten. Modelle werden dann nur "
            "einmal geladen (Speicher kostet dauerhaft ~0,07 $/GB/Monat)"
        ),
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Nur das nackte Image starten, keine Modelle/Pakete installieren",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=BOOT_TIMEOUT_SECONDS,
        metavar="SEK",
        help="Zeitlimit, bis der Anwendungsport antworten muss",
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Pod am Ende NICHT löschen (Abrechnung läuft weiter)",
    )
    return parser.parse_args(argv)


def resolve_api_key() -> str:
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "❌ Umgebungsvariable 'RUNPOD_API_KEY' nicht gefunden.\n"
            "   Setzen mit: export RUNPOD_API_KEY='dein_schlüssel'\n"
            "   Key erstellen: https://www.console.runpod.io/user/settings"
        )
    return api_key


def resolve_gpu_types(profile: Profile, override: Sequence[str] | None) -> tuple[str, ...]:
    gpu_types = tuple(override) if override else profile.gpu_type_ids
    unknown = [g for g in gpu_types if g not in KNOWN_GPU_TYPE_IDS]
    if unknown:
        raise SystemExit(
            "❌ Unbekannte GPU-Typ-ID(s): "
            + ", ".join(repr(g) for g in unknown)
            + "\n   Gültige IDs siehe https://rest.runpod.io/v1/openapi.json "
            "(PodCreateInput.gpuTypeIds)."
        )
    return gpu_types


def print_access_info(
    profile: Profile,
    pod_id: str,
    url: str,
    credentials: Mapping[str, str],
    cost_per_hour: float,
) -> None:
    """Gibt alle Zugänge samt generierter Passwörter aus."""
    print("🎉 DIE INSTANZ IST BETRIEBSBEREIT!")
    print(f"💵 Kosten     : {cost_per_hour:.3f} $/h")
    print(f"🔗 Anwendung  : {url}")
    for port in profile.extra_ports:
        label = {JUPYTER_PORT: "JupyterLab", FILEBROWSER_PORT: "FileBrowser"}.get(
            port, f"Port {port}"
        )
        print(f"🔗 {label:<11}: {proxy_url(pod_id, port)}")
    if profile.jupyter_env in credentials:
        print(f"🔑 Jupyter-Token   : {credentials[profile.jupyter_env]}")
    if "FILEBROWSER_PASSWORD" in credentials:
        print(f"🔑 FileBrowser     : admin / {credentials['FILEBROWSER_PASSWORD']}")
    if profile.bootstrap:
        print(
            f"📥 Provisionierung: {profile.provides}\n"
            f"   Große Downloads laufen im Hintergrund weiter - Log im Pod "
            f"unter {BOOTSTRAP_LOG}."
        )
    if profile.hint:
        print(f"ℹ️  {profile.hint}")


def wait_for_user_shutdown(cost_per_hour: float, started: float) -> None:
    """Blockiert, bis der Nutzer die Arbeit beendet."""
    print("-" * 72)
    print("⚠️  Das Schließen des Browsers beendet die Abrechnung NICHT!")
    try:
        input("\n👉 ENTER LÖSCHT DIE INSTANZ RESTLOS (STOPPT DIE KOSTEN) 👈")
    except EOFError:
        print("\n(kein interaktives Terminal - beende jetzt)")
    minutes = (time.monotonic() - started) / 60
    print(f"⏱️  Laufzeit ca. {minutes:.1f} min ≈ {cost_per_hour * minutes / 60:.2f} $")


def replace_bootstrap(profile: Profile) -> Profile:
    """Variante des Profils ohne Provisionierung (für ``--no-bootstrap``)."""
    return replace(
        profile,
        bootstrap="",
        provides=f"{profile.image} unverändert (ohne Provisionierung)",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list:
        print_menu()
        return 0

    api_key = resolve_api_key()
    profile = PROFILES_BY_KEY[args.profile] if args.profile else choose_profile()
    if args.no_bootstrap:
        profile = replace_bootstrap(profile)
    gpu_types = resolve_gpu_types(profile, args.gpu)
    disk_gb = args.disk or profile.disk_gb
    credentials = build_credentials(profile)
    env = {**profile.env, **credentials}

    install_signal_handlers()
    client = RunpodClient(api_key)

    print("-" * 72)
    print(f"🚀 Profil : {profile.name}")
    print(f"📦 Image  : {profile.image}")
    print(f"🧠 Stack  : {profile.provides or 'nur das Image'}")
    volume_info = (
        f"+ Netzwerk-Volume {args.network_volume_id} auf /workspace"
        if args.network_volume_id
        else "(kein persistentes Volume - Modelle werden je Start geladen)"
    )
    print(f"💾 Disk   : {disk_gb} GB {volume_info}")
    print(f"🎛️  GPU    : {' > '.join(gpu_types)}")
    print(f"☁️  Cloud  : {args.cloud}{' | SPOT' if args.spot else ''}")
    print("-" * 72)

    try:
        print("🔄 Fordere Hardware an (Runpod arbeitet die GPU-Liste selbst ab)...")
        pod = client.create_pod(
            profile=profile,
            gpu_type_ids=gpu_types,
            disk_gb=disk_gb,
            cloud_type=args.cloud,
            interruptible=args.spot,
            env=env,
            network_volume_id=args.network_volume_id,
        )
        pod_id = str(pod["id"])
        print(f"✅ Pod erstellt (ID: {pod_id})")
        print("⏳ Boot, Provisionierung und HTTP-Bereitschaft abwarten...\n")

        started = time.monotonic()
        with PodLease(client, pod_id) as lease:
            if args.keep_alive:
                lease.keep_alive()
            url, cost_per_hour = wait_for_service(client, lease, profile, args.timeout)
            print_access_info(profile, pod_id, url, credentials, cost_per_hour)
            wait_for_user_shutdown(cost_per_hour, started)
        return 0

    except KeyboardInterrupt as exc:
        print(f"\n⚠️  Abbruch ({exc}) - die Instanz wurde gelöscht.", file=sys.stderr)
        return 130
    except PodFailedError as exc:
        print(f"\n❌ {exc}", file=sys.stderr)
        return 1
    except RunpodError as exc:
        print(f"\n❌ Runpod-API: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
