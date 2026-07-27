# Creative Cloud Infrastructure Suite

Ein Python-Skript, das für einen kreativen Anwendungsfall (Bild, Video, Musik,
Sprache, Chat, 3D) einen GPU-Pod auf [RunPod](https://runpod.io) startet, den
passenden KI-Stack darin **automatisch einrichtet**, den Link zur Oberfläche
ausgibt und den Pod anschließend **garantiert wieder löscht**.

Für Gelegenheitsnutzer gedacht: kein Standby-Speicher, keine Fixkosten,
sekundengenaue Abrechnung nur für die tatsächliche Laufzeit.

**Voraussetzungen in Kurzform:** Python 3.10+ · RunPod-Konto mit Guthaben
(ab ~10 $) · API-Key mit Schreibrechten. Mehr Setup ist nicht nötig — Template,
Ports, GPU-Auswahl und Löschen erledigt das Skript.

**Neu bei RunPod?** → [Was ist RunPod?](#-was-ist-runpod) ·
[Einrichten](#-runpod-einrichten-einmalig) · [Kosten](#-was-kostet-das-konkret) ·
[Stolpersteine](#-häufige-stolpersteine)

---

## 🤔 Was ist RunPod?

RunPod ist ein **GPU-Cloud-Anbieter**: du mietest stundenweise einen Rechner mit
Grafikkarte, auf dem ein Docker-Container läuft. Statt eine RTX 4090 für rund
2000 € zu kaufen, zahlst du etwa 0,34 $ pro Stunde — und nur, solange die
Instanz läuft.

Für generative KI ist das der übliche Weg: Modelle wie Stable Diffusion,
LTX-Video oder ein 8B-Sprachmodell brauchen 12–24 GB VRAM, die ein normaler
Laptop nicht hat.

### Die Begriffe, die du kennen musst

| Begriff | Bedeutung |
| --- | --- |
| **Pod** | Eine laufende Instanz: eine (oder mehrere) GPU + Docker-Container. Das ist, was dieses Skript startet und löscht. |
| **Template / Image** | Der Docker-Container, der auf dem Pod läuft — z. B. `runpod/comfyui` mit vorinstalliertem ComfyUI. |
| **Secure Cloud** | GPUs in professionellen T3/T4-Rechenzentren. Zuverlässiger, teurer. Standard in diesem Skript. |
| **Community Cloud** | GPUs von privaten Anbietern. Deutlich günstiger, dafür schwankende Verfügbarkeit. Über `--cloud COMMUNITY`. |
| **On-Demand** | Reservierte Instanz, läuft bis du sie beendest. Standard. |
| **Spot / Interruptible** | Günstiger, kann aber jederzeit von RunPod abgeschaltet werden. Über `--spot`. |
| **Serverless** | Bezahlung pro Request statt pro Stunde. Nutzt dieses Skript **nicht** — hier geht es um interaktive Oberflächen. |
| **Proxy-URL** | `https://<pod-id>-<port>.proxy.runpod.net` — so erreichst du die Weboberfläche im Pod aus dem Browser. |

### Speicher — der häufigste Kostenfehler

RunPod kennt drei Arten von Speicher. Sie unterscheiden sich vor allem darin,
was passiert, wenn gerade nichts rechnet
([Doku](https://docs.runpod.io/pods/storage/types)):

| Art | Verhalten | Preis |
| --- | --- | --- |
| **Container Disk** | Existiert nur, solange der Pod läuft. Beim Löschen ist alles weg. | 0,10 $/GB/Monat (nur während der Laufzeit) |
| **Volume Disk** | Bleibt am Pod hängen, solange dieser existiert — **auch im gestoppten Zustand**. | 0,10 $/GB/Monat laufend, 0,20 $ gestoppt |
| **Network Volume** | Unabhängige Netzwerk-Festplatte, an beliebige Pods anhängbar. | 0,07 $/GB/Monat (dauerhaft) |

Dieses Skript setzt bewusst `volumeInGb = 0` und **löscht** den Pod am Ende
(nicht „stoppen“). Damit bleibt kein Speicher zurück, der weiter abgerechnet
wird. Der Preis dafür: die Modelle werden bei jedem Start neu geladen — dagegen
hilft `--network-volume-id` (siehe unten).

### Abrechnung

- „All compute and storage charges are billed **per second**, with no fees for
  data transfer.“ ([Billing-Doku](https://docs.runpod.io/accounts-billing/billing))
- Du arbeitest mit **Prepaid-Guthaben**. Fällt es auf 0 $, stoppt RunPod deine
  laufenden Pods automatisch.
- Jedes Konto hat ein Standard-Limit von **80 $ pro Stunde** über alle
  Ressourcen — als Schutz gegen Unfälle.
- **Wichtig:** Ein *gestoppter* Pod kostet weiter Speichergebühren. Nur ein
  *gelöschter* (terminated) Pod kostet nichts. Deshalb löscht dieses Skript.

---

## ✅ RunPod einrichten (einmalig)

### 1. Konto anlegen

Auf [console.runpod.io](https://www.console.runpod.io) registrieren, E-Mail
bestätigen, Zwei-Faktor-Authentifizierung aktivieren (empfohlen — der Key kann
echtes Geld ausgeben).

### 2. Guthaben aufladen

Im Menü **Billing** eine Kreditkarte hinterlegen und Guthaben kaufen. Der
Einstieg ist laut Doku ab **10 $** möglich — bei 0,34 $/h für eine RTX 4090 in
der Community Cloud sind das rund 28 Stunden Rechenzeit.

Optional **Auto-Pay** einrichten (lädt automatisch nach, wenn das Guthaben unter
eine Schwelle fällt). Für dieses Skript nicht nötig — ohne Auto-Pay hast du eine
harte Obergrenze, was als Kostenbremse sogar angenehm ist.

### 3. API-Key erzeugen

Unter [Settings → API Keys](https://www.console.runpod.io/user/settings) einen
Key mit Berechtigung **Read/Write** anlegen
([Doku](https://docs.runpod.io/get-started/api-keys)).

> ⚠️ Der Key wird **nur einmal** angezeigt — RunPod speichert ihn nicht. Sofort
> in einen Passwort-Manager kopieren. Ein Key nur mit *Read Only* führt beim
> Start zu einem HTTP-401-Fehler, weil das Skript Pods erzeugen und löschen muss.

Der Key ist ein Zahlungsmittel: nicht in Git committen, nicht in Screenshots
zeigen. Die `.gitignore` dieses Repos schließt `.env` bereits aus.

### 4. Key im Terminal setzen

```bash
# Linux / macOS (bash/zsh)
export RUNPOD_API_KEY="rpa_xxxxxxxxxxxxxxxxxxxx"

# Windows (PowerShell)
$env:RUNPOD_API_KEY="rpa_xxxxxxxxxxxxxxxxxxxx"
```

Das gilt nur für die aktuelle Terminal-Sitzung. Dauerhaft: Zeile in
`~/.zshrc` bzw. `~/.bashrc` aufnehmen oder eine `.env`-Datei nutzen.

### 5. Optional: Netzwerk-Volume für die Modelle

Nur sinnvoll, wenn du ein Profil regelmäßig nutzt — siehe
[Netzwerk-Volumes](https://docs.runpod.io/pods/storage/create-network-volumes).
Unter [Storage](https://www.console.runpod.io/user/storage) ein Volume anlegen
(50–150 GB je Profil) und die ID übergeben.

> Ein Netzwerk-Volume liegt in **einem** Rechenzentrum. Pods können dann nur
> dort starten, was die GPU-Auswahl einschränkt. Wähle bei der Anlage also ein
> Rechenzentrum mit guter Verfügbarkeit deiner Wunsch-GPU.

### 6. Es braucht **kein** manuelles Setup in der Console

Template anlegen, Ports freigeben, GPU auswählen, Volume mounten — all das
erledigt das Skript über die API. Du brauchst in der Console nur Konto,
Guthaben und den API-Key.

---

## 📦 Installation

Python 3.10 oder neuer wird benötigt.

```bash
pip install -r requirements.txt
```

## 🚀 Start

```bash
python main.py                 # interaktives Menü
python main.py --profile 2     # direkt Profil 2 starten
python main.py --list          # Profile anzeigen
```

Ablauf: Profil wählen → Pod wird gebucht → das Skript wartet, bis die
Anwendung antwortet → Link (und ggf. Passwörter) erscheinen im Terminal →
nach der Arbeit `ENTER` drücken, um den Pod zu löschen und die Kosten zu
stoppen.

## 🎨 Profile und was automatisch bereitgestellt wird

| # | Anwendungsfall | Image | Automatisch eingerichtet | Download |
| --- | --- | --- | --- | --- |
| 1 | Bildgenerierung | `runpod/forge:3.3.0` | Forge-UI + mitgeliefertes Modell, dazu SDXL-Turbo | ~7 GB |
| 2 | Cinematic Video | `runpod/comfyui:1.4.4-cuda12.8` | ComfyUI + LTX-Video 2B + T5-Encoder | ~12 GB |
| 3 | Social-Media-Video | `runpod/comfyui:1.4.4-cuda12.8` | ComfyUI + LTX-Video 2B + T5-Encoder | ~12 GB |
| 4 | Radio-Jingles & Musik | `runpod/comfyui:1.4.4-cuda12.8` | ComfyUI + ACE-Step 3.5B | ~8 GB |
| 5 | Sprachsynthese / Voice-Cloning | `runpod/pytorch:1.1.0-cu1290-torch291-ubuntu2404` | F5-TTS inkl. Gradio-UI | pip + Modell |
| 6 | Präsentations-Agent | `ollama/ollama:0.32.4` | Ollama-API mit `llama3.2:3b` | ~2 GB |
| 7 | Interaktiver Text-Chat | `ollama/ollama:0.32.4` | Ollama-API mit `qwen3:8b` | ~5 GB |
| 8 | 3D-Modellierung | `runpod/comfyui:1.4.4-cuda12.8` | ComfyUI + TripoSR-Node + Modell | ~2 GB |
| 9 | Video-Übersetzung & Lippensync | `runpod/pytorch:1.1.0-cu1290-torch291-ubuntu2404` | JupyterLab + ffmpeg + faster-whisper + F5-TTS | pip |

Die Einrichtung läuft als Bootstrap-Skript im Container: der Dienst startet
sofort, Modelle und Pakete werden im Hintergrund geladen und protokolliert
(`/workspace/bootstrap.log`). Das Skript meldet den Pod erst als bereit, wenn
der **Anwendungsport** antwortet — nicht schon, wenn JupyterLab läuft.

**Erwartungsmanagement:** Bereit heißt „Oberfläche bedienbar“. Bei den
ComfyUI-Profilen kann der Modell-Download noch einige Minuten weiterlaufen;
das Modell erscheint danach nach einem Refresh im Dropdown. Profil 9 ist
bewusst ein Werkzeugkasten ohne Ein-Klick-UI, weil Lippensync-Pipelines
projektspezifisch sind.

## ⚙️ Optionen

| Flag | Wirkung |
| --- | --- |
| `-p, --profile N` | Profil ohne Menü wählen |
| `-l, --list` | Profile mit Stack-Beschreibung auflisten |
| `--cloud COMMUNITY` | Community Cloud statt Secure Cloud (günstiger) |
| `--spot` | Interruptible/Spot-Pod: günstiger, jederzeit unterbrechbar |
| `--gpu "NVIDIA GeForce RTX 4090"` | GPU-Wunsch überschreiben (mehrfach = Reihenfolge) |
| `--disk GB` | Container-Disk überschreiben |
| `--network-volume-id ID` | Netzwerk-Volume auf `/workspace` — Modelle nur einmal laden |
| `--no-bootstrap` | nacktes Image ohne Provisionierung starten |
| `--timeout SEK` | Wartezeit auf die Anwendung (Standard 1500 s) |
| `--keep-alive` | Pod am Ende **nicht** löschen |

### Modelle nur einmal laden

Ohne Volume werden die Modelle bei jedem Start neu geladen. Wer ein Profil
regelmäßig nutzt, legt in der RunPod-Console einmalig ein Netzwerk-Volume an
und übergibt dessen ID:

```bash
export RUNPOD_NETWORK_VOLUME_ID="vol_xxxxxxxx"
python main.py --profile 2      # zweiter Start ist in ~1 Minute bereit
```

Das Volume kostet dauerhaft Speichergebühren (~0,07 $/GB/Monat), spart aber
Wartezeit und GPU-Minuten bei jedem weiteren Start.

## 💸 Was kostet das konkret?

Stundenpreise laut [runpod.io/pricing](https://www.runpod.io/pricing)
(Stand 26.07.2026, Momentaufnahme — der tatsächliche Preis steht im Terminal):

| GPU | Community | Secure | VRAM |
| --- | --- | --- | --- |
| RTX A5000 | 0,16 $/h | 0,27 $/h | 24 GB |
| RTX 3090 | 0,22 $/h | 0,50 $/h | 24 GB |
| RTX 4090 | 0,34 $/h | 0,69 $/h | 24 GB |
| RTX 5090 | 0,69 $/h | 0,99 $/h | 32 GB |
| RTX 6000 Ada | 0,74 $/h | 0,84 $/h | 48 GB |
| L40S | 0,79 $/h | 0,99 $/h | 48 GB |

**Rechenbeispiel** — eine Stunde Videogenerierung (Profil 2, RTX 4090, Secure
Cloud, 100 GB Container-Disk):

```text
GPU            1 h × 0,69 $/h                = 0,69 $
Container-Disk 100 GB × 0,10 $/GB/Monat ÷ 730 h ≈ 0,01 $
                                              -------
Summe                                         ≈ 0,70 $
```

Mit `--cloud COMMUNITY` sind es etwa 0,35 $, mit zusätzlich `--spot` noch
weniger. Die ersten Minuten gehen für Boot und Modell-Download drauf — bei
regelmäßiger Nutzung lohnt deshalb das Netzwerk-Volume.

### Kostenlogik im Skript

- `volumeInGb = 0` — ohne `--network-volume-id` entsteht **kein** Dauerspeicher.
- Der Pod wird beim Verlassen des Skripts gelöscht: regulär, bei Exceptions,
  bei `Ctrl-C` und bei `SIGTERM`. Ein `atexit`-Netz greift zusätzlich.
- Schlägt das Löschen fehl, gibt das Skript eine unmissverständliche Warnung
  mit Pod-ID und Console-Link aus und beendet sich mit Fehlercode — es
  scheitert nie stillschweigend.
- Beim Beenden wird die tatsächliche Laufzeit samt geschätzter Kosten
  angezeigt; während des Wartens läuft der Stundenpreis mit.

Trotzdem gilt: nach Abstürzen des Rechners oder Terminals immer kurz in die
[Pod-Übersicht](https://www.console.runpod.io/pods) schauen.

## 🔒 Zugänge

Die Proxy-URLs (`https://<pod-id>-<port>.proxy.runpod.net`) sind öffentlich
erreichbar. Deshalb setzt das Skript für JupyterLab und den FileBrowser bei
jedem Start ein Zufallspasswort und gibt es im Terminal aus, anstatt die
tokenlose bzw. mit `adminadmin12` vorbelegte Standardkonfiguration der Images
zu übernehmen.

## 🩺 Häufige Stolpersteine

| Meldung / Symptom | Ursache und Lösung |
| --- | --- |
| `Runpod lehnt den API-Key ab (401)` | Key fehlt, ist widerrufen oder hat nur *Read Only*. Neuen **Read/Write**-Key erzeugen. |
| `HTTP 400: ... no instances available` o. ä. | Gerade keine passende GPU frei. `--cloud COMMUNITY`, andere `--gpu` oder später erneut versuchen. |
| `Pod-Status EXITED` direkt nach dem Start | Container ist abgestürzt — meist Image- oder CUDA-Problem. Logs in der [Console](https://www.console.runpod.io/pods) ansehen. |
| `Port ... antwortete nach N s nicht` | Modell-Download dauert länger als das Zeitlimit. `--timeout 2400` setzen oder erst mit `--no-bootstrap` prüfen, ob das Image überhaupt startet. |
| Pod verschwindet mitten in der Arbeit | Bei `--spot` normal: die Instanz wurde von RunPod zurückgeholt. Ohne `--spot` neu starten. |
| Guthaben aufgebraucht | RunPod stoppt laufende Pods bei 0 $ Guthaben. Nachladen — Achtung: gestoppte Pods kosten Speichergebühren weiter. |
| Oberfläche lädt lange / bricht ab | Der Proxy trennt nach 100 Sekunden. Lange Jobs in der UI starten, nicht über einen offenen Request. |
| Skript abgebrochen, Pod evtl. noch da | Nach einem Absturz von Terminal oder Rechner in die [Pod-Übersicht](https://www.console.runpod.io/pods) schauen. |

## 🧱 Technische Hinweise

- **API:** RunPod REST v1 (`https://rest.runpod.io/v1`). Der GraphQL-Endpunkt,
  den das `runpod`-Python-SDK intern nutzt, ist laut Release-Notes vom
  23.07.2026 abgekündigt. API v2 ist noch öffentliche Beta und wird bewusst
  noch nicht verwendet.
- **GPU-Fallback:** Die Wunsch-GPUs gehen als `gpuTypeIds`-Liste mit
  `gpuTypePriority: "custom"` an RunPod — die Ersatzsuche passiert
  serverseitig, nicht als Retry-Schleife im Skript.
- **CUDA-Grenzen:** Die Forge- (cu121) und ComfyUI-Images (cu128) laufen nicht
  auf Blackwell-Karten. Diese Profile fragen deshalb Ada/Ampere-GPUs an; die
  PyTorch- und Ollama-Profile dürfen auch die RTX 5090 nutzen.
- **HTTP-Proxy:** Cloudflare trennt Verbindungen nach 100 Sekunden. Lange
  Renderjobs deshalb asynchron in der jeweiligen UI starten, nicht über einen
  offenen HTTP-Request.
- **Ports der Images sind nachgeprüft, nicht geraten:** Bei `runpod/forge`
  lauscht die Anwendung intern auf 3001, erreichbar ist aber nur das Nginx
  davor auf **3000** — das Skript wartet deshalb auf 3000. Dasselbe Nginx
  schreibt einen `502` des Backends per `error_page 502 =200` in ein **HTTP
  200** mit Warteseite um. Der Statuscode allein taugt dort nicht als Signal,
  weshalb die Bereitschaftsprüfung zusätzlich den Seitenanfang auf „Port Not
  Up Yet“ prüft.
- **JupyterLab-Token:** Die älteren Images (Forge) erwarten
  `JUPYTER_LAB_PASSWORD`, die neueren `JUPYTER_PASSWORD`. `runpod/pytorch`
  startet JupyterLab sogar *nur*, wenn die Variable gesetzt ist. Der Name
  steht deshalb pro Profil in `Profile.jupyter_env`.
- Nur idempotente Requests (`GET`, `DELETE`) werden automatisch wiederholt —
  ein wiederholtes `POST /pods` würde einen zweiten Pod erzeugen.
