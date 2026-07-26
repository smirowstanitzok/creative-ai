# Creative Cloud Infrastructure Suite

Ein automatisiertes Python-Skript zur schnellen Bereitstellung von GPU-Instanzen auf RunPod für verschiedene kreative Medienprojekte (Bild, Audio, Video, Chat). 

Das Skript minimiert die Kosten für Gelegenheitsnutzer, indem es auf persistenten Standby-Speicher verzichtet und Instanzen nach der Nutzung restlos löscht.

## 📦 Installation

Stelle sicher, dass du Python (Version 3.8 oder höher) installiert hast. 

Kloniere dieses Repository oder lade die Dateien herunter. Installiere anschließend alle benötigten Abhängigkeiten automatisch über den Paketmanager:

```bash
pip install -r requirements.txt
```

## 🛠️ Einrichtung & Start

1. Erstelle ein Konto auf [RunPod](https://runpod.io) und generiere einen API-Key in deinen Einstellungen.
2. Setze deinen API-Key als Umgebungsvariable in deinem Terminal:

```bash
# Für Linux / macOS:
export RUNPOD_API_KEY="dein_api_schlüssel_hier"

# Für Windows (Eingabeaufforderung / CMD):
set RUNPOD_API_KEY="dein_api_schlüssel_hier"

# Für Windows (PowerShell):
\$env:RUNPOD_API_KEY="dein_api_schlüssel_hier"
```

3. Starte das Hauptskript:

```bash
python deployer.py
```

4. Wähle das gewünschte Produktions-Profil (z. B. Fotorealismus, Radio-Jingles oder Text-Chat) aus dem interaktiven Menü. 
5. Kopiere den im Terminal generierten Link in deinen Browser, um mit der Arbeit in der Web-Oberfläche zu beginnen.
6. **Wichtig:** Drücke nach Abschluss deiner Arbeit die `ENTER`-Taste im Terminal, um die Instanz zu löschen und die Abrechnung sekundengenau zu stoppen.
