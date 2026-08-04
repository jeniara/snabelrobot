# Snabelrobot V1

## Hailo AI-kamera

`ai_app.py` har kameraval och två lägen på port 8082: föremålsdetektering på
Hailo-8 (AI HAT+ 26 TOPS) och integritetssäker ansiktsdetektering. Ansiktsläget
identifierar inte personer och bedömer inte kön, ålder, humör eller egenskaper.

Installera Raspberry Pi-paketet med `sudo apt install hailo-all`. Montera HAT+
när Pi:n är avstängd, starta om och verifiera med
`hailortcli fw-control identify`. Bookworm-paketets särskilda ansiktsmodell är
bara kompilerad för Hailo-8L, så Hailo-8 använder tills vidare CPU-läget för
ansikten.

Styrning av en vajerdriven flexibel robot med Raspberry Pi, Arduino Uno,
CNC Shield V3 och GRBL 1.1h.

## Verifierad hårdvara

- Raspberry Pi 5, Raspberry Pi OS 12 (Bookworm)
- Arduino Uno på `/dev/ttyACM0`
- GRBL 1.1h, 115200 baud
- CNC Shield V3 med A4988
- NEMA23-motor på X
- 24 V motorförsörjning

Den 4 augusti 2026 verifierades fysisk X-rörelse vid både 10 och 60 mm/min.

## Säkerhet

- Ha alltid en fysisk möjlighet att bryta 24 V-matningen.
- Koppla aldrig in eller ur motorer medan 24 V är på.
- A4988 ligger nära sin praktiska gräns med en 2 A-motor. Kontrollera Vref mot
  drivmodulens shuntmotstånd och använd kylfläns/luftflöde.
- Programvarans standardgräns är 10 mm per jogg och 500 mm/min.
- `stop()` skickar GRBL jog cancel (`0x85`). `soft_reset()` skickar Ctrl-X.

## Installation på Raspberry Pi

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Användaren måste ha åtkomst till serieporten, normalt genom gruppen `dialout`.
Detta är redan verifierat för användaren `pi` på projektets Raspberry Pi.

## Kommandon

```bash
python main.py info
python main.py status
python main.py settings
python main.py jog --x 2 --speed 60
python main.py jog --y -1 --speed 30
```

## Fristående webbpanel

Starta manuellt med `.venv/bin/python webapp.py` och öppna sedan
`http://pibox.local:8080`. Webbpanelen skickar heartbeat under joggning; om
kontakten upphör skickar serverns watchdog jog cancel inom 0,75 sekunder.

För automatisk start används `deploy/snabelrobot.service` som systemd-tjänst.

## Stereokameror

`stereo_app.py` visar två Camera Module 3-flöden på port 8081. Avståndsmätning
kräver en kalibreringsfil skapad med `stereo_calibrate.py`; den fysiska baslinjen
är 3,45 cm. Kalibreringsbilder och resultat lagras lokalt under `camera_data/`
och versionshanteras inte.

All hårdvarukommunikation går genom `GrblController` i `grbl.py`. GUI och högre
logik ska aldrig skicka G-kod direkt.

## Tester

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

Testerna använder en simulerad serieport och kräver ingen ansluten motor.

## Nästa steg

1. Montera och verifiera Y-motorn.
2. Lägg till kontinuerlig knappjoggning med jog cancel när knappen släpps.
3. Bygg ett enkelt Tkinter-GUI ovanpå `GrblController`.
4. Lägg projektet i `github.com/jeniara/snabelrobot`.
