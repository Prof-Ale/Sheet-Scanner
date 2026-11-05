---
title: Sheet Scanner OMR
emoji: 🎼
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# 🎼 Sheet Scanner OMR — Conversor de Partituras (Foto/PDF ➔ MusicXML, MIDI & MuseScore 4)

Aplicativo híbrido (**Web Local + Pronto para GitHub / Nuvem via Docker / Hugging Face Spaces**) para converter fotos ou arquivos PDF de partituras musicais em:

- **MuseScore 4 Nativo (`.mscz`)** — Usa automaticamente o `MuseScore4.exe` instalado no seu PC (e inclui botão **"Abrir Direto no MuseScore 4"**).
- **MusicXML (`.musicxml` e `.mxl`)** — Formato universal compatível com MuseScore, Finale, Sibelius e Dorico.
- **MIDI (`.mid`)** — Com **Player de Áudio Polifônico** integrado na tela e controle de andamento (BPM).
- **Transposição Musical Instantânea** — Transponha a partitura em semitons ou oitavas com 1 clique antes de baixar.

## 🚀 Como usar no seu Computador (1 Clique)

1. Dê dois cliques no arquivo **`iniciar_app.bat`**.
2. O navegador abrirá automaticamente em **`http://localhost:7860`**.
3. Arraste uma foto (`PNG`, `JPG`, `WEBP`) ou `PDF` de partitura — ou clique em um dos botões de exemplo (**Ode à Alegria** / **Brilha, Brilha Estrelinha**) para testar na hora!

## 🧠 3 Motores de Reconhecimento Musical (OMR)

1. **Leitor Óptico Local (OpenCV + Music21)**: Funciona 100% offline sem precisar de chave de API nem programas extras.
2. **Gemini Vision IA (Multimodal)**: Ideal para fotos tiradas com o celular, iluminação irregular ou manuscritos (basta informar uma chave gratuita do Google AI Studio na interface).
3. **Audiveris OMR Desktop**: Integração automática caso você instale o Audiveris (`C:\Program Files\Audiveris`) para partituras polifônicas densas.

## ☁️ Como publicar no GitHub / Hugging Face Spaces (Opção 3)

1. Suba esta pasta para um repositório no **GitHub** ou crie um Space gratuito no **Hugging Face Spaces** selecionando o SDK **Docker**.
2. O `Dockerfile` incluso já instala o Python, OpenCV, Music21 e MuseScore automaticamente na nuvem!
