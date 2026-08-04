# Friday AI Voice Assistant 🎙️🤖

**Friday** is a production-quality, real-time AI voice assistant that runs entirely in your terminal. It listens continuously, understands natural speech, responds with a calm and professional voice, and maintains conversation context — just like Jarvis or Friday from the Iron Man movies.

---

## ✨ Features

- **🎤 Real-time Voice Interaction** — Speak naturally, get spoken responses
- **🔊 LiveKit-Powered Audio** — Voice Activity Detection (VAD), Speech-to-Text (STT), and Text-to-Speech (TTS) via LiveKit's plugin ecosystem
- **🧠 Google Gemini LLM** — Intelligent, context-aware conversations
- **💬 Conversation Memory** — Remembers what you've said during the session
- **⏱️ Streaming Responses** — Responses appear/talk as they're generated
- **✋ Speech Interruption** — Speak while Friday is talking to interrupt and ask something else
- **⌨️ Typing Fallback** — If your microphone isn't available, type instead
- **🎨 Rich Terminal UI** — Colourful, well-formatted output using Rich
- **🔧 Modular Architecture** — Clean, documented, PEP8-compliant Python
- **🚦 Graceful Shutdown** — Ctrl+C to quit cleanly

### Bonus Features
- Wake word detection ("Friday")
- Automatic reconnection to LiveKit
- Configurable STT/TTS backends

---

## 🏗️ Architecture

```
                   ┌──────────────────────┐
                   │   Terminal UI (Rich)  │
                   └──────────┬───────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐         ┌─────▼──────┐        ┌─────▼──────┐
   │  User   │         │  Friday    │        │  Gemini    │
   │ Speech  │         │ Assistant  │        │  Service   │
   └────┬────┘         └─────┬──────┘        └─────┬──────┘
        │                    │                     │
   ┌────▼────┐         ┌─────▼──────┐              │
   │  VAD    │◄────────│  Voice     │              │
   │(Silero) │         │  Pipeline  │              │
   └─────────┘         └─────┬──────┘              │
        │                    │                     │
   ┌────▼────┐         ┌─────▼──────┐              │
   │  STT    │◄────────│  Audio     │              │
   │(Deepgram)         │  Capture   │              │
   └─────────┘         └────────────┘              │
                                                   │
        ┌────────────┐        ┌────────────────────┘
        │  TTS       │        │
        │(Cartesia)  │◄───────┘
        └─────┬──────┘
              │
        ┌─────▼──────┐
        │  Audio     │
        │  Playback  │
        └────────────┘
```

### Project Structure

```
friday/
│
├── main.py                      # Entry point with CLI argument parsing
├── assistant.py                 # Main orchestrator (FridayAssistant)
├── config.py                    # Environment variable loading & validation
├── prompts.py                   # System prompt & personality definition
├── logger.py                    # Rich-themed logging & console output
├── utils.py                     # Audio conversion & text utilities
├── worker.py                    # LiveKit worker mode entrypoint
│
├── services/
│   ├── __init__.py              # Package exports
│   ├── gemini_service.py        # Gemini API integration with memory
│   ├── speech_service.py        # STT/TTS factory functions
│   ├── voice_pipeline.py        # Real-time audio pipeline
│   └── livekit_adapter.py       # Gemini LLM adapter for LiveKit worker mode
│
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .env.example                 # Example environment configuration
└── .gitignore                   # Git ignore rules
```

---

## 📋 Prerequisites

Before you begin, you'll need accounts and API keys for the following services:

| Service | Purpose | Required | Cost |
|---------|---------|----------|------|
| [LiveKit](https://cloud.livekit.io) | Voice infrastructure | Yes | Free tier available |
| [Google Gemini](https://aistudio.google.com/apikey) | AI language model | Yes | Free tier available |
| [Deepgram](https://console.deepgram.com) | Speech-to-Text | Yes (STT) | Free credits |
| [Cartesia](https://play.cartesia.ai) | Text-to-Speech | Yes (TTS) | Free tier available |

---

## 🔧 Installation

### 1. Clone the project

```bash
git clone <your-repo-url>
cd friday
```

### 2. Set up a virtual environment

#### Using `uv` (recommended — fast)
```bash
uv venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```

#### Using `venv`
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

*Or with `uv`:*
```bash
uv pip install -r requirements.txt
```

---

## ⚙️ Configuration

### 1. Set up environment variables

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
# ── LiveKit ──
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# ── Google Gemini ──
GOOGLE_API_KEY=your_gemini_api_key

# ── STT ──
STT_BACKEND=deepgram
DEEPGRAM_API_KEY=your_deepgram_api_key

# ── TTS ──
TTS_BACKEND=cartesia
CARTESIA_API_KEY=your_cartesia_api_key

# ── Logging ──
LOG_LEVEL=INFO
```

### 2. Get a Google Gemini API key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Click **Get API Key**
3. Create a new API key (free tier: 60 requests per minute)
4. Copy the key to `GOOGLE_API_KEY` in `.env`

### 3. Configure LiveKit

**Option A: LiveKit Cloud (recommended)**

1. Sign up at [cloud.livekit.io](https://cloud.livekit.io)
2. Create a new project
3. Go to **Settings → Keys** and generate an API key/secret pair
4. Copy your server URL, API key, and secret to `.env`

**Option B: Local LiveKit server**

```bash
# Install the LiveKit CLI
curl -sSL https://get.livekit.io | bash

# Start a local server
lk-server --port 7880

# Generate credentials
lk-server --create-keys
```

### 4. Get Deepgram API Key (for STT)

1. Sign up at [console.deepgram.com](https://console.deepgram.com)
2. Create an API key (free tier: $200 credit)
3. Add to `DEEPGRAM_API_KEY` in `.env`

### 5. Get Cartesia API Key (for TTS)

1. Sign up at [play.cartesia.ai](https://play.cartesia.ai)
2. Generate an API key
3. Add to `CARTESIA_API_KEY` in `.env`

---

## 🚀 Running the Assistant

### Voice Mode (default)

```bash
python main.py
```

This starts the assistant in voice mode. Speak into your microphone and Friday will respond.

### Text-Only Mode

```bash
python main.py --text
```

Use this if you don't have a microphone or want to type instead.

### Voice Mode with Wake Word

```bash
python main.py --wake-word
```

Friday will only respond when you say "Friday" first.

### LiveKit Worker Mode

```bash
python main.py --worker
```

Runs Friday as a LiveKit worker that processes audio from room participants.

---

## 🎯 Usage

When the assistant starts, you'll see:

```
==================================================
   ███████╗██████╗ ██╗██████╗  █████╗ ██╗   ██╗
   ██╔════╝██╔══██╗██║██╔══██╗██╔══██╗╚██╗ ██╔╝
   █████╗  ██████╔╝██║██║  ██║███████║ ╚████╔╝
   ██╔══╝  ██╔══██╗██║██║  ██║██╔══██║  ╚██╔╝
   ██║     ██║  ██║██║██████╔╝██║  ██║   ██║
   ╚═╝     ╚═╝  ╚═╝╚═╝╚═════╝ ╚═╝  ╚═╝   ╚═╝
   🤖  AI Voice Assistant  🤖
==================================================

Connected and listening
🎤 Listening...
```

Just speak naturally. The conversation appears in real-time:

```
You: What is artificial intelligence?
Friday: Artificial intelligence is a branch of computer science that creates
systems capable of performing tasks that typically require human intelligence.
These include learning, reasoning, problem-solving, and language understanding.

🎤 Listening...
```

Press **Ctrl+C** to exit gracefully.

---

## 🔄 Conversation Memory

Friday maintains conversation history during your session:

```
You: My name is Joe.
Friday: Nice to meet you, Boss!

You: What is my name?
Friday: Your name is Joe, Boss.
```

Memory is stored in-memory and resets when you restart the assistant.

---

## ⚠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| **"No module named 'sounddevice'"** | Install audio dependencies: `pip install sounddevice numpy` |
| **"No audio devices found"** | Check your microphone is connected. Run `python -c "import sounddevice; print(sounddevice.query_devices())"` |
| **"Missing required environment variable"** | Ensure `.env` exists and has all required API keys |
| **`SSL: CERTIFICATE_VERIFY_FAILED`** | Antivirus software (e.g. AVG "Web Shield") re-signs HTTPS traffic with a root CA Python's OpenSSL rejects. Set `FRIDAY_SSL_VERIFY=false` in `.env`. |
| **`Attempted to use an http session outside of a job context`** | LiveKit plugins need a shared aiohttp session outside the worker. `main.py` opens one automatically; don't call the pipeline from a bare script without it. |
| **Gemini API errors** | Check your `GOOGLE_API_KEY` is valid. Verify internet connection |
| **Deepgram STT errors** | Check `DEEPGRAM_API_KEY` and internet connection |
| **Cartesia TTS errors** | Check `CARTESIA_API_KEY` and internet connection |
| **LiveKit connection errors** | Verify `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` |
| **Microphone not working** | Check OS microphone permissions. Try text mode: `python main.py --text` |
| **Friday keeps interrupting herself** | Acoustic echo — the mic hears the speakers. Use headphones. A short echo-guard delay is already built in. |
| **Audio crackling/pops** | Try adjusting `block_size` in `voice_pipeline.py` |
| **High latency** | Use faster models: set `GEMINI_MODEL=gemini-2.0-flash-lite` |

> **Note for LiveKit version:** this project targets `livekit-agents` v1.x (the APIs changed significantly from v0.x). Make sure you have at least 1.0: `pip install "livekit-agents>=1.0"`.

---

## 🧪 Alternative Backends

### STT Backends
| Backend | Config Value | Required Key |
|---------|-------------|--------------|
| Deepgram | `deepgram` | `DEEPGRAM_API_KEY` |
| Google Cloud | `google` | `GOOGLE_APPLICATION_CREDENTIALS` |

### TTS Backends
| Backend | Config Value | Required Key |
|---------|-------------|--------------|
| Cartesia | `cartesia` | `CARTESIA_API_KEY` |
| ElevenLabs | `elevenlabs` | `ELEVENLABS_API_KEY` |
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Google Cloud | `google` | `GOOGLE_APPLICATION_CREDENTIALS` |

To switch backends, set the appropriate variables in `.env`.

---

## 🚀 Future Improvements

- [ ] **Wake word engine** — Use Porcupine/Picovoice for true wake word detection
- [ ] **Persistent memory** — Save conversation history across sessions using vector databases
- [ ] **Tool/function calling** — Let Friday check weather, set reminders, search the web
- [ ] **Multi-language support** — Detect and respond in the user's language
- [ ] **Custom voice models** — Train a custom TTS voice for Friday
- [ ] **WebRTC client** — Built-in web interface for remote access
- [ ] **Docker support** — Containerized deployment
- [ ] **Plugin system** — Extend Friday with community plugins

---

## 📜 License

MIT License. See `LICENSE` file for details.

---

## 🙌 Acknowledgements

- [LiveKit](https://livekit.io) — Real-time audio infrastructure
- [Google Gemini](https://deepmind.google/technologies/gemini/) — AI language model
- [Deepgram](https://deepgram.com) — Speech recognition
- [Cartesia](https://cartesia.ai) — Voice synthesis
- [Rich](https://rich.readthedocs.io) — Terminal UI toolkit

---

*Built with ❤️ for developers who want their own Jarvis.*

