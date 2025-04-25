# TTS Evaluation Framework

This is a full evaluation framework to benchmark Text-to-Speech (TTS) providers using SLSRD and LSRD distance metrics, with automatic trend analysis, MOS prediction, and side-by-side comparisons.

## ✨ Features
- Calculate SLSRD and LSRD metrics
- Predict MOS (Mean Opinion Score) automatically
- Plot trends and provider comparisons
- Auto-generate HTML and CSV reports
- Support `.wav` and `.mp3` files

## 🛠 Installation

1. Clone the repo:

```bash
git clone https://github.com/yourusername/tts-evaluation-framework.git
cd tts-evaluation-framework

pip install -r requirements.txt

reference_audio/       # Put your reference WAV or MP3 files here
synth_elevenlabs/      # Put ElevenLabs outputs here (matching filenames)
synth_openai/          # Put OpenAI TTS outputs here
synth_playht/          # Put PlayHT outputs here
synth_google/          # Put Google TTS outputs here
synth_puretalk/        # Put Puretalk TTS outputs here
synth_rime/            # Put Rime TTS output here
synth_polly            # Put Amazon Polly output here
synth_microsoft        # Put Micorsoft Azure TTS here
