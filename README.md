# LG-Vision

LangGraph agentic system leveraging VLMs for complex visual tasks. Using LangGraph v1 alpha.

## Quick Start 

Clone this repository:

```bash
git clone https://github.com/MatteoFalcioni/LG-Vision
cd LG-Vision
``` 

Copy the `.env.example` file into a `.env` file, and fill at least one of the api keys fields (you can select which provider you want to use with the `PROVIDER` env variable)

```bash
cp .env.example .env
``` 

```
FIREWORKS_API_KEY=__FIREWORKS_API_KEY__  # QWEN (ALIBABA) models https://fireworks.ai/models/fireworks/qwen2p5-vl-32b-instruct
OPENAI_API_KEY=__OPENAI_API_KEY__  # GPT
ANTHROPIC_API_KEY=__ANTHROPIC_API_KEY__  # CLAUDE

PROVIDER="OPENAI"  # Options: "OPENAI", "ANTHROPIC", "QWEN"
MODEL_DIM="SMALL" # ONLY FOR QWEN - Options: "SMALL", "LARGE"
```

create a fresh conda env with python >= 3.11:

```bash
conda create env -n LG-Vision python=3.11 -y
``` 

Install all requirements:

```bash
pip install -r requirements.txt
``` 

Enter `src/` and launch `main.py`:

```bash
cd src
python main.py
```

You will be asked to enter an image path for the model to see: you must enter the absolute path to the image you choose, for example `/home/matteo/LG-Vision/example_imgs/ortofoto_comparison_giardini_2017_2024.png` (adjust `/home/matteo/` to your path).
