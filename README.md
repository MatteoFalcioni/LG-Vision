# LG-Vision
LangGraph agentic system leveraging VLMs for complex visual tasks

## Quick Start 

Clone this repository:

```bash
git clone https://github.com/MatteoFalcioni/LG-Vision
``` 

create a fresh conda env with python >= 3.11:

```bash
conda create env -n langgraph-multimodal python=3.11 -y
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
