# Lecture 3: Audio Generation

## Environment
1. make sure your base python version is `>= 3.9`

2. create virtual environment
``` bash
python -m venv ugenai4
source ugenai4/bin/activate
pip install --upgrade pip
```

3. [install pytorch with correct cuda version](https://pytorch.org/get-started/locally/)
e.g. for cuda 11.8
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. [install coqui-tts from source](https://github.com/coqui-ai/tts)
```bash
mkdir tmp; cd tmp
git clone https://github.com/coqui-ai/TTS
cd TTS
pip install -e .[all]
cd ../..
```

5. install other dependencies:
```bash
pip install -r requirements.txt
```

## Tortoise TTS fine tuning
1. Data preparation
An example of recording training data: https://youtu.be/xgvT7UnUTng?t=248

2. Fine tuning using [ai-voice-cloning](https://git.ecker.tech/mrq/ai-voice-cloning) tool
A guide for fine tuning https://youtu.be/6sTsqSQYIzs
