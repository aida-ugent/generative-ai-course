# Lecture 3: Image Generation

## Environment

1. create virtual environment
``` bash
python -m venv ugenai3
source ugenai3/bin/activate
pip install --upgrade pip
```

2. [install pytorch with correct cuda version](https://pytorch.org/get-started/locally/)
e.g. for cuda 11.8
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
3. source install diffusers, [required by lora training script](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/README.md)
``` bash
cd tmp
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
cd ../..
```

4. install other dependencies:
```bash
pip install -r requirements.txt
```

5. download trained LoRA model and training data using [GitHub large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)



## Examples of image generation
1. enable the virtual environment in notebook. E.g., using vscode [Use pyvenv virtual environment in vscode](https://techinscribed.com/python-virtual-environment-in-vscode/)
2. execute through the notebook `1.examples.ipynb`.

## Lora training
1. download HuggingFace script
```bash
wget https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image_lora.py
```

2. unzip training data `tmp/lora/flemish_tapestry_512.zip` or `tmp/lora/rorschach_inkblot_512.zip`

3. activate `ugenai3` environment and uncomment the relevant command, then:
```bash
sh train_lora.sh
```
