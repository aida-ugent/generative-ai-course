# Lecture 2: Applications and Tools in Text Generation

## Environment

1. create virtual environment
``` bash
python -m venv ugenai2
source ugenai2/bin/activate
pip install --upgrade pip
```

2. [install pytorch with correct cuda version](https://pytorch.org/get-started/locally/)
e.g. for cuda 11.8
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. install other dependencies:
```bash
pip install -r requirements.txt
```


## Examples of text generation tools
1. enable the virtual environment in notebook. E.g., using vscode [Use pyvenv virtual environment in vscode](https://techinscribed.com/python-virtual-environment-in-vscode/)
2. execute through the notebook `1.examples.ipynb`.



## Chat with help desk example
1. enable the virtual environment.
2. scrap website https://helpdesk.ugent.be using scrapy ([documents](https://docs.scrapy.org/en/latest))
```bash
cd scrape
scrapy crawl helpdesk -s JOBDIR=crawls/helpdesk
```

3. launch notebook `2.chat_helpdesk.ipynb` and go through processing steps.

4. lunch model server from chat folder.
```bash
cd chat
CUDA_VISIBLE_DEVICES=0 python -m model_server
```

5. lunch chat server from 'chat' folder.
``` bash
CUDA_VISIBLE_DEVICES=2 chainlit run chat_server.py --port 5051
```
