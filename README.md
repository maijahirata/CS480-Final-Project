# READ ME

## Installing dependencies and setting up the virtual environment
```
python3 -m venv cocoenv
source cocoenv/bin/activate
pip install -r requirements.txt
```

## SAM Requirements
You must also download the SAM ViT-H model weights (`sam_vit_h_4b8939.pth`). 
These are not included in the repository and must be downloaded separately from the official Segment Anything GitHub page.
Place the file in the project root.
