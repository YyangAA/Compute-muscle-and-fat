**Calculate the density and area of abdominal muscles and fat**

In this code,the system directly received raw DICOM image sequences as input  and automatically producted two quantitative indicators—cross-sectional area and density—for three anatomical regions: abdominal muscle, subcutaneous fat, and visceral fat.

**Prepare your env**
```bash
$ conda create -n compute python=3.10
$ pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
$ pip install -r requirements.txt
```

**Data**

The data you need to test should be placed in the path ./ct/volume-64/ScalarVolume_526/. The data will generate intermediate results under demo/.

**Execute commands to calculate area and density**
```bash
$ python pipeline/pipeline.py
```
