<p align="center">
    <img width="70%" src="static/playground.png" alt>
</p>

# **classifier-playground**
Playground for image classification models, techniques, etc.. (with cuda and MPS support)

## **Train**

You can train your own classifier with your own dataset (image folder dataset) with the script *train.py*.

Please have a look at the *config/config.yml* file to setup your training (e.g. timm model, batch size, transformation set, etc.)

```
python train.py \
    --data-dir PATH/TO/YOUR/TRAIN/DATASET \
    --config config/config.yml \
    --output-dir PATH/TO/YOUR/OUTPUT/DIR  
```

## **Inference**

The script *inference.py* is an entry point to predict labels on your dataset. 

```
python inference.py \
    --data-dir PATH/TO/YOUR/DATASET 
    --model-dir PATH/TO/TRAIN/OUTPUT/DIR
    --config CONFIG/NAME/IN/TRAIN/OUTPUT/DIR.yml
    --ckpt CHECKPOINT/FILE/IN/OUTPUT/DIR.ckpt 
    --batch-size 128 
    --split [true|false] 
    --output OUTPUT/WITH/PREDICTION/FILENAME.json
```
:warning:

If *--split* is set to *false*, it means that the dataset is not split into classes' folder. In this case, the inference script will run only if *--data-dir* is a folder with just images in it.

## **Misclassified Analysis**

You can train 

You can run the streamlit webpage with *streamlit/pseudo_label.py*. 


PYTORCH_ENABLE_MPS_FALLBACK=1 python3 train.py --data-dir /Users/riccardomusmeci/Developer/data/enel/classification/spain/broken-insulator/broken-insulator-clean --config config/experiments/spain/ins_brok.yml --output-dir /Users/riccardomusmeci/Developer/experiments/github/classifier-playground/broken-insulator-clean/

