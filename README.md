# **tocla**
PyTorch Image Classification library with support to PyTorch Lightning and easy access to experiment with your own dataset.

## **How to install 🔨**
```
git clone https://github.com/riccardomusmeci/tocla
cd tocla
pip install .
```


## **Concepts 💡**
tocla tries to avoid writing again, again, and again (and again) the same code to train, test and make predictions with a image classification model.

tocla works in three different ways:
* fully automated with configuration files 🚀
* semi-automated with full support to PyTorch Lightning ⚡️
* I-want-to-write-my-own-code-but-also-using-tocla 🧑‍💻

### **ToclaConfiguration 📄**
With toclaConfiguration file you don't need to write any code for training an inference.

A configuration file is like the on in config/config.yaml.

## **Train**

### **Dataset Structure**
tocla dataset must have the following structure:
```
dataset
      |__train
      |       |__class_a
      |       |        |__img_1.jpg
      |       |        |__img_2.jpg
      |       |        |__ ...
      |       |__class_b
      |                |__img_1.png
      |                |__img_2.png
      |____val
              |__class_a
              |        |__img_1.jpg
              |        |__img_2.jpg
              |        |__ ...
              |__class_b
                       |__img_1.png
                       |__img_2.png

```


### **Fully Automated 🚀**
Once configuration experiment file is ready, just use tocla like this:

```python
from tocla.core import train

train(
    config_path="PATH/TO/CONFIG.YAML",
    train_data_dir="PATH/TO/TRAIN/DATA/DIR",
    val_data_dir="PATH/TO/VAL/DATA/DIR",
    output_dir="PATH/TO/OUTPUT/DIR",
    resume_from="PATH/TO/CKPT/TO/RESUME/FROM", # this is when you want to start retraining from a Lightning ckpt
)
```

### **Semi-Automated ⚡️**
tocla delivers some pre-built modules based on PyTorch-Lightning to speed up experiments.

```python
from tocla.model import create_model
from tocla.transform import Transform
from tocla.loss import create_criterion
from tocla.optimizer import create_optimizer
from tocla.lr_scheduler import create_lr_scheduler
from tocla.pl import create_callbacks
from pytorch_lightning import Trainer
from tocla.pl import from ..pl import ClassificationDataModule, ClassificationModelModule

# Setting up datamodule, model, callbacks, logger, and trainer
datamodule = ClassificationDataModule(
    train_data_dir=...,
    val_data_dir=...,
    train_transform=Transform(train=True, ...),
    val_transform=Transform(train=False, ...),
    class_map={                                    # class map for classification
        0: "class_a",                                 # class index: class name or [class_name1, class_name2]
        1: "class_b",
        2: ["class_c", "class_d"],
    },
)
model = create_model("resnet18", num_classes=3)
criterion = create_criterion("xent", ...)
optimizer = create_optimizer(params=model.parameters(), optimizer="sgd", lr=.001, ...)
lr_scheduler = create_lr_scheduler(optimizer=optimizer, ...)
pl_model = ClassificationModelModule(
    model=model,
    num_classes=3,
    loss=criterion,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    task="multiclass",
)
callbacks = create_callbacks(output_dir=..., ...)
trainer = Trainer(callbacks=callbacks, ...)

# Training
trainer.fit(model=pl_model, datamodule=datamodule)
```

### **I want to write my own code 🧑‍💻**
Use tocla `ImageFolderDataset`, `Transform`, and `create_stuff` functions to write your own training loop.

```python
from tocla.transform import Transform
from tocla.dataset import ImageFolderDataset
from tocla.model import create_model
from tocla.loss import create_loss
from tocla.optimizer import create_optimizer
from torch.utils.data import DataLoader
import torch

ImageFolderDataset(
                root_dir=self.train_data_dir,
                class_map=self.class_map,
                transform=self.train_transform,
                engine=self.engine,
            )


train_dataset = ImageFolderDataset(
    root_dir=...,
    class_map={                                    # class map for classification
        0: "class_a",                                 # class index: class name or [class_name1, class_name2]
        1: "class_b",
        2: ["class_c", "class_d"],
    },
    transform=Transform(train=True, input_size=224)
)
train_dl = DataLoader(dataset=train_dataset, batch_size=16)

model = create_model(
    model_name="resnet18",
    num_classes=3,
    pretrained=True
)
criterion = create_loss(loss="xent", label_smoothing=0.1)
optimizer = create_optimizer(params=model.parameters(), optimizer="sgd", lr=0.0005)

for epoch in range(NUM_EPOCHS):
    model.train()
    for batch in train_dl:
        optimizer.zero_grad()
        x, target = batch
        logits = model(x)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
```

## **Inference 🧐**
Also in inference mode, you can pick between "fully automated", "semi-automated", "write my own code" mode.


### **Fully Automated 🚀**
Once the train is over, you'll find a *config.yaml* file merging all the setups from different sections.

```python
from tocla.core import predict

predict(
    ckpt_path="PATH/TO/OUTPUT/DIR/checkpoints/model.ckpt",
    config_path="PATH/TO/OUTPUT/DIR/config.yaml",
    images_dir="PATH/TO/IMAGES",
    output_dir="PATH/TO/OUTPUT/DIR/predictions", # you can choose your own path
    apply_gradcam=True, # save gradcam images
    gradcam_with_preds=True, # if True, split gradcam images based on model predicitons
    layer="...", # layer to use for gradcam
)
```
