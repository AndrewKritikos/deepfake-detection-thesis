# Data Pipeline

### 1. DatasetWrapper Class
This class is used as a wrapper on the data subset in order to apply different transformations
to each image on the fly (when its aquired by the _\___getitem___\__ ). The use of the wrapepr is mandatory in order to apply different kind of transforms in training and validation subsets, because the **random_split** method makes the subsets to inherit the same transforms.

### 2. DataModule Class
This class is used to orchestrate the data preprocesing, by applying the transforms to each subset.

* It uses the **.yml** config file to get the hyperparameters of the transforms e.g: img size, batch size, data paths, normalization metrics.

* It applies the different transforms with the **_get_train_transforms** and **_get_val_transforms**.

  * For the **training** subset augmentation is applied with the use of **RandomCrop** and **RandomHorizontalFlip**. The randomness is used to prevent the model from memorizing the images (overfitting) and it is forcing it to learn the general artifacts of **AI-generated** ones.

  * For the **validation** subset randomness is removed and I only use **CenterCrop**. That is mandatory because validation must be deterministic so the model sees the same version of each image in every epoch in order to compare results and metrics.

* The **setup()** method
This method uses the **datasets.ImageFolder** in order to set the label (true, fake) of each image. It also computes the length of the subsets and splits them with a manual seed.

* Export of the **DataLoaders**
Methods **get_train_dataloader** and **get_val_dataloader** create the iterators taht will feed the model. I am using **shuffle=True** only in the train set in order to prevent the model from learning the order of the images, very important for the use of the **Stochastic Gradient Descent (SGD)**.
