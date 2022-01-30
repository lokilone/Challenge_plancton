{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torchvision\n",
    "import torchvision.datasets as datasets                                                     \n",
    "import torchvision.transforms \n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.cm as cm\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional\n",
    "\n",
    "import os.path \n",
    "import sys"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Load data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "params = {\"padding\" : (300,300)}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "##### Defining transforms #####\n",
    "pad = torchvision.transforms.Pad(padding = params[\"padding\"], fill = 0) # pad smaller images to desired size\n",
    "centercrop = torchvision.transforms.CenterCrop(300) # Crop images to be of same 300x300\n",
    "greyscale = orchvision.transforms.Grayscale(num_output_channels=1) # Images are B/W but in 3 channels, we only need one\n",
    "\n",
    "#Compose transforms\n",
    "composed_transforms = torchvision.transforms.Compose([centercrop, greyscale,\n",
    "                torchvision.transforms.ToTensor()])\n",
    "\n",
    "#Create transformer class\n",
    "class DatasetTransformer(torch.utils.data.Dataset):\n",
    "    \n",
    "    def __init__(self, base_dataset, transforms):\n",
    "        self.base_dataset = base_dataset\n",
    "        self.transform = transforms\n",
    "\n",
    "    def __getitem__(self, index):\n",
    "        img, target = self.base_dataset[index]\n",
    "        return self.transform(img), target\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.base_dataset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "True\n",
      "ok\n",
      "ok\n"
     ]
    }
   ],
   "source": [
    "##### Loading Data #####\n",
    "print(os.path.exists(\"/mounts/Datasets1/ChallengeDeep/train/\"))\n",
    "train_path = \"/mounts/Datasets1/ChallengeDeep/train/\"\n",
    "valid_ratio = 0.2\n",
    "\n",
    "# Load learning data\n",
    "print('ok')\n",
    "train_path = \"/mounts/Datasets1/ChallengeDeep/train/\"\n",
    "dataset = datasets.ImageFolder(train_path,composed_transforms)\n",
    "print('ok')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train test split \n",
    "nb_train = int((1.0 - valid_ratio) * len(dataset))\n",
    "nb_valid =  len(dataset)-nb_train\n",
    "train_dataset, valid_dataset = torch.utils.data.dataset.random_split(dataset, [nb_train, nb_valid])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The train set contains 684315 images, in 10693 batches\n",
      "The validation set contains 171079 images, in 2674 batches\n"
     ]
    }
   ],
   "source": [
    "##### Generating Loaders #####\n",
    "num_workers = 4\n",
    "batch_size = 64\n",
    "\n",
    "# training loader\n",
    "train_loader = torch.utils.data.DataLoader(dataset = train_dataset,\n",
    "                                            batch_size = batch_size,\n",
    "                                            num_workers = num_workers,\n",
    "                                            shuffle = True)\n",
    "\n",
    "# validation loader\n",
    "valid_loader = torch.utils.data.DataLoader(dataset = valid_dataset,\n",
    "                                            batch_size = batch_size,\n",
    "                                            num_workers= num_workers,\n",
    "                                            shuffle = True)\n",
    "\n",
    "# Data Inspect\n",
    "print(\"The train set contains {} images, in {} batches\".format(len(train_loader.dataset), len(train_loader)))\n",
    "print(\"The validation set contains {} images, in {} batches\".format(len(valid_loader.dataset), len(valid_loader)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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344/+ZM/Ufe/4YYb8Nhjj5XEUdM0eDwevP7667BtG/fddx8CgYCq1srfW7JkCebMmYPGxkasXLkSp556KgYHB3HppZeqyvPrr7+Os846C7NmzYJlWfD7/fjmN7+J2267DZ2dnbjnnnvU9Eypra0N7733Hi6++GLceOON6g32nXfewbJly3D33XfjtddewwknnIAbb7wR8Xgcq1evhtvtVkl60UUX4aGHHsJrr72GG264AZZlHTIu1R5Hl8tVEsOHHnoILS0t+Pu//3vU19fDtm3ceeed8Hq9apTimWeewXe/+11cdtlluPDCC9Hf34/3338fhmHg2muvxR133AHDMLB+/XqcccYZME0Tvb29OOecc0YUx/vvvx+LFy/GxRdfjMsvvxwAsHr1agCFXASAvXv3Yu7cuZg3bx7WrVsHn8+H1157DU8//TSuueYaLFu2DI8++ijOPfdcXH755fjlL3950N1+qj2GzMWCao8jc7H6Y8hcLKj2ODIXqz+GzMWCao8jc7H6Y8hcLKj2OO6vra0N8+bNw9133w2v16viKIuCQGHnzltvvRVXXXUVpk+fjv7+fqxcuRJz5sxR95s5cybWrVuHdevWIRqNYtq0aaPKxRtvvBE+nw8bNmzAvHnzsGzZMixfvhxPPfUULrzwQtx8881obW1FV1cX1q9fjwULFuCaa65Rcfze976HSy+9FL///e/x1FNPjeyPeMg29/9v06ZN4oILLhCmaYrm5mZx1113CcuySu4jd1fY/2v/Dv+ffvqpuPjii4XX6xXhcFhcd911oq+vbySHoZ6neLcEIYbfIeHJJ58U06dPF6ZpirPPPlu89dZbB+y+kc1mxS233CKamppEOBwWf/M3fyMee+yxYXdxeOWVV8SXvvQl4fF4RGtrq/jnf/7nQx7XeO7iIKXTafHAAw+IOXPmCI/HIzwej5g3b5548MEHRSqVOuA49o/jX//1X4svfOELwuv1ihkzZoi1a9eKpqamYeOoaZpoamoSV199tejo6FBx9Hg8Qtd1YRiGACAef/xxIYQQL7/8sjjllFOEaZpi9uzZ4qWXXirZIUL+vd58801x3nnnCY/HIwKBgAAgHn74YfUaH374YXW7w+EQzc3Nwufzlfy7uu+++4Tf7xcAhK7r4tprrz3obhzVHseNGzcekIubN28WCxcuVHG84IILho2h3+8XbrdbxXH9+vUqF4PBoGhpaVF/65HEsfhvJePodrsFAHHqqaeKP/7xj0KIQi5OmTJFOJ1Ooeu60HVdOJ1OMWvWLPHhhx8KIQq5uGDBAqFpmgAg2traxI9+9CMBQKxataqmYshcrI04MherP4bMxdqII3Ox+mPIXKyNODIXqz+GzMXaieP+Pv3005JcXLt2bcnfuqurS3z9618X06ZNK8nFjo4O9Rhbt24VixYtEsFg8IjiKBXHMRQKifPPP1/lohBC7Nq1S1x//fWisbFRuFwuMWXKFPHnf/7nKheFEGLlypWira1NeL1esWjRIrFlyxYBQDzyyCOH/Ps5hDjIHDMCALz55pu44IIL8MEHH+DUU08t9+HQKDGO1Y8xrA2MY/VjDGsD41j9GMPawDhWP8awNjCOtWvsuscTERERERERERFViMP29JpItm3Dtu2D/ry4MX01E0Ign88f9OcOh+OgTd4cDkdZmugdCcaxoJrjyBgWVHMMAcZRquY4MoYF1RxDgHGUqjmOjGFBNccQYBylao4jY1hQzTEEGEep2uNYNQ65+HGCHawvmPzavn17uQ9xTMj1tgf7mjJlyiF/VukYx+qPI2NY/TEUgnGshTgyhtUfQyEYx1qII2NY/TEUgnGshTgyhtUfQyEYx1qJY7WoqJ5e7e3t6OnpOejPTzvtNLhcrgk8ovHR29t70F0/ACCRSJRs71vM7XZj9uzZ43VoY4JxLKjmODKGBdUcQ4BxlKo5joxhQTXHEGAcpWqOI2NYUM0xBBhHqZrjyBgWVHMMAcZRqvY4VouKKnoRERERERERERGNhZpYLOtwOMp9CMessaqZMoblE41GDznSciQYx/JhLlY/5mJtYC5WP+ZibWAuVj/mYm1gLla/sczFcuDujUTHuKlTp5b7EIgIzEWiSsFcJKoMzEWiylDtuciiFxERERERERER1RwWvYiIiIiIiIiIqOaw6EVERERERERERDWHRS8iIiIiIiIiIqo5LHoREREREREREVHNYdGLiIiIiIiIiIhqDoteRERERERERERUc1j0IiIiIiIiIiKimsOiFxERERERERER1RwWvYiIiIiIiIiIqOaw6EVERERERERERDWHRS8iIiIiIiIiIqo5LHoREREREREREVHNYdGLiIiIiIiIiIhqDoteRERERERERERUc1j0IiIiIiIiIiKimsOi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EhERVSsWveiYd9ttt+Hee+8d9e/Lghgwuu2D2T+ounGmFxFRgWmaGBgYgK7rXApXwWzbRkdHB8LhMIQQCAaDiEajql0DERFRLWHRi455vb296O3tLfdhUJXiRgRERAWy2BWLxWAYBvsdVqhsNosNGzYgEomoFg6jLXZt3LhxjI+OiIhobLHoRUR0FHhRR0T0OdnbK5PJwOv1lvtwaBixWAwrVqwo92EQERFNCM5hJiI6Cg6Hg4UvIqL/53Q64fP5kMlkOBOWiIiIyo4zvYiIjhIv7IiolsRiMdx///2qmfmREkLAsizoun7Eu/598skno3pOIiIiouGw6EVEdBQcDge3cieimpJIJPDAAw+U+zCIiIiIjhqXNxIRHQUhBHe7IiIiIiIiqkC8UiMiOgoseBEREREREVUmLm8kIjoKtm1DCFHuwyAiIiIiIqL9sOhFRDXBsixYllWW57Vtm83siYiIiIiIKgyLXkRUE+655x488sgjE/68tm0DAPr6+ib8uYmIiIiIiOjgWPQiopqwdetWbN26tdyHQURERERERBWCHZiJiIiIiIiIiKjmsOhFREREREREREQ1h0UvIiIiIiIiIiKqOSx6ERERERERERFRzWHRi4iIiIiIiIiIag6LXkREREREREREVHNY9CIiIiIiIiIioprDohcREREREREREdUcFr2IiIiIiIiIiKjmsOhFREREREREREQ1h0UvIiIiIiIiIiKqOSx6ERERERERERFRzWHRi4iIiIiIiIiIag6LXkREREREREREVHNY9CIiIiIiIiIioprDohcREREREREREdUcFr2IiIiIiIiIiKjmsOhFREREREREREQ1h0UvIiIiIiIiIiKqOSx6ERERERERERFRzWHRi4iIiIiIiIiIao5e7gMYC9FoFFOnTi33YRxz2tvbx+yxGMPyYRyrH2NYGxjH6scY1gbGsfoxhrWBcax+jGFtGMs4loNDCCHKfRBERERERERERERjicsbiYiIiIiIiIio5rDoRURERERERERENYdFLyIiIiIiIiIiqjksehERERERERERUc1h0YuIiIiIiIiIiGoOi15ERERERERERFRzWPQiIiIiIiIiIqKaw6IXERERERERERHVHBa9iIiIiIiIiIio5vwf9rUR824n7TwAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 1440x360 with 10 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "##### Display Some data #####\n",
    "n_samples = 10\n",
    "\n",
    "class_names = dataset.classes\n",
    "imgs, labels = next(iter(train_loader))\n",
    "\n",
    "fig=plt.figure(figsize=(20,5),facecolor='w')\n",
    "for i in range(n_samples) : \n",
    "    ax = plt.subplot(1,n_samples, i+1)\n",
    "    plt.imshow(imgs[i, 0, :, :], vmin=0, vmax=1.0, cmap=cm.gray)\n",
    "    ax.set_title(\"{}\".format(class_names[labels[0]]), fontsize=15)\n",
    "    ax.get_xaxis().set_visible(False)\n",
    "    ax.get_yaxis().set_visible(False)\n",
    "\n",
    "plt.savefig('plancton.png', bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "##### Model construction items #####\n",
    "# number of layers\n",
    "num_blocs = \n",
    "\n",
    "# A convolutional base block\n",
    "def conv_relu_maxpool():\n",
    "    return nn.sequential([nn.Conv2d(),\n",
    "        nn.ReLu(inplace=True),\n",
    "        nn.MaxPool2d(,step=2)])\n",
    "\n",
    "# A linear base block\n",
    "def linear_relu(dim_in, dim_out):\n",
    "    return nn.sequential([nn.Linear(dim_in,dim_out),\n",
    "                nn.Relu(inplace=True)])\n",
    "\n",
    "# Loss function \n",
    "class F1_Loss(nn.Module):\n",
    "    '''Calculate F1 score. Can work with gpu tensors\n",
    "    \n",
    "    The original implmentation is written by Michal Haltuf on Kaggle.\n",
    "    \n",
    "    Returns\n",
    "    -------\n",
    "    torch.Tensor\n",
    "        `ndim` == 1. epsilon <= val <= 1\n",
    "    '''\n",
    "    def __init__(self, epsilon=1e-7):\n",
    "        super().__init__()\n",
    "        self.epsilon = epsilon\n",
    "        \n",
    "    def forward(self, y_pred, y_true,):\n",
    "        assert y_pred.ndim == 2  \n",
    "        assert y_true.ndim == 1\n",
    "        y_true = nn.functional.one_hot(y_true, -1).to(torch.float32) # qui va à la place du -1\n",
    "        y_pred = nn.functional.softmax(y_pred, dim=1)\n",
    "        \n",
    "        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)\n",
    "        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)\n",
    "        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)\n",
    "        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)\n",
    "\n",
    "        precision = tp / (tp + fp + self.epsilon)\n",
    "        recall = tp / (tp + fn + self.epsilon)\n",
    "\n",
    "        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)\n",
    "        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)  # clamp ?\n",
    "        return 1 - f1.mean()\n",
    "    \n",
    "f_loss = F1_loss\n",
    "\n",
    "dummy_loss = f_loss(torch.Tensor([[-100, 10, 8]]), torch.LongTensor([1]))) # f1 test\n",
    "print(\"on calcule une loss f1 de : {}\".format(dummy_loss))\n",
    "\n",
    "# Optimizer \n",
    "optimizer = torch.optim.Adam(model.parameters())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "##### Building a Model #####\n",
    "\n",
    "class convClassifier(nn.Module):\n",
    "    \n",
    "    def __init__(self, input_size, num_classes, num_blocs):\n",
    "        super(convClassifier, self).__init__()\n",
    "        self.input_size = input_size\n",
    "        self.num_classes = num_classes \n",
    "        self.num_blocs = num_blocs\n",
    "    \n",
    "    def forward(self, x):\n",
    "        x = x.view(x.size[0], -1)\n",
    "        for i in range(self.num_blocs):\n",
    "            x = con_relu_maxpool(x)\n",
    "        y = self.classifier(x)\n",
    "        return y\n",
    "\n",
    "model = convClassifier(,,num_blocs = 2)\n",
    "\n",
    "use_gpu = torch.cuda.is_available()\n",
    "if use_gpu :\n",
    "    device = torch.device('cuda')\n",
    "else : \n",
    "    device = torch.device('cpu')\n",
    "\n",
    "model.to(device)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "##### training and testing #####\n",
    "\n",
    "# One train step\n",
    "def train(model, loader, f_loss, optimizer, device):\n",
    "\n",
    "    # enter train mode\n",
    "    model.train()\n",
    "    \n",
    "    N = 0\n",
    "    tot_loss, correct = 0.0, 0.0\n",
    "\n",
    "    for i, (inputs, targets) in enumerate(loader):\n",
    "        inputs, targets = inputs.to(device), targets.to(device)\n",
    "\n",
    "        # Compute the forward pass through the network up to the loss\n",
    "        outputs = model(inputs)\n",
    "        loss = f_loss(outputs, targets)\n",
    "\n",
    "        # Backward and optimize\n",
    "        optimizer.zero_grad()\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        \n",
    "        # Accumulate the exact number of processed samples\n",
    "        N += inputs.shape[0]\n",
    "\n",
    "        # Accumulate the loss considering\n",
    "        tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()\n",
    "\n",
    "        # For the accuracy, we compute the labels for each input image\n",
    "        predicted_targets = outputs.argmax(dim=1)\n",
    "        correct += (predicted_targets == targets).sum().item()\n",
    "        \n",
    "        print(\" Training : Loss : {:.4f}, Acc : {:.4f}\".format(tot_loss/N, correct/N))\n",
    "\n",
    "# Test\n",
    "def test(model, loader, f_loss, device):\n",
    "    \n",
    "    # We disable gradient computation \n",
    "    with torch.no_grad():\n",
    "        \n",
    "        # We enter evaluation mode\n",
    "        model.eval()\n",
    "        \n",
    "        N = 0\n",
    "        tot_loss, correct = 0.0, 0.0\n",
    "        \n",
    "        for i, (inputs, targets) in enumerate(loader):\n",
    "            inputs, targets = inputs.to(device), targets.to(device)\n",
    "\n",
    "            # Compute the forward pass, i.e. the scores for each input image\n",
    "            outputs = model(inputs)\n",
    "\n",
    "            # Accumulate the exact number of processed samples\n",
    "            N += inputs.shape[0]\n",
    "\n",
    "            # Accumulate the loss considering\n",
    "            tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()\n",
    "\n",
    "            # For the accuracy, we compute the labels for each input image\n",
    "            predicted_targets = outputs.argmax(dim=1)\n",
    "            correct += (predicted_targets == targets).sum().item()\n",
    "            \n",
    "        return tot_loss/N, correct/N"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "##### Save the best model #####\n",
    "\n",
    "def generate_unique_logpath(logdir, raw_run_name):\n",
    "    i = 0\n",
    "    while(True):\n",
    "        run_name = raw_run_name + \"_\" + str(i)\n",
    "        log_path = os.path.join(logdir, run_name)\n",
    "        if not os.path.isdir(log_path):\n",
    "            return log_path\n",
    "        i = i + 1\n",
    "\n",
    "# 1- create the logs directory if it does not exist\n",
    "top_logdir = \"./logs\"\n",
    "if not os.path.exists(top_logdir):\n",
    "    os.mkdir(top_logdir)\n",
    "    \n",
    "logdir = generate_unique_logpath(top_logdir, \"run\")\n",
    "print(\"Logging to {}\".format(logdir))\n",
    "\n",
    "class ModelCheckpoint:\n",
    "\n",
    "    def __init__(self, filepath, model):\n",
    "        self.min_loss = None\n",
    "        self.filepath = filepath\n",
    "        self.model = model\n",
    "\n",
    "    def update(self, loss):\n",
    "        if (self.min_loss is None) or (loss < self.min_loss):\n",
    "            print(\"Saving a better model\")\n",
    "            torch.save(self.model.state_dict(), self.filepath)\n",
    "            self.min_loss = loss\n",
    "\n",
    "# Define the callback object\n",
    "model_checkpoint = ModelCheckpoint(logdir + \"/best_model.pt\", model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "##### learning loop #####\n",
    "epochs = \n",
    "\n",
    "for t in range(epochs):\n",
    "    print(\"Epoch {}\".format(t))\n",
    "    train(model, train_loader, f_loss, optimizer, device)\n",
    "\n",
    "    val_loss, val_acc = test(model, valid_loader, f_loss, device)\n",
    "    print(\" Validation : Loss : {:.4f}, Acc : {:.4f}\".format(val_loss, val_acc))\n",
    "    model_checkpoint.update(val_loss)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "##### Save a summary of the run #####\n",
    "\n",
    "summary_file = open(logdir + \"/summary.txt\", 'w')\n",
    "summary_text = \"\"\"\n",
    "\n",
    "Executed command\n",
    "================\n",
    "{}\n",
    "\n",
    "Dataset\n",
    "=======\n",
    "FashionMNIST\n",
    "\n",
    "Model summary\n",
    "=============\n",
    "{}\n",
    "\n",
    "{} trainable parameters\n",
    "\n",
    "Optimizer\n",
    "========\n",
    "{}\n",
    "\n",
    "\"\"\".format(\" \".join(sys.argv), model, sum(p.numel() for p in model.parameters() if p.requires_grad), optimizer)\n",
    "summary_file.write(summary_text)\n",
    "summary_file.close()\n"
   ]
  }
 ],
 "metadata": {
  "interpreter": {
   "hash": "31f2aee4e71d21fbe5cf8b01ff0e069b9275f58929596ceb00d14d90e3e16cd6"
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}