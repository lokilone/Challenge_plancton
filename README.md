# Challenge_plancton
This is a challenge on kaggle. 

Current best model is minimal CNN with data augmentation

## Running a job
Training a model (default is current best) : `python3 main.py train path_to_train`<br />
Generate test predictions (default model is current best) : `python3 main.py test path_to_test`<br />

Optional run arguments : 
       * `'--debug'` : This loads a smaller dataset for debug purposes <br />
       * `'--model'` : specify model to use from model zoo <br />
              *currently supported : ['custom1', 'custom2', 'minimal', 'minimal_softmax', 'minimal_dropout', 'resnet', 'resnet152', 'vgg', 'vgg19']<br />
       * `'--loss'` : loss function for train. Default is cross entropy<br />
              currently supported : ['f1', 'crossentropy']<br />
       - `'--optimizer'` : training optimizer, currently only uses adam<br />
       - `'--n_epochs'` : number of training epochs, default is 20<br />
       - `'--valid'` : valid split ratio, default is 0.2 <br />
       - `'--batch_size'` : data loader batch size, default is 64, but best model uses 256 <br />
       - `'--num_workers'` : number of data loader workers, default=4 <br />
       - `'--sampler'` : whether to use a weighted sampler for data loader<br />
       - `'--run_name'` : used for logging and model loading purposes : <br />
              - default log_path is `'./logs/run_run_name/'`<br />
              - best model is saved and loaded to/from `'./logs/run_run_name/best_model.pt'`<br />
       - `'--preprocessing'` :  sequence of preprocessing operations : <br />
              - takes any sequence of arguments in ['centercrop', 'resize', 'totensor', 'invert', 'normalization', 'greyscale', 'greyscale3']<br />
              - default is 'greyscale invert centercrop totensor' <br />
       - `'--augmentation'` : list of data augmentation transformations from ['flip', 'rotate', 'blur']. Default is none<br />
