# Challenge_plancton
This is a challenge on kaggle. 

Current best model is minimal CNN with data augmentation

## Running a job
Training a model (default is current best) : `python3 main.py train path_to_train`
Generate test predictions (default model is current best) : `python3 main.py test path_to_test`

Optional run arguments : 
       - `'--debug'` : This loads a smaller dataset for debug purposes
       - `'--model'` : specify model to use from model zoo 
              currently supported : ['custom1', 'custom2', 'minimal', 'minimal_softmax', 'minimal_dropout', 'resnet', 'resnet152', 'vgg', 'vgg19']
       - `'--loss'` : loss function for train. Default is cross entropy
              currently supported : ['f1', 'crossentropy']
       - `'--optimizer'` : training optimizer, currently only uses adam
       - `'--n_epochs'` : number of training epochs, default is 20
       - `'--valid'` : valid split ratio, default is 0.2 
       - `'--batch_size'` : data loader batch size, default is 64, but best model uses 256 
       - `'--num_workers'` : number of data loader workers, default=4 
       - `'--sampler'` : whether to use a weighted sampler for data loader
       - `'--run_name'` : used for logging and model loading purposes : 
              - default log_path is `'./logs/run_run_name/'`
              - best model is saved and loaded to/from `'./logs/run_run_name/best_model.pt'`
       - `'--preprocessing'` :  sequence of preprocessing operations : 
              - takes any sequence of arguments in ['centercrop', 'resize', 'totensor', 'invert', 'normalization', 'greyscale', 'greyscale3']
              - default is 'greyscale invert centercrop totensor' 
       - `'--augmentation'` : list of data augmentation transformations from ['flip', 'rotate', 'blur']. Default is none
