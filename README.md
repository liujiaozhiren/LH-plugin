

# LH-plugin

## Introduction
This repository showcases the LH-plugin currently. The LH-plugin has been adapted for use with the following models: TrajGAT, Neutraj, ST2Vec, and Traj2SimVec (with slight variations for each model). 
For upcoming(now finished), we will pluginize the LH-plugin to adapt it for various trajectory similarity models and enhance its usability.

### 2024.10.11: Update
We have integrated the usage of LH-plugin, and now it does not need to be customized and modified for different training methods of different models (it may cause you to misunderstand, thinking that we have made different optimizations for different models).

Now you can add a few lines of code to any model (refer to "Using in My Code") to use LH-plugin to solve the triangular inequality problem.


## Using in My Code
Next, we will introduce how to add LH-plugin to any model.

### Step 1: Decorate the Embedding Model
```python
embedding_model = EmbeddingModel(*args, **kwargs)
#-------- add the following code
#########
embedding_model = EmbeddingModelHandler(embedding_model,
                                    lh_input_size=2, # input size of the trajectory data
                                    lh_target_size=target_size, # target size of the embedding_model
                                    lh_lorentz=config.lorentz, # 1 enable, 0 disable
                                    lh_trajs=trajs, # trajectory data
                                    lh_model_type='lstm', # 'lstm' or 'transformer' 
                                    lh_sqrt=8,
                                    lh_C=1)
trainer_lh = LH_trainer(embedding_model, config.lorentz, every_epoch=3, grad_reduce=0.1, loss_cmb=5)
#########
#--------
```

### Step 2: Add the Training Handler (4 LH-plugin training)
```python
for epoch in range(epochs):
    embedding_model.train()
    print("Start training Epochs : {}".format(epoch))

    #-------- add the following code
    #########
    with trainer_lh.get_iter(epoch) as iter_lh:
        for train_str, loss_cmb in iter_lh:
    #########
    #--------
            cnt = 0
            sum_loss = 0
            with tqdm(dataloader, train_str) as tq:
                for i, batch in enumerate(tq):
                    ...... # your training code
```

### Step 3: Add the LH_grad_reduce
```python
                    ...... # your training code
                        
                    optimizer.zero_grad()
                    sum_loss.backward()
                        
                    # ---- add the following code
                    ###### 
                    iter_lh.LH_grad_reduce()
                    ######
                    # ----
                        
                    optimizer.step()
                        
                    ...... # your training code
```

### Step 4: Replace The Distance Calculation Part
```python
    ...... # in validation
    with torch.no_grad():
        embedding_model.lorentz.both_train(False, False)
        embeddings = get_validate_embedding(embedding_model, valid_loader)

        #------------- using the following code
        ##############
        hr10, hr50, hr5, ndcg, _10in50, ret_metric, extra_msg = cal_top10_acc(groundtruth_distance, embeddings,
                                                [valid_start, valid_end], embedding_model.lorentz, config.lorentz)
        ##############
        #-------------
                        
        ...... # your valid code
```


## Using in Model We Mentioned

### TrajGAT
To use the TrajGAT model, follow these steps:
```
cd TrajGAT
...envs/{env_name}/bin/python main.py -L 1 -s {sim_func} -c {city} --other_args {argsvalue}            # with LH-plugin
...envs/{env_name}/bin/python main.py -L 0 -s {sim_func} -c {city} --other_args {argsvalue}            # without LH-plugin
```

### Neutraj
To use the Neutraj model, follow these steps:
```
cd neutraj
...envs/{env_name}/bin/python train.py -L 1 -s {sim_func} -c {city} --other_args {argsvalue}           # with LH-plugin
...envs/{env_name}/bin/python train.py -L 0 -s {sim_func} -c {city} --other_args {argsvalue}           # without LH-plugin
```


### ST2Vec
To use the ST2Vec model, follow these steps:
```
cd ST2Vec
...envs/{env_name}/bin/python main.py -L 1 -c {config.yaml|config-DITA.yaml|config-*.yaml}             # with LH-plugin
...envs/{env_name}/bin/python main.py -L 0 -c {config.yaml|config-DITA.yaml|config-*.yaml}             # without LH-plugin
```

### Traj2SimVec
To use the Traj2SimVec model, follow these steps:
```
cd Traj2SimVec
...envs/{env_name}/bin/python main.py -L 1 -s {sim_func} -c {city} --other_args {argsvalue}            # with LH-plugin
...envs/{env_name}/bin/python main.py -L 0 -s {sim_func} -c {city} --other_args {argsvalue}            # without LH-plugin
```

other args : using -q to specific \gamma args 

## Data Preparation

Here we provide a small-scale demo data for validation.

To prepare your data for use with the LH-plugin, follow these instructions:

1. Place your dataset in the directory structure: `data_set/{city}/`.
2. Save your trajectory data as `data_set/{city}/trajs.pkl`.
3. Name your similarity data according to `data_set/{city}/{sim}.pkl`.
4. Modify the above file names in the config file of each model.

For preprocessing data for the ST2Vec model, please refer to the instructions provided in `ST2Vec/README.md`.
