## Xavaki's notes on Nerfstudio

### Basic config anatomy
Most components have their corresponding <a target="_blank" href="https://docs.nerf.studio/en/latest/developer_guides/config.html">config</a> object. (trainers, pipelines, models, optimizers, etc.)

*Configs* hold component-specific data and also act as common interfaces, which allows them to "comunicate" with each other easily.
- This enforces a separation between the data and behaviour of a component's implementation, which is neat.
- This also facilitates modularity and scalability. 
- Most importantly, variables that reside within such config classes can have their values modified via the command line (thanks to tyro), which makes specifying and testing different training configurations really easy.

*Config* hierarchy is very organized:

<!-- <object style="width:75%;height:75%" data="../diagrams/config_hierarchy.svg?sanitize=true" type="image/svg+xml"></object> -->
![config hierarchy](../diagrams/config_hierarchy.svg)

A config object always contains: 
- Its own corresponding fields, specific to its purpose.     
- A _target field, which points to its corresponding class.
```python
# trainer.py
@dataclass
class TrainerConfig(ExperimentConfig):
"""Configuration for training regimen"""

_target: Type = field(default_factory=lambda: Trainer)
```
- A .setup() method, which is in charge of **initializing and returning** the _target field. All config classes inherit this behaviour from InstantiateConfig, which is almost at the top of the hierarchy:.
```python
# base_config.py
# Base instantiate configs
@dataclass
class InstantiateConfig(PrintableConfig):  # pylint: disable=too-few-public-methods
    """Config class for instantiating an the class specified in the _target attribute."""

    _target: Type

    def setup(self, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(self, **kwargs)
```

### Trainers

To implement a trainable method we must define a TrainerConfig object with the desired sub-component configs:

```python
# trainer.py
method_configs["vanilla-nerf"] = TrainerConfig(
    method_name="vanilla-nerf",
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=BlenderDataParserConfig(),
        ),
        model=VanillaModelConfig(_target=NeRFModel),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
        "temporal_distortion": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
    },
)

method_configs["my-method"] = ...
```
(note: ViewerConfig can also be specified, it defaults to WANDB, I think.)

This makes a lot of conceptual sense:
- A 'Trainer' trains a method/pipeline, as dictated by a specific optimizer. 
    - A <a href="https://docs.nerf.studio/en/latest/developer_guides/pipelines/index.html" target="_blank">pipeline</a> consists of a data manager and a model.
        - A data manager contains a dataparser.
        - ...

Say we want to implement a model that takes in the Blender dataset as input. We can probably recycle most of the config above, changing only specific methods in the pipeline and model objects in order to achieve the desired behavior without actually having to worry about data parsing, training iterations, etc.


### What happens during a 'ns-train' call?
<span style="color:red;font-weight:bold">Diagram</span>

(ns-train is the command we use to start or resume a particular training session.) 

##### train.py
- Tyro reads command line arguments passed by user (such as method, data dir, etc.) and instantiates the corresponding TrainerConfig object. 
- (config object self prints to terminal)
```python
# train.py
def train_loop(local_rank: int, world_size: int, config: TrainerConfig, global_rank: int = 0):
    """Main training function that sets up and runs the trainer per process

    Args:
        local_rank: current rank of process
        world_size: total number of gpus available
        config: config file specifying training regimen
    """
    _set_random_seed(config.machine.seed + global_rank)
    trainer = config.setup(local_rank=local_rank, world_size=world_size) # (1)
    trainer.setup() # <---- triggers all configs to setup() (2)
    trainer.train() # (3)
```
- (1) The trainer object is instantiated by calling the setup() method of the config class.
- (2) trainer.setup() is called, which triggers the setup of all its subcomponent (e.g pipeline, optimizers, etc.) configs, thus making everything ready to begin training.
- (3) trainer.run() is called. 

##### trainer.train()
As expected, many things happen when trainer.train() is invoked. Below is an ultra stripped down version of the actual function implementation.
```python
# trainer.py
def train(self) -> None:
        """Train the model."""
            ...
            for step in range(self._start_step, self._start_step + num_iterations): # (1)
                ...
                self.pipeline.train() # (2)
                ...
                # time the forward pass
                loss, loss_dict, metrics_dict = self.train_iteration(step) # (3)
                
                # metrics and viewer stuff
                ...
```

- (1) Train loop
- (2) The pipeline is prepared for train mode. The reason is that the Pipeline class inherits from nn.Module.
- (3) **Forward pass + backpropagation** of the pipeline:
    ```python
    # trainer.py
    def train_iteration(self, step: int) -> TRAIN_INTERATION_OUTPUT:
        ...
        _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
        ...
    ```
    Which is what actually performs the current forward step through the whole pipeline, reporting the losses back to the trainer so they can be backpropagated.


##### pipeline.get_train_loss_dict()

```python
def get_train_loss_dict(self, step: int):
    """This function gets your training loss dict. This will be responsible for
    getting the next batch of data from the DataManager and interfacing with the
    Model class, feeding the data to the model's forward function.

    Args:
        step: current iteration step to update sampler if using DDP (distributed)
    """
    ...
    ray_bundle, batch = self.datamanager.next_train(step) # (1)
    model_outputs = self.model(ray_bundle, batch) # (2)
    metrics_dict = self.model.get_metrics_dict(model_outputs, batch) # (3)
    loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict) # (4)

    return model_outputs, loss_dict, metrics_dict
```

- (1) (2) The pipeline obtains the next batch of data and runs it through the model.
- (3) (4) Metrics and losses are obtained so they can be reported back to the trainer.