## Xavaki's notes on Nerfstudio

### Basic config anatomy
All nerfstudio components have their corresponding <a target="_blank" href="https://docs.nerf.studio/en/latest/developer_guides/config.html">config</a> object. (trainers, pipelines, models, optimizers, etc.)

*Configs* hold component-specific data and also act as common interfaces, which allows them to "comunicate" with each other easily.
- This enforces a separation between the data and behaviour of a component's implementation, which is neat.
- This also facilitates modularity and scalability. 
- (Having everything implemented as dataclasses also works really well with *tyro*, the library in charge of providing cli interactivity.)

*Config* hierarchy is very organized:

<object style="width:75%;height:75%" data="../diagrams/config_hierarchy.svg?sanitize=true" type="image/svg+xml"></object>

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

###### train.py
- Tyro reads command line arguments passed by user (such as method, data dir, etc.) and instantiates the corresponding TrainerConfig object. 
-
- The trainer object is instantiated by calling the setup() method of the config class.