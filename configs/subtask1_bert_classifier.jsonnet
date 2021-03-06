local bert_model = "bert-base-uncased";
local SEED = std.parseInt(std.extVar("ALLENNLP_SEED"));

{
  "numpy_seed": SEED,
  "pytorch_seed": SEED,
  "random_seed": SEED,
  "dataset_reader": {
    "type": "subtask1_reader",
    "token_indexers": {
      "bert": {
        "type": "bert-pretrained",
        "pretrained_model": bert_model,
      }
    }
  },
  "train_data_path": "data/deft_split/subtask1_raw/train/",
  "validation_data_path": "data/deft_split/subtask1_raw/dev/",
  "model": {
    "type": "subtask1_classifier_wrapper",
    "model": {
      "type": "bert_for_classification",
      "bert_model": bert_model,
      "dropout": 0.5,
      "trainable": false,
    },
  },
  "iterator": {
    "type": "basic",
    "batch_size": 32,
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 1e-3,
    },
    "patience": 10,
    "validation_metric": "+f1-measure",
    "num_epochs": 100,
    "num_serialized_models_to_keep": 1,
    "grad_clipping": 5.0,
    "cuda_device": std.parseInt(std.extVar("ALLENNLP_DEVICE")),
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "mode": "min",
      "factor": 0.5,
      "patience": 3,
      "verbose": false,
    },
  },
}
