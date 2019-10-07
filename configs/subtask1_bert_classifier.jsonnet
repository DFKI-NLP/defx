{
  local bert_model = "bert-base-uncased",

  "dataset_reader": {
    "type": "subtask1_reader",
    "token_indexers": {
      "bert": {
        "type": "bert-pretrained",
        "pretrained_model": bert_model,
      }
    }
  },
  "train_data_path": "data/subtask1/split/train.tsv",
  "validation_data_path": "data/subtask1/split/dev.tsv",
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
    "cuda_device": 0,
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "mode": "min",
      "factor": 0.5,
      "patience": 3,
      "verbose": false,
    },
  },
}
