local bert_model = "bert-base-uncased";
local encoder_hidden_dim = 200;
local SEED = std.parseInt(std.extVar("ALLENNLP_SEED"));

{
  "numpy_seed": SEED,
  "pytorch_seed": SEED,
  "random_seed": SEED,
  "dataset_reader": {
    "type": "jsonl_reader",
    "subtasks": [2],
    "token_indexers": {
      "tokens": {
        "type": "bert-pretrained",
        "pretrained_model": bert_model,
        "do_lowercase": true,
        "use_starting_offsets": true,
        "truncate_long_sequences": false,
      },
    },
  },
  "train_data_path": "~/code/semeval_2020_task_6/data/deft_split/jsonl/train.jsonl",
  "validation_data_path": "~/code/semeval_2020_task_6/data/deft_split/jsonl/dev.jsonl",
  "model": {
    "type": "crf_tagger",
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
        "tokens": ["tokens", "tokens-offsets"],
      },
      "token_embedders": {
        "tokens": {
          "type": "bert-pretrained",
          "pretrained_model": bert_model,
        },
      },
    },
    "encoder": {
      "type": "lstm",
      "input_size": 768,
      "hidden_size": encoder_hidden_dim,
      "num_layers": 2,
      "bidirectional": true,
      "dropout": 0.3,
    },
    "label_encoding": "BIO",
    "calculate_span_f1": true,
    "verbose_metrics": false,
    "dropout": 0.0,
  },
  "iterator": {
    "type": "basic",
    "batch_size": 25,
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": "0.0016610248112331916"
    },
    "num_serialized_models_to_keep": 1,
    "patience": 20,
    "cuda_device": std.parseInt(std.extVar("ALLENNLP_DEVICE")),
    "validation_metric": "-loss",
    "num_epochs": 100,
    "grad_clipping": 5.0,
  },
}