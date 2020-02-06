local token_emb_dim = 300;
local char_num_filters = 128;
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
        "type": "single_id",
        "lowercase_tokens": true,
      },
      "characters": {
        "type": "characters",
        "min_padding_length": 3,
      }
    },
  },
  "train_data_path": "~/code/semeval_2020_task_6/data/deft_split/jsonl/train.jsonl",
  "validation_data_path": "~/code/semeval_2020_task_6/data/deft_split/jsonl/dev.jsonl",
  "model": {
    "type": "simple_tagger",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": token_emb_dim,
          "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
          "trainable": false,
        },
        "characters": {
          "type": "character_encoding",
          "embedding": {
            "embedding_dim": 16,
          },
          "encoder": {
            "type": "cnn",
            "embedding_dim": 16,
            "num_filters": char_num_filters,
            "ngram_filter_sizes": [3],
            "conv_layer_activation": "relu",
          },
        }
      },
    },
    "encoder": {
      "type": "lstm",
      "input_size": token_emb_dim + char_num_filters,
      "hidden_size": 187,
      "num_layers": 2,
      "bidirectional": true,
      "dropout": std.extVar("DROPOUT"),
    },
    "label_namespace": "tags",
    "label_encoding": "BIO",
    "calculate_span_f1": true,
    "regularizer": [
      ["encoder._module.weight*", {"type": "l2", "alpha": std.pow(10, std.parseInt(std.extVar("L2_NORM_EXP")))}],
    ],
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
    "patience": 50,
    "validation_metric": "+f1-measure-overall",
    "num_epochs": 100,
    "num_serialized_models_to_keep": 1,
    "grad_clipping": 5.0,
    "cuda_device": 0,
//    "learning_rate_scheduler": {
//      "type": "reduce_on_plateau",
//      "mode": "max",
//      "factor": 0.5,
//      "patience": std.parseInt(std.extVar("SCHEDULER_PATIENCE")),
//      "verbose": false,
//    },
  },
}