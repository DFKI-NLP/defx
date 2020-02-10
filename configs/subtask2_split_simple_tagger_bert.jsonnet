local token_emb_dim = 768;
local char_emb_dim = 15;
local char_num_filters = 128;
local dropout = 0.3767912341549483;
local bert_model = "bert-base-uncased";
local SEED = std.parseInt(std.extVar("ALLENNLP_SEED"));

{
  "numpy_seed": SEED,
  "pytorch_seed": SEED,
  "random_seed": SEED,
  "dataset_reader": {
    "type": "jsonl_reader",
    "subtasks": [2],
    "split_ner_labels": true,
    "token_indexers": {
      "tokens": {
        "type": "bert-pretrained",
        "pretrained_model": bert_model,
        "do_lowercase": true,
        "use_starting_offsets": true,
        "truncate_long_sequences": false,
      },
      "characters": {
        "type": "characters",
        "min_padding_length": 3,
      }
    },
  },
  "train_data_path": "data/deft_split/jsonl/train.jsonl",
  "validation_data_path": "data/deft_split/jsonl/dev.jsonl",
  "model": {
    "type": "subtask2-split-simple-tagger",
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
        "tokens": ["tokens", "tokens-offsets"],
        "characters": ["characters"],
      },
      "token_embedders": {
        "tokens": {
          "type": "bert-pretrained",
          "pretrained_model": bert_model,
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
      "dropout": dropout,
    },
    "modifier_loss_weight": 1.0,
//    "regularizer": [
//      ["encoder._module.weight*", {"type": "l2", "alpha": std.exp(std.parseInt("-3"))}],
//    ],
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
    "patience": 15,
    "validation_metric": "-loss",
    "num_epochs": 100,
    "num_serialized_models_to_keep": 1,
    "grad_clipping": 5.0,
    "cuda_device": std.parseInt(std.extVar("ALLENNLP_DEVICE")),
//    "learning_rate_scheduler": {
//      "type": "reduce_on_plateau",
//      "mode": "min",
//      "factor": 0.5,
//      "patience": 3,
//      "verbose": false,
//    },
  },
}