{
  local token_emb_dim = 50,
  local char_emb_dim = 15,

  "dataset_reader": {
    "type": "jsonl_reader",
    "subtasks": [2],
    "sample_limit": 100,
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
  "train_data_path": "~/code/dfki/semeval_2020_task_6/data/deft_split/jsonl/train.jsonl",
  "model": {
    "type": "simple_tagger",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": token_emb_dim,
          "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.50d.txt.gz",
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
            "num_filters": char_emb_dim,
            "ngram_filter_sizes": [3],
            "conv_layer_activation": "relu",
          },
        }
      },
    },
    "encoder": {
      "type": "lstm",
      "input_size": token_emb_dim + char_emb_dim,
      "hidden_size": std.parseInt(std.extVar("LSTM_HIDDEN_SIZE")),
      "num_layers": std.parseInt(std.extVar("LSTM_NUM_LAYERS")),
      "bidirectional": true,
      "dropout": 0.0,
    },
    "label_namespace": "tags",
    "label_encoding": "BIO",
    "calculate_span_f1": true,
  },
  "iterator": {
    "type": "basic",
    "batch_size": std.parseInt(std.extVar("BATCH_SIZE")),
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": std.extVar("LEARNING_RATE"),
    },
    "num_epochs": 100,
    "num_serialized_models_to_keep": 0,
    "cuda_device": std.parseInt(std.extVar("CUDA_DEVICE")),
  },
}
