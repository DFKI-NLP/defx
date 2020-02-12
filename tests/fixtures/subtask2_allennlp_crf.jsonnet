{
  local token_emb_dim = 2,
  local dropout = 0.5,

  "dataset_reader": {
    "type": "jsonl_reader",
    "subtasks": [2],
    "read_spacy_pos_tags": false,
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true,
      },
    },
  },
  "train_data_path": "tests/fixtures/jsonl_format_samples.jsonl",
  "validation_data_path": "tests/fixtures/jsonl_format_samples.jsonl",
  "model": {
    "type": "crf_tagger",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": token_emb_dim,
          "trainable": false,
        },
      },
    },
    "encoder": {
      "type": "lstm",
      "input_size": token_emb_dim,
      "hidden_size": 2,
      "num_layers": 1,
      "bidirectional": true,
      "dropout": dropout,
    },
    "label_encoding": "BIO",
    "calculate_span_f1": true,
    "dropout": dropout,
  },
  "iterator": {
    "type": "basic",
    "batch_size": 2,
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 1e-3,
    },
    "patience": 1,
    "cuda_device": -1,
    "validation_metric": "-loss",
    "num_epochs": 1,
    "grad_clipping": 5.0,
  },
}