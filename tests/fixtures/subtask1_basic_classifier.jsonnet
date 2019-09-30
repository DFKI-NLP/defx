{
  local token_emb_dim = 2,

  "dataset_reader": {
    "type": "subtask1_reader",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true,
      },
    },
  },
  "train_data_path": "tests/fixtures/deft_subtask1_sample.deft",
  "validation_data_path": "tests/fixtures/deft_subtask1_sample.deft",
  "model": {
    "type": "subtask1_basic_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": token_emb_dim,
          "trainable": false,
        },
      },
    },
    "seq2vec_encoder": {
      "type": "cnn",
      "embedding_dim": token_emb_dim,
      "num_filters": 2,
      "ngram_filter_sizes": [1, 2],
    },
    "dropout": 0.5
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
    "validation_metric": "+f1-measure",
    "num_epochs": 1,
    "grad_clipping": 5.0,
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "mode": "min",
      "factor": 0.5,
      "patience": 3,
      "verbose": false,
    },
  },
}
