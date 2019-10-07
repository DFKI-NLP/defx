{
  local token_emb_dim = 50,

  "dataset_reader": {
    "type": "subtask1_reader",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true,
      },
    },
  },
  "train_data_path": "data/subtask1/split/train.tsv",
  "validation_data_path": "data/subtask1/split/dev.tsv",
  "model": {
    "type": "basic_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": token_emb_dim,
          "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.50d.txt.gz",
          "trainable": false,
        },
      },
    },
    "seq2vec_encoder": {
      "type": "cnn",
      "embedding_dim": token_emb_dim,
      "num_filters": 100,
      "ngram_filter_sizes": [1, 2, 3],
    },
    "dropout": 0.5
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
    "validation_metric": "-loss",
    "num_epochs": 100,
    "num_serialized_models_to_keep": 1,
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
