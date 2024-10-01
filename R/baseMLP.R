


BaseMLP <- torch::nn_module(
  classname = "BaseMLP",
  initialize = function(in_features, num_nodes, batch_norm, dropout, activation,
                        out_features = 1, out_bias = FALSE) {

    # Check if in_features is a list
    if (is.list(in_features) & length(in_features) == 1) {
      in_features <- in_features[[1]]
    } else {
      # TODO
      stop("Currently only one input layer is supported for class BaseMLP.")
    }

    # Prepare num_nodes
    num_nodes <- c(in_features, num_nodes)

    # Define architecture
    net <- list()
    for (i in seq_len(length(num_nodes) - 1)) {
      net <- append(net, BaseMLP_dense(num_nodes[i], num_nodes[i + 1], batch_norm, dropout, activation))
    }

    # Add output layer
    net <- append(net, list(torch::nn_linear(num_nodes[length(num_nodes)],
                                             out_features, bias = out_bias)))

    self$net <- do.call(torch::nn_sequential, net)
  },

  forward = function(x) {
    if (is.list(x)) x <- x[[1]]
    self$net(x)
  }
)

# Create dense module
BaseMLP_dense <- torch::nn_module(
  initialize = function(in_feat, out_feat, batch_norm, dropout, activation) {
    self$linear <- torch::nn_linear(in_feat, out_feat)
    self$activation <- get_activation(activation)
    self$batch_norm <- if (batch_norm) torch::nn_batch_norm1d(out_feat) else NULL
    self$dropout <- if (dropout > 0) torch::nn_dropout(dropout) else NULL
  },

  forward = function(x) {
    x <- self$linear(x)
    x <- self$activation(x)
    if (!is.null(self$batch_norm)) x <- self$batch_norm(x)
    if (!is.null(self$dropout)) x <- self$dropout(x)
    x
  }
)

get_activation <- function(act) {
  if (act == "relu") {
    torch::nn_relu()
  }  else if (act == "linear") {
    identity()
  } else if (act == "tanh") {
    torch::nn_tanh()
  } else {
    stop("Activation '", act, "' not supported.")
  }
}

