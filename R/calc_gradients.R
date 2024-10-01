
################################################################################
#                           Utility methods
################################################################################

# Calculate gradients ----------------------------------------------------------
# Function to calculate the gradient of the inputs w.r.t to the output
# inputs:  a list of model inputs
# model: torch model
calc_gradient <- function(predict_fun, inputs, target, return_out = FALSE,
                          output_wise = FALSE) {

  # Clone tensor
  if (is.list(inputs)) {
    inputs <- lapply(inputs, function(i) torch::torch_clone(i))
  } else {
    inputs <- torch::torch_clone(inputs)
  }

  # Set 'requires_grad' for the input tensors
  if (is.list(inputs)) {
    lapply(inputs, function(i) i$requires_grad <- TRUE)
  } else {
    inputs$requires_grad <- TRUE
  }

  # Run input through the model
  output <- predict_fun(inputs, target = target)

  # Make sure output is a list
  if (!is.list(output)) output <- list(output)

  # Calculate gradients for each output layer
  # If output_wise is TRUE, calculate gradients for each output node
  # otherwise, calculate gradients for the sum of all output nodes
  grads <- lapply(output, function(out) {
    if (output_wise) {
      grads <- lapply(seq_len(out$shape[2]), function(i_risk) {
        lapply(seq_len(out$shape[3]), function(i_time) {
          torch::autograd_grad(out[, i_risk, i_time]$sum(), inputs, retain_graph = TRUE)[[1]]
        }) |> torch::torch_stack(dim = -1)
      }) |> torch::torch_stack(dim = -2)
    } else {
      out <- out$sum()
      return(torch::autograd_grad(out, inputs)[[1]])
    }
  })

  # Delete output if not required
  if (!return_out) output <- NULL

  list(grads = grads, outs = output)
}

# Convert to torch tensor and repeat rows --------------------------------------
to_tensor <- function(x, instance, repeats = 1) {
  # Convert to list if not already
  if (!is.list(x)) {
    x <- list(x)
  }

  lapply(x, function(i) {
    if (inherits(i, "torch_tensor")) {
      res <- i[instance,, drop = FALSE]
    } else {
      res <- torch::torch_tensor(as.matrix(i[instance,,drop = FALSE]))
    }

    # Convert to tensor and repeat rows
    res <- res$repeat_interleave(repeats = as.integer(repeats), dim = 1)

    res
  })
}

# Split tensor into batches ----------------------------------------------------
split_batches <- function(inputs, batch_size, n_timepoints, n = NULL) {
  # Calculate batch size
  if (!is.null(n)) {
    batch_size <- max(1, batch_size %/% n) * n * n_timepoints
    rows_per_instance <- n * n_timepoints
  } else {
    batch_size <- batch_size * n_timepoints
    rows_per_instance <- n_timepoints
  }

  # Split inputs into batches
  total_rows <- if (is.list(inputs)) inputs[[1]]$shape[1] else inputs$shape[1]
  idx <- lapply(seq(1, total_rows, by = batch_size), function(i) {
    c(i, min(i + batch_size - 1, total_rows))
  })
  lapply(idx, function(i) {
    if (is.list(inputs)) {
      list(
        batch = lapply(inputs, function(x) x[i[1]:i[2],,drop = FALSE]),
        num = as.integer((i[2] - i[1] + 1) / rows_per_instance),
        idx = i
      )
    } else {
      list(
        batch = inputs[i[1]:i[2],,drop = FALSE],
        num = as.integer((i[2] - i[1] + 1) / rows_per_instance),
        idx = i
      )
    }
  })
}


# Add noise --------------------------------------------------------------------
add_noise <- function(inputs, orig_data, noise_level) {
  # Make sure both are lists
  if (!is.list(inputs)) inputs <- list(inputs)
  if (!is.list(orig_data)) orig_data <- list(orig_data)

  lapply(seq_along(inputs), function(i) {
    orig <- orig_data[[i]]
    names(orig) <- NULL

    # Calculate standard deviation
    std <- torch::torch_tensor(apply(orig, seq_along(dim(orig))[-1], sd))$unsqueeze(1)

    # Generate noise
    noise <- torch::torch_tensor(array(rnorm(prod(dim(inputs[[i]]))), dim = dim(inputs[[i]])))
    noise <- noise * noise_level *std

    # Add noise
    inputs[[i]] + noise
  })
}

