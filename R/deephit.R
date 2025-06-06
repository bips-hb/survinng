################################################################################
#                             DeepHit
#   A Deep Learning Approach for Survival Analysis with Competing Risks
################################################################################


DeepHit <- torch::nn_module(
  classname = "DeepHit",
  initialize = function(net, time_bins, preprocess_fun = NULL,
                        postprocess_fun = NULL) {
    net$to(torch::torch_float())
    self$net <- net
    self$dtype <- torch::torch_float()
    self$time_bins <- time_bins

    if (is.null(preprocess_fun)) {
      self$preprocess_fun <- identity
    } else {
      self$preprocess_fun <- preprocess_fun
    }

    if (is.null(postprocess_fun)) {
      self$postprocess_fun <- function(x) {
        x$unsqueeze(-1)$unsqueeze(-1)
      }
    } else {
      self$postprocess_fun <- postprocess_fun
    }
  },

  # Input is a list of lists of (features, time)
  forward = function(input, target = "pmf", use_base_hazard = FALSE) {

    # Remove list if only one element
    if (is.list(input) && length(input) == 1) input <- input[[1]]

    # Calculate net output (with shape (batch_size, num_risks, time_bins)
    out <- self$net$forward(input)

    # Make sure output as an event dimension
    if(length(dim(out)) == 2) {
      out <- out$unsqueeze(2)
    }

    # Postprocess output
    out_shape <- dim(out)
    out <- torch::torch_cat(
      list(out$view(c(out_shape[1], -1)),
           torch::torch_zeros(out_shape[1], 1, dtype = self$dtype)),
      dim = 2)
    out <- out$softmax(dim = 2)[, seq_len(dim(out)[2] - 1)]$view(out_shape)

    # Calculate target
    if (target == "pmf") {
      out <- out
    } else if (target == "cif") {
      out <- out$cumsum(dim = -1)
    } else if (target == "survival") {
      out <- 1 - out$cumsum(dim = -1)$sum(dim = 2, keepdim = TRUE)
    }

    list(out)
  },

  # Calculate the gradients
  calc_gradients = function(inputs, target, return_out = FALSE, use_base_hazard = FALSE,
                            second_order = FALSE) {
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
    output <- self$forward(inputs, target = target)

    # Make sure output is a list
    if (!is.list(output)) output <- list(output)

    # Calculate gradients for each output layer
    grads <- lapply(output, function(out) {
      lapply(seq_len(out$shape[2]), function(i_risk) {
        lapply(seq_len(out$shape[3]), function(i_time) {
          if (second_order) {
            grads_1_order <- torch::autograd_grad(out[, i_risk, i_time]$sum(), inputs, create_graph = TRUE)
            n_cols <- dim(grads_1_order)[2]
            grads_2_order <- torch::torch_stack(lapply(1:n_cols, function(i) {
              torch::autograd_grad(grads_1_order[, i]$sum(), inputs, create_graph = TRUE)[[1]]
            }), dim = -1)
          } else {
            grads_1_order <- torch::autograd_grad(out[, i_risk, i_time]$sum(), inputs, retain_graph = TRUE)
            grads_2_order <- NULL
          }
          # Both grads and grads_2 are lists of tensors
          list(grads = grads_1_order, grads_2 = grads_2_order)
        }) |> list_stack(dim = -1)
      }) |> list_stack(dim = -2)
    })

    # Delete output if not required
    if (!return_out) output <- NULL

    list(grads = lapply(grads, function(x) x$grads), outs = output,
         grads_2 = lapply(grads, function(x) x$grads_2))
  },

  set_dtype = function(dtype) {
    self$dtype <- dtype
    self$net <- self$net$to(dtype)
  }
)
