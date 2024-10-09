################################################################################
#                             CoxTime model
################################################################################

DeepSurv <- torch::nn_module(
  classname = "DeepSurv",
  initialize = function(net, base_hazard,
                        preprocess_fun = NULL,
                        postprocess_fun = NULL) {
    self$net <- net
    self$base_hazard <- base_hazard
    self$t <- torch::torch_tensor(base_hazard$time)
    self$t_orig <- self$base_hazard$time

    if (is.null(preprocess_fun)) {
      self$preprocess_fun <- function(x) {
        if (is.list(x) && length(x) == 1) {
          x <- x[[1]]
        } else if (is.list(x)) {
          stop("Currently, only one input layer is supported!")
        }

        x
      }
    } else {
      self$preprocess_fun <- preprocess_fun
    }

    if (is.null(postprocess_fun)) {
      self$postprocess_fun <- function(x) {
        x <- x$unsqueeze(-1) * torch_ones(c(dim(x), length(self$t)))
        x$unsqueeze(-2)
      }
    } else {
      self$postprocess_fun <- postprocess_fun
    }
  },

  # Input is a list of lists of (features, time)
  forward = function(input, target = "survival", use_base_hazard = TRUE) {

    # Remove list if only one element
    if (is.list(input) && length(input) == 1) input <- input[[1]]

    # Calculate net output of shape (batch_size, out_features)
    out <- self$net$forward(input)

    # Get baseline hazards
    if (use_base_hazard) {
      base_hazard <- torch::torch_tensor(self$base_hazard$hazard)$reshape(c(rep(1, out$dim()), -1))
    } else {
      base_hazard <- 1
    }

    # Add pseudo time dimension, i.e. shape (batch_size, out_features, 1)
    out <- out$unsqueeze(-1)

    # Calculate target
    if (target == "hazard") {
      out <- torch::torch_exp(out) * base_hazard
    } else if (target == "cum_hazard") {
      out <- (torch::torch_exp(out) * base_hazard)$cumsum(dim = -1)
    } else if (target == "survival") {
      out <- torch::torch_exp(-(torch::torch_exp(out) * base_hazard)$cumsum(dim = -1))
    }

    list(out$unsqueeze(-2))
  },

  # Calculate gradients of a CoxTime model
  calc_gradients = function(inputs, return_out = FALSE, use_base_hazard = TRUE,
                            target = "hazard") {

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
    output <- self$forward(inputs, target = "hazard",
                           use_base_hazard = FALSE)

    # Make sure output is a list
    if (!is.list(output)) output <- list(output)

    # Calculate gradients for each output layer
    grads <- lapply(output, function(out) {
      torch::autograd_grad(out$sum(), inputs)[[1]]
    })

    # Delete output if not required
    if (!return_out) output <- NULL

    list(grads = grads, outs = output)
  }
)

CoxTime <- torch::nn_module(
  classname = "CoxTime",
  initialize = function(net, base_hazard,
                        labtrans = NULL,
                        preprocess_fun = NULL,
                        postprocess_fun = NULL) {
    self$net <- net
    self$base_hazard <- base_hazard
    self$t <- torch::torch_tensor(base_hazard$time)

    if (!is.null(labtrans)) {
      if (!is.function(labtrans$transform) || !is.function(labtrans$inv_transform)) {
        stop("Argument `labtrans` must be a list with 'transform' and 'inv_transform' functions!")
      }
      self$labtrans <- labtrans
    } else {
      self$labtrans <- list(
        transform = function(a) a,
        inv_transform = function(a) a
      )
    }
    self$t_orig <- self$labtrans$inv_transform(self$base_hazard$time)

    if (is.null(preprocess_fun)) {
      self$preprocess_fun <- function(x) {
        if (is.list(x) && length(x) == 1) {
          x <- x[[1]]
        } else if (is.list(x)) {
          stop("Currently, only one input layer is supported!")
        }
        batch_size <- x$size(1)

        # Input (batch_size, in_features) -> (batch_size * t, in_features) (replicate each row t times)
        res <- x$repeat_interleave(length(self$t), dim = 1)

        # Add time to input
        time <- torch::torch_vstack(replicate(batch_size, self$t$unsqueeze(-1)))
        torch::torch_cat(list(res, time), dim = -1)
      }
    } else {
      self$preprocess_fun <- preprocess_fun
    }

    if (is.null(postprocess_fun)) {
      self$postprocess_fun <- function(x) {
        # Output (batch_size * t, out_features) -> (batch_size, out_features, 1, t)
        x$reshape(c(-1, length(self$t), x$size(-1), 1))$movedim(2, -1)
      }
    } else {
      self$postprocess_fun <- postprocess_fun
    }
  },

  # Input is a list of lists of (features, time)
  forward = function(input, target = "survival", use_base_hazard = TRUE) {

    # Remove list if only one element
    if (is.list(input) && length(input) == 1) input <- input[[1]]

    # Calculate net output and transform to (batch_size, out_features, 1, t)
    out <- self$net$forward(input)
    out <- out$reshape(c(-1, length(self$t), out$size(-1), 1))$movedim(2, -1)

    # Get baseline hazards
    if (use_base_hazard) {
      base_hazard <- torch::torch_tensor(self$base_hazard$hazard)$unsqueeze(1)$unsqueeze(1)$unsqueeze(1)
    } else {
      base_hazard <- 1
    }

    # Calculate target
    if (target == "hazard") {
      out <- torch::torch_exp(out) * base_hazard
    } else if (target == "cum_hazard") {
      out <- (torch::torch_exp(out) * base_hazard)$cumsum(dim = -1)
    } else if (target == "survival") {
      out <- torch::torch_exp(-(torch::torch_exp(out) * base_hazard)$cumsum(dim = -1))
    }

    list(out)
  },

  # Calculate gradients of a CoxTime model
  calc_gradients = function(inputs, return_out = FALSE, use_base_hazard = TRUE,
                            target = "hazard") {

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
    output <- self$forward(inputs, target = "hazard",
                           use_base_hazard = use_base_hazard)

    # Make sure output is a list
    if (!is.list(output)) output <- list(output)

    # Calculate gradients for each output layer
    grads <- lapply(output, function(out) {
      torch::autograd_grad(out$sum(), inputs)[[1]]
    })

    # Delete output if not required
    if (!return_out) output <- NULL

    list(grads = grads, outs = output)
  }
)

#' Load a pre-trained PyCox model
#'
#' This function loads a pre-trained PyCox model from a given path.
#'
#' @param path The path to the weights of the PyCox model.
#' @param type The type of the model. Currently only "CoxTime" is supported.
#' @param architecture The architecture of the model. If NULL, the default
#' architecture is used. It needs to fit to the weights given in path.

load_pycox_model <- function(path, in_features, base_hazard, type = "CoxTime", architecture = NULL,
                             num_nodes = c(32, 32), batch_norm = TRUE, dropout = 0.1,
                             activation = "relu", output_activation = "linear") {

  # Check arguments
  assertFileExists(path)
  assertDataFrame(as.data.frame(base_hazard))
  assertSubset(c("time", "baseline_hazards"), names(base_hazard))
  assertCharacter(type, ignore.case = TRUE)
  type <- tolower(type)
  assertChoice(type, c("coxtime"))
  assert(check_true(torch::is_nn_module(architecture) | is.null(architecture)),
         "architecture")

  # Rebuild architecture and load state dict
  if (is.null(architecture)) {
    net <- BaseMLP$new(in_features + 1, num_nodes, batch_norm,
                       dropout, activation, output_activation)
  }

  # Load and preprocess state dict
  state_dict <- torch::load_state_dict(path)
  if (all(startsWith(names(state_dict), "net.net."))) {
    names(state_dict) <- sub("net.", "", names(state_dict))
  }
  net$load_state_dict(state_dict)
  net$eval()

  structure(net, model_type = "pycox_coxtime", base_hazard = base_hazard)
}
