################################################################################
#                           Survival Gradient
################################################################################

#' Calculate the Gradient of the Survival Function
#'
#' @family Attribution Methods
#' @export
surv_grad <- function(exp, target = "survival", instance = 1, times_input = FALSE,
                      use_base_hazard = TRUE, include_time = FALSE, batch_size = 50) {
  UseMethod("surv_grad")
}

# DeepSurv ---------------------------------------------------------------------
#'
#' @rdname surv_grad
#' @export
surv_grad.explainer_deepsurv <- function(exp, target = "survival", instance = 1,
                                        times_input = FALSE, use_base_hazard = TRUE,
                                        batch_size = 50) {
  # Check arguments
  assertClass(exp, "explainer_deepsurv")
  assertChoice(target, c("survival", "cum_hazard", "hazard"))
  assertIntegerish(instance, lower = 1, upper = dim(exp$input_data[[1]])[1])
  assertIntegerish(batch_size, lower = 1)
  assertLogical(times_input)
  assertLogical(use_base_hazard)

  # Get the input tensor
  inputs_orig <- to_tensor(exp$input_data, instance)

  # Preprocess inputs
  inputs <- lapply(inputs_orig, exp$model$preprocess_fun)

  # Split inputs into batches
  batches <- split_batches(inputs, batch_size, 1)

  # Calculate the gradients for each batch
  res <- lapply(batches, function(batch) {

    # Calculate the gradients
    # Note: For all targets, the gradients are w.r.t. the hazard and
    # will be transformed to the target of interest afterwards
    grads <- exp$model$calc_gradients(batch$batch,
                                      return_out = TRUE,
                                      use_base_hazard = use_base_hazard)
    outs <- grads$outs[[1]]
    grads <- grads$grads

    grads <- lapply(seq_along(grads), function(i) {
      # Remove the time from the gradient and reshape to (n, input_shape, 1)
      g <- exp$model$postprocess_fun(grads[[i]])
      base_haz <- torch::torch_tensor(exp$model$base_hazard$hazard)$reshape(c(rep(1, g$dim() - 1), -1))
      g <- g * base_haz


      # Aggregate gradients for the cum. hazard or survival outcome
      # Note: This is allowed due to the linearity of the gradient operator
      if (target == "cum_hazard") {
        g <- torch::torch_cumsum(g, dim = -1)
      } else if (target == "survival") {
        g <- -torch::torch_cumsum(g, dim = -1) * torch::torch_exp(-(outs * base_haz)$cumsum(-1))
      }

      # Multiply the gradients with the input
      if (times_input) {
        g <- g * exp$model$postprocess_fun(batch$batch[[i]])
      }

      g
    })

    # Calculate output because 'outs' is only for the hazard
    base_haz <- torch::torch_tensor(exp$model$base_hazard$hazard)$reshape(c(1,1,-1))
    outs <- outs * base_haz
    if (target == "cum_hazard") {
      outs <- torch::torch_cumsum(outs, dim = -1)
    } else if (target == "survival") {
      outs <- torch::torch_exp(-torch::torch_cumsum(outs, dim = -1))
    }

    list(grads = grads, outs = outs)
  })

  # Combine the results
  grads <- combine_batch_grads(res, exp$input_names, exp$model$t_orig, FALSE)
  output <- as.array(torch::torch_cat(lapply(res, function(x) x$outs), dim = 1))

  result <- list(
    res = grads,
    pred = output,
    time = exp$model$t_orig,
    competing_risks = FALSE,
    method = "Surv_Gradient",
    method_args = list(
      target = target, instance = instance, times_input = times_input,
      use_base_hazard = use_base_hazard
    )
  )
  class(result) <- c("surv_result", class(result))
  result
}

# CoxTime ----------------------------------------------------------------------
#'
#' @rdname surv_grad
#' @export
surv_grad.explainer_coxtime <- function(exp, target = "survival", instance = 1,
                                        times_input = FALSE, use_base_hazard = TRUE,
                                        include_time = FALSE, batch_size = 50) {

  # Check arguments
  assertClass(exp, "explainer_coxtime")
  assertChoice(target, c("survival", "cum_hazard", "hazard"))
  assertIntegerish(instance, lower = 1, upper = dim(exp$input_data[[1]])[1])
  assertIntegerish(batch_size, lower = 1)
  assertLogical(times_input)
  assertLogical(use_base_hazard)
  assertLogical(include_time)

  # Get the input tensor
  inputs_orig <- to_tensor(exp$input_data, instance)

  # Add time to input
  inputs <- lapply(inputs_orig, exp$model$preprocess_fun)

  # Split inputs into batches
  batches <- split_batches(inputs, batch_size, length(exp$model$t_orig))

  # Calculate the gradients for each batch
  res <- lapply(batches, function(batch) {

    # Calculate the gradients
    # Note: For all targets, the gradients are w.r.t. the hazard and
    # will be transformed to the target of interest afterwards
    grads <- exp$model$calc_gradients(batch$batch,
                                      return_out = TRUE,
                                      use_base_hazard = use_base_hazard)
    outs <- grads$outs[[1]]
    grads <- grads$grads

    # Remove the gradients w.r.t. the time and add the time
    grads <- lapply(seq_along(grads), function(i) {
      # Remove the time from the gradient and reshape to (n, input_shape, time)
      if (include_time) {
        g <- exp$model$postprocess_fun(grads[[i]])
      } else {
        g <- exp$model$postprocess_fun(grads[[i]][, seq_len(ncol(grads[[i]])-1)])
      }

      # Aggregate gradients for the cum. hazard or survival outcome
      # Note: This is allowed due to the linearity of the gradient operator
      if (target == "cum_hazard") {
        g <- torch::torch_cumsum(g, dim = -1)
      } else if (target == "survival") {
        g <- -torch::torch_cumsum(g, dim = -1) * torch::torch_exp(-outs$cumsum(-1))
      }

      # Multiply the gradients with the input
      if (times_input) {
        if (include_time) {
          g <- g * exp$model$postprocess_fun(batch$batch[[i]])
        } else {
          g <- g * exp$model$postprocess_fun(batch$batch[[i]][, seq_len(ncol(grads[[i]])-1)])
        }
      }

      g
    })

    # Calculate output because 'outs' is only for the hazard
    outs <- outs$squeeze(dim = 2)
    if (target == "cum_hazard") {
      outs <- torch::torch_cumsum(outs, dim = -1)
    } else if (target == "survival") {
      outs <- torch::torch_exp(-torch::torch_cumsum(outs, dim = -1))
    }

    list(grads = grads, outs = outs)
  })

  # Combine the results
  grads <- combine_batch_grads(res, exp$input_names, exp$model$t_orig, include_time)
  output <- as.array(torch::torch_cat(lapply(res, function(x) x$outs), dim = 1))

  result <- list(
    res = grads,
    pred = output,
    time = exp$model$t_orig,
    competing_risks = FALSE,
    method = "Surv_Gradient",
    method_args = list(
      target = target, instance = instance, times_input = times_input,
      use_base_hazard = use_base_hazard, include_time = include_time
    )
  )
  class(result) <- c("surv_result", class(result))
  result
}


# DeepHit ----------------------------------------------------------------------
#'
#' @rdname surv_grad
#' @export
surv_grad.explainer_deephit <- function(exp, target = "survival", instance = 1,
                                        times_input = FALSE, batch_size = 50) {

  # Check arguments
  assertClass(exp, "explainer_deephit")
  assertChoice(target, c("pmf", "cif", "survival"))
  assertIntegerish(instance, lower = 1, upper = dim(exp$input_data[[1]])[1])
  assertIntegerish(batch_size, lower = 1)
  assertLogical(times_input)
  assertLogical(include_time)

  # Get the input tensor
  inputs_orig <- to_tensor(exp$input_data, instance)

  # Add time to input
  inputs <- lapply(inputs_orig, exp$model$preprocess_fun)

  # Split inputs into batches
  batches <- split_batches(inputs, batch_size, 1)

  # Calculate the gradients for each batch
  res <- lapply(batches, function(batch) {

    # Calculate the gradients
    grads <- exp$model$calc_gradients(batch$batch,
                           return_out = TRUE,
                           target = target)
    outs <- grads$outs[[1]]
    grads <- grads$grads

    # Remove the gradients w.r.t. the time and add the time
    grads <- lapply(seq_along(grads), function(i) {
      g <- grads[[i]]

      # Multiply the gradients with the input
      if (times_input) {
        g <- g * batch$batch[[i]]$unsqueeze(-1)$unsqueeze(-1)
      }

      g
    })

    list(grads = grads, outs = outs)
  })

  # Combine the results
  num_risks <- rev(dim(res[[1]]$grads[[1]]))[2]
  event_names <- paste0("Event ", seq_len(num_risks))
  grads <- combine_batch_grads(res, exp$input_names,
                               exp$model$time_bins, FALSE, event_names)

  # Calculate the output
  output <- as.array(torch::torch_cat(lapply(res, function(x) x$outs), dim = 1))

  result <- list(
    res = grads,
    pred = output,
    time = exp$model$time_bins,
    competing_risks = num_risks > 1,
    method = "Surv_Gradient",
    method_args = list(
      target = target, instance = instance, times_input = times_input
    )
  )
  class(result) <- c("surv_result", class(result))
  result
}
