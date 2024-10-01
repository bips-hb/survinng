################################################################################
#                       Survival IntegratedGradients
################################################################################

#' Calculate the Integrated Gradients of the Survival Function
#'
#' @family Attribution Methods
#' @export
surv_intgrad <- function(exp, target = "survival", instance = 1,
                         n = 10, x_ref = NULL, use_base_hazard = TRUE,
                         batch_size = 50) {
  UseMethod("surv_intgrad")
}

# DeepSurv ----------------------------------------------------------------------

#' @rdname surv_intgrad
#' @export
surv_intgrad.explainer_deepsurv <- function(exp, target = "survival", instance = 1,
                                           n = 10, x_ref = NULL, use_base_hazard = TRUE,
                                           batch_size = 50) {

  # Check arguments
  assertClass(exp, "explainer_deepsurv")
  assertChoice(target, c("survival", "cum_hazard", "hazard"))
  assertIntegerish(instance, lower = 1, upper = dim(exp$input_data[[1]])[1])
  assertIntegerish(n, lower = 1)
  assertIntegerish(batch_size, lower = 1)
  assertLogical(use_base_hazard)
  assertArgData(x_ref, null.ok = TRUE)

  # Set reference value
  if (is.null(x_ref)) {
    x_ref <- lapply(exp$input_data,
                    function(x) {
                      res <- apply(x, seq_along(dim(x))[-1], mean, simplify = TRUE)
                      if (is.null(dim(res))) dim(res) <- c(1, length(res))
                      res
                    })
  }

  # Get the input tensor
  inputs <- to_tensor(exp$input_data, instance, repeats = n)
  inputs_ref <- to_tensor(x_ref, 1, repeats = n * length(instance))

  # Preprocess inputs
  inputs_time <- lapply(inputs, exp$model$preprocess_fun)
  inputs_ref_time <- lapply(inputs_ref, exp$model$preprocess_fun)

  # Interpolate between reference and input
  scale <- torch::torch_tensor(rep(seq(1/n, 1, length.out = n),
                                   times = length(instance)))$unsqueeze(-1)
  inputs <- lapply(seq_along(inputs_time), function(i) {
    inputs_ref_time[[i]] + scale * (inputs_time[[i]] - inputs_ref_time[[i]])
  })

  # Split inputs into batches
  batches <- split_batches(inputs, batch_size, 1, n)

  # Calculate the gradients
  # Note:
  # grads$grads has a batch size of n * time
  # grads$outs has a size of (n, 1, time)
  res <- lapply(batches, function(batch) {

    # Calculate the gradients
    grads <- exp$model$calc_gradients(batch$batch,
                                      return_out = TRUE,
                                      use_base_hazard = use_base_hazard)
    outs <- grads$outs[[1]]
    grads <- grads$grads

    # Remove the gradients w.r.t. the time and add the time
    grads <- lapply(seq_along(grads), function(i) {
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

      # Times the input difference
      rows <- batch$idx[1]:batch$idx[2]
      input_diff <- inputs_time[[i]][rows,, drop = FALSE] -
        inputs_ref_time[[i]][rows,, drop = FALSE]
      g <- g * exp$model$postprocess_fun(input_diff)

      # Take mean over repeated values
      g <- torch::torch_stack(g$chunk(batch$num, dim = 1L), dim = 2)$mean(dim = 1L)

      g
    })

    list(grads = grads, outs = NULL)
  })

  # Combine the results
  grads <- combine_batch_grads(res, exp$input_names, exp$model$t_orig, FALSE)

  # Calculate the predictions
  outs <- to_tensor(exp$input_data, instance, repeats = 1) |>
    lapply(FUN = exp$model$preprocess_fun) |>
    exp$model$forward(target = target, use_base_hazard = use_base_hazard) |>
    lapply(FUN = torch::torch_squeeze, dim = 2)

  # Calculate the reference predictions
  outs_ref <- to_tensor(x_ref, 1, repeats = length(instance)) |>
    lapply(FUN = exp$model$preprocess_fun) |>
    exp$model$forward(target = target, use_base_hazard = use_base_hazard) |>
    lapply(FUN = torch::torch_squeeze, dim = 2)

  result <- list(
    res = grads,
    pred = as.array(outs[[1]]),
    pred_diff = as.array(outs[[1]] - outs_ref[[1]]),
    time = exp$model$t_orig,
    competing_risks = FALSE,
    method = "Surv_IntGrad",
    method_args = list(
      target = target, instance = instance, n = n,
      use_base_hazard = use_base_hazard
    )
  )
  class(result) <- c("surv_result", class(result))
  result
}

# CoxTime ----------------------------------------------------------------------

#' @rdname surv_intgrad
#' @export
surv_intgrad.explainer_coxtime <- function(exp, target = "survival", instance = 1,
                                           n = 10, x_ref = NULL, use_base_hazard = TRUE,
                                           batch_size = 50) {

  # Check arguments
  assertClass(exp, "explainer_coxtime")
  assertChoice(target, c("survival", "cum_hazard", "hazard"))
  assertIntegerish(instance, lower = 1, upper = dim(exp$input_data[[1]])[1])
  assertIntegerish(n, lower = 1)
  assertIntegerish(batch_size, lower = 1)
  assertLogical(use_base_hazard)
  assertArgData(x_ref, null.ok = TRUE)

  # Set reference value
  if (is.null(x_ref)) {
    x_ref <- lapply(exp$input_data,
                    function(x) {
                      res <- apply(x, seq_along(dim(x))[-1], mean, simplify = TRUE)
                      if (is.null(dim(res))) dim(res) <- c(1, length(res))
                      res
                    })
  }

  # Get the input tensor
  inputs <- to_tensor(exp$input_data, instance, repeats = n)
  inputs_ref <- to_tensor(x_ref, 1, repeats = n * length(instance))

  # Add time to input
  inputs_time <- lapply(inputs, exp$model$preprocess_fun)
  inputs_ref_time <- lapply(inputs_ref, exp$model$preprocess_fun)

  # Interpolate between reference and input
  scale <- torch::torch_tensor(rep(seq(1/n, 1, length.out = n),
                            times = length(instance),
                            each = length(exp$model$t_orig)))$unsqueeze(-1)
  inputs <- lapply(seq_along(inputs_time), function(i) {
    inputs_ref_time[[i]] + scale * (inputs_time[[i]] - inputs_ref_time[[i]])
  })

  # Split inputs into batches
  batches <- split_batches(inputs, batch_size, length(exp$model$t_orig), n)

  # Calculate the gradients
  # Note:
  # grads$grads has a batch size of n * time
  # grads$outs has a size of (n, 1, time)
  res <- lapply(batches, function(batch) {

    # Calculate the gradients
    grads <- exp$model$calc_gradients(batch$batch,
                                      return_out = TRUE,
                                      use_base_hazard = use_base_hazard)
    outs <- grads$outs[[1]]
    grads <- grads$grads

    # Remove the gradients w.r.t. the time and add the time
    grads <- lapply(seq_along(grads), function(i) {
      # Remove the time from the gradient and reshape to (n, input_shape, time)
      g <- exp$model$postprocess_fun(grads[[i]][, seq_len(ncol(grads[[i]])-1)])

      # Aggregate gradients for the cum. hazard or survival outcome
      # Note: This is allowed due to the linearity of the gradient operator
      if (target == "cum_hazard") {
        g <- torch::torch_cumsum(g, dim = -1)
      } else if (target == "survival") {
        g <- -torch::torch_cumsum(g, dim = -1) * torch::torch_exp(-outs$cumsum(-1))
      }

      # Times the input difference
      rows <- batch$idx[1]:batch$idx[2]
      input_diff <- inputs_time[[i]][rows, seq_len(ncol(grads[[i]])-1), drop = FALSE] -
        inputs_ref_time[[i]][rows, seq_len(ncol(grads[[i]])-1), drop = FALSE]
      g <- g * exp$model$postprocess_fun(input_diff)

      # Take mean over repeated values
      g <- torch::torch_stack(g$chunk(batch$num, dim = 1L), dim = 2)$mean(dim = 1L)

      g
    })

    list(grads = grads, outs = NULL)
  })

  # Combine the results
  grads <- combine_batch_grads(res, exp$input_names, exp$model$t_orig, FALSE)

  # Calculate the predictions
  outs <- to_tensor(exp$input_data, instance, repeats = 1) |>
    lapply(FUN = exp$model$preprocess_fun) |>
    exp$model$forward(target = target, use_base_hazard = use_base_hazard) |>
    lapply(FUN = torch::torch_squeeze, dim = 2)

  # Calculate the reference predictions
  outs_ref <- to_tensor(x_ref, 1, repeats = length(instance)) |>
    lapply(FUN = exp$model$preprocess_fun) |>
    exp$model$forward(target = target, use_base_hazard = use_base_hazard) |>
    lapply(FUN = torch::torch_squeeze, dim = 2)

  result <- list(
    res = grads,
    pred = as.array(outs[[1]]),
    pred_diff = as.array(outs[[1]] - outs_ref[[1]]),
    time = exp$model$t_orig,
    competing_risks = FALSE,
    method = "Surv_IntGrad",
    method_args = list(
      target = target, instance = instance, n = n,
      use_base_hazard = use_base_hazard
    )
  )
  class(result) <- c("surv_result", class(result))
  result
}

# DeepHit ----------------------------------------------------------------------

#' @rdname surv_intgrad
#' @export
surv_intgrad.explainer_deephit <- function(exp, target = "survival", instance = 1,
                                           n = 10, x_ref = NULL, batch_size = 50) {

  # Check arguments
  assertClass(exp, "explainer_deephit")
  assertChoice(target, c("survival", "cif", "pmf"))
  assertIntegerish(instance, lower = 1, upper = dim(exp$input_data[[1]])[1])
  assertIntegerish(n, lower = 1)
  assertIntegerish(batch_size, lower = 1)
  assertArgData(x_ref, null.ok = TRUE)

  # Set reference value
  if (is.null(x_ref)) {
    x_ref <- lapply(exp$input_data,
                    function(x) {
                      res <- apply(x, seq_along(dim(x))[-1], mean, simplify = TRUE)
                      if (is.null(dim(res))) dim(res) <- c(1, length(res))
                      res
                    })
  }

  # Get the input tensor
  inputs <- to_tensor(exp$input_data, instance, repeats = n)
  inputs_ref <- to_tensor(x_ref, 1, repeats = n * length(instance))

  # Preprocess inputs
  inputs_time <- lapply(inputs, exp$model$preprocess_fun)
  inputs_ref_time <- lapply(inputs_ref, exp$model$preprocess_fun)

  # Interpolate between reference and input
  scale <- torch::torch_tensor(rep(seq(1/n, 1, length.out = n),
                                   times = length(instance)))$unsqueeze(-1)
  inputs <- lapply(seq_along(inputs_time), function(i) {
    inputs_ref_time[[i]] + scale * (inputs_time[[i]] - inputs_ref_time[[i]])
  })

  # Split inputs into batches
  batches <- split_batches(inputs, batch_size, 1, n)

  # Calculate the gradients
  # Note:
  # grads$grads has a batch size of n * time
  # grads$outs has a size of (n, 1, time)
  res <- lapply(batches, function(batch) {

    # Calculate the gradients
    grads <- exp$model$calc_gradients(batch$batch,
                                      target = target,
                                      return_out = TRUE)
    outs <- grads$outs[[1]]
    grads <- grads$grads

    # Remove the gradients w.r.t. the time and add the time
    grads <- lapply(seq_along(grads), function(i) {
      g <- grads[[i]]

      # Times the input difference
      rows <- batch$idx[1]:batch$idx[2]
      input_diff <- inputs_time[[i]][rows,, drop = FALSE] -
        inputs_ref_time[[i]][rows,, drop = FALSE]
      g <- g * input_diff$unsqueeze(-1)$unsqueeze(-1)

      # Take mean over repeated values
      g <- torch::torch_stack(g$chunk(batch$num, dim = 1L), dim = 2)$mean(dim = 1L)

      g
    })

    list(grads = grads, outs = NULL)
  })

  # Combine the results
  num_risks <- rev(dim(res[[1]]$grads[[1]]))[2]
  event_names <- paste0("Event ", seq_len(num_risks))
  grads <- combine_batch_grads(res, exp$input_names, exp$model$time_bins, FALSE,
                               event_names)

  # Calculate the predictions
  outs <- to_tensor(exp$input_data, instance, repeats = 1) |>
    lapply(FUN = exp$model$preprocess_fun) |>
    exp$model$forward(target = target) |>
    lapply(FUN = torch::torch_squeeze, dim = 2)

  # Calculate the reference predictions
  outs_ref <- to_tensor(x_ref, 1, repeats = length(instance)) |>
    lapply(FUN = exp$model$preprocess_fun) |>
    exp$model$forward(target = target) |>
    lapply(FUN = torch::torch_squeeze, dim = 2)

  result <- list(
    res = grads,
    pred = as.array(outs[[1]]),
    pred_diff = as.array(outs[[1]] - outs_ref[[1]]),
    time = exp$model$time_bins,
    competing_risks = num_risks > 1,
    method = "Surv_IntGrad",
    method_args = list(
      target = target, instance = instance, n = n
    )
  )
  class(result) <- c("surv_result", class(result))
  result
}
