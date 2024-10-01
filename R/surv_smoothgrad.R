
################################################################################
#                       Survival SmoothGrad
################################################################################

#' Calculate the SmoothGrad values of the Survival Function
#'
#' @family Attribution Methods
#' @export
surv_smoothgrad <- function(
    exp,
    target = "survival",
    instance = 1,
    times_input = FALSE,
    n = 10,
    noise_level = 0.1,
    use_base_hazard = TRUE,
    include_time = FALSE,
    batch_size = 50) {

  UseMethod("surv_smoothgrad")
}

# CoxTime ----------------------------------------------------------------------

#' @rdname surv_smoothgrad
#' @export
surv_smoothgrad.explainer_coxtime <- function(exp, target = "survival", instance = 1,
                                              times_input = FALSE, n = 10, noise_level = 0.1,
                                              use_base_hazard = TRUE, include_time = FALSE, batch_size = 50) {

  # Check arguments
  assertClass(exp, "explainer_coxtime")
  assertChoice(target, c("survival", "cum_hazard", "hazard"))
  assertIntegerish(instance, lower = 1, upper = dim(exp$input_data[[1]])[1])
  assertIntegerish(n, lower = 1)
  assertNumeric(noise_level, lower = 0)
  assertIntegerish(batch_size, lower = 1)
  assertLogical(times_input)
  assertLogical(use_base_hazard)
  assertLogical(include_time)

  # Get the input tensor
  inputs_orig <- to_tensor(exp$input_data, instance, repeats = n)

  # Add noise
  inputs_noise <- add_noise(inputs_orig, exp$input_data, noise_level)

  # Add time to input
  inputs <- lapply(inputs_noise, exp$model$preprocess_fun)

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

      # Take mean over repeated values
      g <- torch::torch_stack(g$chunk(batch$num, dim = 1L), dim = 2)$mean(dim = 1L)

      g
    })

    list(grads = grads, outs = NULL)
  })



  # Combine the results
  grads <- combine_batch_grads(res, exp$input_names, exp$model$t_orig, include_time)

  # Calculate output
  output <- to_tensor(exp$input_data, instance, repeats = 1) |>
    lapply(FUN = exp$model$preprocess_fun) |>
    exp$model$forward(target = target, use_base_hazard = use_base_hazard) |>
    lapply(FUN = torch::torch_squeeze, dim = 2)

  result <- list(
    res = grads,
    pred = as.array(output[[1]]),
    time = exp$model$t_orig,
    competing_risks = FALSE,
    method = "Surv_SmoothGrad",
    method_args = list(
      target = target, instance = instance, times_input = times_input, n = n,
      noise_level = noise_level, use_base_hazard = use_base_hazard
    )
  )
  class(result) <- c("surv_result", class(result))
  result
}

# DeepHit ----------------------------------------------------------------------

#' @rdname surv_smoothgrad
#' @export
surv_smoothgrad.explainer_deephit <- function(exp, target = "survival", instance = 1,
                                              times_input = FALSE, n = 10, noise_level = 0.1,
                                              use_base_hazard = TRUE, include_time = FALSE, batch_size = 50) {

  # Check arguments
  assertClass(exp, "explainer_deephit")
  assertChoice(target, c("survival", "cum_hazard", "hazard"))
  assertIntegerish(instance, lower = 1, upper = dim(exp$input_data[[1]])[1])
  assertIntegerish(n, lower = 1)
  assertNumeric(noise_level, lower = 0)
  assertIntegerish(batch_size, lower = 1)
  assertLogical(times_input)
  assertLogical(use_base_hazard)
  assertLogical(include_time)

  # Get the input tensor
  inputs_orig <- to_tensor(exp$input_data, instance, repeats = n)

  # Add noise
  inputs_noise <- add_noise(inputs_orig, exp$input_data, noise_level)

  # Add time to input
  inputs <- lapply(inputs_noise, exp$model$preprocess_fun)

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

      # Multiply the gradients with the input
      if (times_input) {
        g <- g * batch$batch[[i]]$unsqueeze(-1)$unsqueeze(-1)
      }

      # Take mean over repeated values
      g <- torch::torch_stack(g$chunk(batch$num, dim = 1L), dim = 2)$mean(dim = 1L)

      g
    })

    list(grads = grads, outs = NULL)
  })



  # Combine the results
  num_risks <- rev(dim(res[[1]]$grads[[1]]))[2]
  event_names <- paste0("Event ", seq_len(num_risks))
  grads <- combine_batch_grads(res, exp$input_names, exp$model$t_orig, FALSE,
                               event_names)

  # Calculate output
  output <- to_tensor(exp$input_data, instance, repeats = 1) |>
    lapply(FUN = exp$model$preprocess_fun) |>
    exp$model$forward(target = target) |>
    lapply(FUN = torch::torch_squeeze, dim = 2)

  result <- list(
    res = grads,
    pred = as.array(output[[1]]),
    time = exp$model$time_bins,
    competing_risks = num_risks > 1,
    method = "Surv_SmoothGrad",
    method_args = list(
      target = target, instance = instance, times_input = times_input, n = n,
      noise_level = noise_level
    )
  )
  class(result) <- c("surv_result", class(result))
  result
}
