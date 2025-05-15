
################################################################################
#                       Survival SmoothGrad
################################################################################

#' Calculate the SmoothGrad values of the Survival Function
#'
#' This function calculates the SmoothGrad values of the survival function with
#' respect to the input features and time points for a given instance. In the
#' paper, this is referred to as the *"SG(t)"* method. It shows the smoothed
#' sensitivity of the survival function to changes in the input features at a
#' specific time point.
#'
#' @param exp An object of class `explainer_deepsurv`, `explainer_coxtime`, or
#' `explainer_deephit`.
#' @param target A character string indicating the target output. For `DeepSurv`
#' and `CoxTime`, it can be either `"survival"` (default), `"cum_hazard"`, or
#' `"hazard"`. For `DeepHit`, it can be `"survival"` (default), `"cif"`, or
#' `"pmf"`.
#' @param instance An integer specifying the instance for which the SmoothGrad
#' is calculated. It should be between 1 and the number of instances in the
#' dataset.
#' @param times_input A logical value indicating whether the SmoothGrad should
#' be multiplied with input. In the paper, this variant is referred to as
#' `"SGxI(t)"`.
#' @param batch_size An integer specifying the batch size for processing. The
#' default is 1000. This value describes the number of instances within one batch
#' and not the final number of rows in the batch. For example, `CoxTime`
#' replicates each instance for each time point.
#' @param n An integer specifying the number of noise samples to be added to
#' the input features. The default is 10.
#' @param noise_level A numeric value specifying the level of Gaussian noise to
#' be added to the input features. The default is 0.1.
#' @param dtype A character string indicating the data type for the tensors.
#' It can be either `"float"` (default) or `"double"`.
#' @param include_time A logical value indicating whether to also calculate
#' the gradients with respect to the time. This is only relevant for
#' `CoxTime` and is ignored for `DeepSurv` and `DeepHit`.
#' @param ... Unused arguments.
#'
#'
#' @family Attribution Methods
#' @export
surv_smoothgrad <- function(
    exp,
    target = "survival",
    instance = 1,
    times_input = FALSE,
    batch_size = 50,
    n = 10,
    noise_level = 0.1,
    dtype = "float",
    include_time = FALSE) {

  UseMethod("surv_smoothgrad")
}



# DeepSurv ---------------------------------------------------------------------
#'
#' @rdname surv_smoothgrad
#' @export
surv_smoothgrad.explainer_deepsurv <- function(exp, target = "survival", instance = 1,
                                               times_input = FALSE, batch_size = 1000,
                                               n = 10, noise_level = 0.1,
                                               dtype = "float", ...) {

  # Check arguments
  assertClass(exp, "explainer_deepsurv")
  assertChoice(target, c("survival", "cum_hazard", "hazard"))
  assertIntegerish(instance, lower = 1, upper = dim(exp$input_data[[1]])[1])
  assertIntegerish(n, lower = 1)
  assertNumeric(noise_level, lower = 0)
  assertIntegerish(batch_size, lower = 1)
  assertLogical(times_input)
  assertChoice(dtype, c("float", "double"))

  # Set dtype of all tensors
  dtype_name <- dtype
  dtype <- switch(dtype_name,
                  "float" = torch::torch_float(),
                  "double" = torch::torch_double())

  # Get noise function
  noise_fun <- function(x) add_noise(x, exp$input_data, noise_level, dtype)

  result <- base_method(exp = exp,
                        instance = instance,
                        n = n,
                        model_class = "DeepSurv",
                        inputs_ref = NULL,
                        method_pre_fun = noise_fun,
                        scale_fun = NULL,
                        n_timepoints = 1,
                        return_out = TRUE,
                        remove_time = FALSE,
                        batch_size = batch_size,
                        times_input = times_input,
                        target = target,
                        dtype = dtype)

  result <- append(result, list(
    model_class = "DeepSurv",
    method = "Surv_SmoothGrad",
    method_args = list(
      target = target, instance = instance, times_input = times_input,
      n = n, noise_level = noise_level, dtype = dtype_name
    )
  ))
  class(result) <- c("surv_result", class(result))

  result
}


# CoxTime ----------------------------------------------------------------------
#'
#' @rdname surv_smoothgrad
#' @export
surv_smoothgrad.explainer_coxtime <- function(exp, target = "survival", instance = 1,
                                              times_input = FALSE, batch_size = 1000,
                                              n = 10, noise_level = 0.1,
                                              dtype = "float", include_time = FALSE) {

  # Check arguments
  assertClass(exp, "explainer_coxtime")
  assertChoice(target, c("survival", "cum_hazard", "hazard"))
  assertIntegerish(instance, lower = 1, upper = dim(exp$input_data[[1]])[1])
  assertIntegerish(n, lower = 1)
  assertNumeric(noise_level, lower = 0)
  assertIntegerish(batch_size, lower = 1)
  assertLogical(times_input)
  assertLogical(include_time)
  assertChoice(dtype, c("float", "double"))

  # Set dtype of all tensors
  dtype_name <- dtype
  dtype <- switch(dtype_name,
                  "float" = torch::torch_float(),
                  "double" = torch::torch_double())

  # Get noise function
  noise_fun <- function(x) add_noise(x, exp$input_data, noise_level, dtype)

  result <- base_method(exp = exp,
                        instance = instance,
                        n = n,
                        model_class = "CoxTime",
                        inputs_ref = NULL,
                        method_pre_fun = noise_fun,
                        scale_fun = NULL,
                        n_timepoints = length(exp$model$t_orig),
                        return_out = TRUE,
                        remove_time = !include_time,
                        batch_size = batch_size,
                        times_input = times_input,
                        target = target,
                        dtype = dtype)

  result <- append(result, list(
    model_class = "CoxTime",
    method = "Surv_SmoothGrad",
    method_args = list(
      target = target, instance = instance, times_input = times_input,
      include_time = include_time, n = n, noise_level = noise_level,
      dtype = dtype_name
    )
  ))
  class(result) <- c("surv_result", class(result))

  result
}


# DeepHit ----------------------------------------------------------------------
#'
#' @rdname surv_smoothgrad
#' @export
surv_smoothgrad.explainer_deephit <- function(exp, target = "survival", instance = 1,
                                              times_input = FALSE, batch_size = 1000,
                                              n = 10, noise_level = 0.1,
                                              dtype = "float", ...) {

  # Check arguments
  assertClass(exp, "explainer_deephit")
  assertChoice(target, c("survival", "cif", "pmf"))
  assertIntegerish(instance, lower = 1, upper = dim(exp$input_data[[1]])[1])
  assertIntegerish(n, lower = 1)
  assertNumeric(noise_level, lower = 0)
  assertIntegerish(batch_size, lower = 1)
  assertLogical(times_input)
  assertChoice(dtype, c("float", "double"))

  # Set dtype of all tensors
  dtype_name <- dtype
  dtype <- switch(dtype_name,
                  "float" = torch::torch_float(),
                  "double" = torch::torch_double())

  # Get noise function
  noise_fun <- function(x) add_noise(x, exp$input_data, noise_level, dtype)

  result <- base_method(exp = exp,
                        instance = instance,
                        n = n,
                        model_class = "DeepHit",
                        inputs_ref = NULL,
                        method_pre_fun = noise_fun,
                        scale_fun = NULL,
                        n_timepoints = 1,
                        return_out = TRUE,
                        remove_time = FALSE,
                        batch_size = batch_size,
                        times_input = times_input,
                        target = target,
                        dtype = dtype)

  result <- append(result, list(
    model_class = "DeepHit",
    method = "Surv_SmoothGrad",
    method_args = list(
      target = target, instance = instance, times_input = times_input,
      n = n, noise_level = noise_level, dtype = dtype_name
    )
  ))
  class(result) <- c("surv_result", class(result))

  result

}

########################## Utility functions ###################################

# Add noise --------------------------------------------------------------------
add_noise <- function(inputs, orig_data, noise_level, dtype = torch::torch_float()) {
  # Make sure both are lists
  if (!is.list(inputs)) inputs <- list(inputs)
  if (!is.list(orig_data)) orig_data <- list(orig_data)

  lapply(seq_along(inputs), function(i) {
    orig <- orig_data[[i]]
    names(orig) <- NULL

    # Calculate standard deviation
    std <- torch::torch_tensor(apply(orig, seq_along(dim(orig))[-1], sd), dtype = dtype)$unsqueeze(1)

    # Generate noise
    noise <- torch::torch_tensor(array(rnorm(prod(dim(inputs[[i]]))), dim = dim(inputs[[i]])), dtype = dtype)
    noise <- noise * noise_level *std

    # Add noise
    inputs[[i]] + noise
  })
}


