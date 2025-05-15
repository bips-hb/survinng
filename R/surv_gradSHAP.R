
################################################################################
#                       Survival GradSHAP
################################################################################

#' Calculate the GradSHAP values of the Survival Function
#'
#' This function calculates the GradSHAP values of the survival function with
#' respect to the input features and time points for a given instance. In the
#' paper, this is referred to as the *"GradSHAP(t)"* method. It is a fast and
#' model-specific method for calculating the Shapley values for a deep
#' survival model.
#'
#' @param exp An object of class `explainer_deepsurv`, `explainer_coxtime`, or
#' `explainer_deephit`.
#' @param target A character string indicating the target output. For
#' `DeepSurv` and `CoxTime`, it can be either `"survival"` (default),
#' `"cum_hazard"`, or `"hazard"`. For `DeepHit`, it can be `"survival"`
#' (default), `"cif"`, or `"pmf"`.
#' @param instance An integer specifying the instance for which the GradSHAP
#' values are calculated. It should be between 1 and the number of instances in
#' the dataset.
#' @param times_input A logical value indicating whether the GradSHAP values
#' should be multiplied with input.
#' @param batch_size An integer specifying the batch size for processing. The
#' default is 1000. This value describes the number of instances within one
#' batch and not the final number of rows in the batch. For example,
#' `CoxTime` replicates each instance for each time point.
#' @param n An integer specifying the number of samples to be used for
#' approximating the integral. The default is 50.
#' @param num_samples An integer specifying the number of samples to be used
#' for the baseline distribution. The default is 10.
#' @param data_ref A reference dataset for sampling. If `NULL`, the reference
#' dataset is taken from the input data of the model. This dataset should
#' contain the same number of features as the input data.
#' @param dtype A character string indicating the data type for the tensors.
#' It can be either `"float"` (default) or `"double"`.
#' @param replace A logical value indicating whether to sample from the baseline
#' distribution with replacement. The default is `TRUE`.
#' @param include_time A logical value indicating whether to calculate GradSHAP
#' also for each time point. This is only relevant for `CoxTime` and is ignored
#' for `DeepSurv` and `DeepHit`. The default is `FALSE`.
#' @param ... Unused arguments.
#'
#' #' @return Returns an object of class `surv_result`.
#'
#'
#' @family Attribution Methods
#' @export
surv_gradSHAP <- function(exp, target = "survival", instance = 1,
                         times_input = TRUE, batch_size = 1000,
                         n = 50, num_samples = 10, data_ref = NULL,
                         dtype = "float", replace = TRUE, include_time = FALSE) {
  UseMethod("surv_gradSHAP")
}

# DeepSurv ----------------------------------------------------------------------

#' @rdname surv_gradSHAP
#' @export
surv_gradSHAP.explainer_deepsurv <- function(exp, target = "survival", instance = 1,
                                             times_input = TRUE, batch_size = 1000,
                                             n = 50, num_samples = 10,
                                             data_ref = NULL, dtype = "float",
                                             replace = TRUE,
                                             ...) {
  # Check arguments
  assertClass(exp, "explainer_deepsurv")
  assertChoice(target, c("survival", "cum_hazard", "hazard"))
  assertIntegerish(instance, lower = 1, upper = dim(exp$input_data[[1]])[1])
  assertIntegerish(n, lower = 1)
  assertIntegerish(num_samples, lower = 1)
  assertIntegerish(batch_size, lower = 1)
  assertArgData(data_ref, null.ok = TRUE)
  assertChoice(dtype, c("float", "double"))
  assertLogical(replace)

  # Set dtype of all tensors
  dtype_name <- dtype
  dtype <- switch(dtype_name,
                  "float" = torch::torch_float(),
                  "double" = torch::torch_double())

  # Set reference value
  if (is.null(data_ref)) {
    data_ref <- exp$input_data
  }
  if (!is.list(data_ref)) data_ref <- list(data_ref)

  # Sample from baseline distribution
  idx <- rep(sample.int(dim(data_ref[[1]])[1], num_samples, replace = replace),
             times = length(instance), each = n)
  data_ref <- lapply(data_ref, function(x) x[idx, , drop = FALSE])

  # Sample value between 0 and 1 (same for each instance)
  scale_fun <- function(n, ...) torch::torch_rand(n, dtype = dtype)

  result <- base_method(exp = exp,
                        instance = instance,
                        n = n,
                        model_class = "DeepSurv",
                        inputs_ref = data_ref,
                        method_pre_fun = NULL,
                        scale_fun = scale_fun,
                        n_timepoints = 1,
                        return_out = TRUE,
                        remove_time = TRUE,
                        batch_size = batch_size,
                        times_input = times_input,
                        target = target,
                        num_samples = num_samples,
                        dtype = dtype)

  result <- append(result, list(
    model_class = "DeepSurv",
    method = "Surv_GradSHAP",
    method_args = list(
      target = target, instance = instance, times_input = times_input,
      n = n, num_samples = num_samples, dtype = dtype_name
    )
  ))
  class(result) <- c("surv_result", class(result))

  result
}

# CoxTime ----------------------------------------------------------------------

#' @rdname surv_gradSHAP
#' @export
surv_gradSHAP.explainer_coxtime <- function(exp, target = "survival", instance = 1,
                                            times_input = TRUE, batch_size = 1000,
                                            n = 50, num_samples = 10, data_ref = NULL,
                                            dtype = "float", replace = TRUE,
                                            include_time = FALSE) {
  # Check arguments
  assertClass(exp, "explainer_coxtime")
  assertChoice(target, c("survival", "cum_hazard", "hazard"))
  assertIntegerish(instance, lower = 1, upper = dim(exp$input_data[[1]])[1])
  assertIntegerish(n, lower = 1)
  assertIntegerish(num_samples, lower = 1)
  assertIntegerish(batch_size, lower = 1)
  assertArgData(data_ref, null.ok = TRUE)
  assertChoice(dtype, c("float", "double"))
  assertLogical(replace)

  # Set dtype of all tensors
  dtype_name <- dtype
  dtype <- switch(dtype_name,
                  "float" = torch::torch_float(),
                  "double" = torch::torch_double())

  # Set reference value
  if (is.null(data_ref)) {
    data_ref <- exp$input_data
  }
  if (!is.list(data_ref)) data_ref <- list(data_ref)

  # Sample from baseline distribution (same for each timepoint and instance)
  num_samples <- min(num_samples, dim(data_ref[[1]])[1])
  idx <- rep(sample.int(dim(data_ref[[1]])[1], num_samples, replace = replace),
             times = length(instance), each = n)
  data_ref <- lapply(data_ref, function(x) x[idx, , drop = FALSE])

  # Sample value between 0 and 1 (same for each timepoint and instance)
  scale_fun <- function(n, ...) torch::torch_rand(n, dtype = dtype)

  result <- base_method(exp = exp,
                        instance = instance,
                        n = n,
                        model_class = "CoxTime",
                        inputs_ref = data_ref,
                        method_pre_fun = NULL,
                        scale_fun = scale_fun,
                        n_timepoints = length(exp$model$t_orig),
                        return_out = TRUE,
                        remove_time = !include_time,
                        batch_size = batch_size,
                        times_input = times_input,
                        target = target,
                        num_samples = num_samples,
                        dtype = dtype)

  result <- append(result, list(
    model_class = "CoxTime",
    method = "Surv_GradSHAP",
    method_args = list(
      target = target, instance = instance, times_input = times_input,
      n = n, num_samples = num_samples, include_time = include_time,
      dtype = dtype_name
    )
  ))
  class(result) <- c("surv_result", class(result))

  result
}


# DeepHit ----------------------------------------------------------------------

#' @rdname surv_gradSHAP
#' @export
surv_gradSHAP.explainer_deephit <- function(exp, target = "survival", instance = 1,
                                            times_input = TRUE, batch_size = 1000,
                                            n = 50, num_samples = 10, data_ref = NULL,
                                            dtype = "float", replace = TRUE, ...) {
  # Check arguments
  assertClass(exp, "explainer_deephit")
  assertChoice(target, c("survival", "pmf", "cif"))
  assertIntegerish(instance, lower = 1, upper = dim(exp$input_data[[1]])[1])
  assertIntegerish(n, lower = 1)
  assertIntegerish(num_samples, lower = 1)
  assertIntegerish(batch_size, lower = 1)
  assertArgData(data_ref, null.ok = TRUE)
  assertChoice(dtype, c("float", "double"))
  assertLogical(replace)

  # Set dtype of all tensors
  dtype_name <- dtype
  dtype <- switch(dtype_name,
                  "float" = torch::torch_float(),
                  "double" = torch::torch_double())

  # Set reference value
  if (is.null(data_ref)) {
    data_ref <- exp$input_data
  }
  if (!is.list(data_ref)) data_ref <- list(data_ref)

  # Sample from baseline distribution
  idx <- rep(sample.int(dim(data_ref[[1]])[1], num_samples, replace = replace),
             times = length(instance), each = n)
  data_ref <- lapply(data_ref, function(x) x[idx, , drop = FALSE])

  # Sample value between 0 and 1 (same for each timepoint and instance)
  scale_fun  <- function(n, ...) torch::torch_rand(n, dtype = dtype)

  result <- base_method(exp = exp,
                        instance = instance,
                        n = n,
                        model_class = "DeepHit",
                        inputs_ref = data_ref,
                        method_pre_fun = NULL,
                        scale_fun = scale_fun,
                        n_timepoints = 1,
                        return_out = TRUE,
                        remove_time = FALSE,
                        batch_size = batch_size,
                        times_input = times_input,
                        target = target,
                        num_samples = num_samples,
                        dtype = dtype)

  result <- append(result, list(
    model_class = "DeepHit",
    method = "Surv_GradSHAP",
    method_args = list(
      target = target, instance = instance, times_input = times_input,
      n = n, num_samples = num_samples, dtype = dtype_name
    )
  ))
  class(result) <- c("surv_result", class(result))

  result
}
