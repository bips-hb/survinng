
################################################################################
#                       Survival GradSHAP
################################################################################

#' Calculate the GradSHAP values of the Survival Function
#'
#' @family Attribution Methods
#' @export
surv_gradSHAP <- function(exp, target = "survival", instance = 1,
                         times_input = TRUE, batch_size = 50,
                         n = 50, num_samples = 10, data_ref = NULL,
                         include_time = FALSE) {
  UseMethod("surv_gradSHAP")
}

# DeepSurv ----------------------------------------------------------------------

#' @rdname surv_gradSHAP
#' @export
surv_gradSHAP.explainer_deepsurv <- function(exp, target = "survival", instance = 1,
                                             times_input = TRUE, batch_size = 50,
                                             n = 50, num_samples = 10,
                                             data_ref = NULL, ...) {
  # Check arguments
  assertClass(exp, "explainer_deepsurv")
  assertChoice(target, c("survival", "cum_hazard", "hazard"))
  assertIntegerish(instance, lower = 1, upper = dim(exp$input_data[[1]])[1])
  assertIntegerish(n, lower = 1)
  assertIntegerish(num_samples, lower = 1)
  assertIntegerish(batch_size, lower = 1)
  assertArgData(data_ref, null.ok = TRUE)

  # Set reference value
  if (is.null(data_ref)) {
    data_ref <- exp$input_data
  }
  if (!is.list(data_ref)) data_ref <- list(data_ref)

  # Sample from baseline distribution
  idx <- rep(sample.int(dim(data_ref[[1]])[1], num_samples, replace = TRUE),
             times = length(instance), each = n)
  data_ref <- lapply(data_ref, function(x) x[idx, , drop = FALSE])

  # Sample value between 0 and 1
  scale <- torch::torch_tensor(rep(runif(n * num_samples),
                                   times = length(instance)), dtype = torch_double())$unsqueeze(-1)

  result <- base_method(exp = exp,
                        instance = instance,
                        n = n,
                        model_class = "DeepSurv",
                        inputs_ref = data_ref,
                        method_pre_fun = NULL,
                        scale_tensor = scale,
                        n_timepoints = 1,
                        return_out = TRUE,
                        remove_time = TRUE,
                        batch_size = batch_size,
                        times_input = times_input,
                        target = target,
                        num_samples = num_samples)

  result <- append(result, list(
    model_class = "DeepSurv",
    method = "Surv_GradSHAP",
    method_args = list(
      target = target, instance = instance, times_input = times_input,
      n = n, num_samples = num_samples
    )
  ))
  class(result) <- c("surv_result", class(result))

  result
}

# CoxTime ----------------------------------------------------------------------

#' @rdname surv_gradSHAP
#' @export
surv_gradSHAP.explainer_coxtime <- function(exp, target = "survival", instance = 1,
                                            times_input = TRUE, batch_size = 50,
                                            n = 50, num_samples = 10, data_ref = NULL,
                                            include_time = FALSE) {
  # Check arguments
  assertClass(exp, "explainer_coxtime")
  assertChoice(target, c("survival", "cum_hazard", "hazard"))
  assertIntegerish(instance, lower = 1, upper = dim(exp$input_data[[1]])[1])
  assertIntegerish(n, lower = 1)
  assertIntegerish(num_samples, lower = 1)
  assertIntegerish(batch_size, lower = 1)
  assertArgData(data_ref, null.ok = TRUE)

  # Set reference value
  if (is.null(data_ref)) {
    data_ref <- exp$input_data
  }
  if (!is.list(data_ref)) data_ref <- list(data_ref)

  # Sample from baseline distribution
  idx <- rep(sample.int(dim(data_ref[[1]])[1], num_samples, replace = TRUE),
             times = length(instance), each = n)
  data_ref <- lapply(data_ref, function(x) x[idx, , drop = FALSE])

  # Sample value between 0 and 1
  scale <- torch::torch_tensor(rep(runif(n * length(exp$model$t_orig) * num_samples),
                                   times = length(instance)), dtype = torch_double())$unsqueeze(-1)

  result <- base_method(exp = exp,
                        instance = instance,
                        n = n,
                        model_class = "CoxTime",
                        inputs_ref = data_ref,
                        method_pre_fun = NULL,
                        scale_tensor = scale,
                        n_timepoints = length(exp$model$t_orig),
                        return_out = TRUE,
                        remove_time = !include_time,
                        batch_size = batch_size,
                        times_input = times_input,
                        target = target,
                        num_samples = num_samples)

  result <- append(result, list(
    model_class = "CoxTime",
    method = "Surv_GradSHAP",
    method_args = list(
      target = target, instance = instance, times_input = times_input,
      n = n, num_samples = num_samples, include_time = include_time
    )
  ))
  class(result) <- c("surv_result", class(result))

  result
}


# DeepHit ----------------------------------------------------------------------

#' @rdname surv_gradSHAP
#' @export
surv_gradSHAP.explainer_deephit <- function(exp, target = "survival", instance = 1,
                                            times_input = TRUE, batch_size = 50,
                                            n = 50, num_samples = 10, data_ref = NULL,
                                            ...) {
  # Check arguments
  assertClass(exp, "explainer_deephit")
  assertChoice(target, c("survival", "pmf", "cif"))
  assertIntegerish(instance, lower = 1, upper = dim(exp$input_data[[1]])[1])
  assertIntegerish(n, lower = 1)
  assertIntegerish(num_samples, lower = 1)
  assertIntegerish(batch_size, lower = 1)
  assertArgData(data_ref, null.ok = TRUE)

  # Set reference value
  if (is.null(data_ref)) {
    data_ref <- exp$input_data
  }
  if (!is.list(data_ref)) data_ref <- list(data_ref)

  # Sample from baseline distribution
  idx <- rep(sample.int(dim(data_ref[[1]])[1], num_samples, replace = TRUE),
             times = length(instance), each = n)
  data_ref <- lapply(data_ref, function(x) x[idx, , drop = FALSE])

  # Sample value between 0 and 1
  scale <- torch::torch_tensor(rep(runif(n * num_samples),
                                   times = length(instance)), dtype = torch_double())$unsqueeze(-1)

  result <- base_method(exp = exp,
                        instance = instance,
                        n = n,
                        model_class = "DeepHit",
                        inputs_ref = data_ref,
                        method_pre_fun = NULL,
                        scale_tensor = scale,
                        n_timepoints = 1,
                        return_out = TRUE,
                        remove_time = FALSE,
                        batch_size = batch_size,
                        times_input = times_input,
                        target = target,
                        num_samples = num_samples)

  result <- append(result, list(
    model_class = "DeepHit",
    method = "Surv_GradSHAP",
    method_args = list(
      target = target, instance = instance, times_input = times_input,
      n = n, num_samples = num_samples
    )
  ))
  class(result) <- c("surv_result", class(result))

  result
}
