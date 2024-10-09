
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
    batch_size = 50,
    n = 10,
    noise_level = 0.1,
    include_time = FALSE) {

  UseMethod("surv_smoothgrad")
}



# DeepSurv ---------------------------------------------------------------------
#'
#' @rdname surv_smoothgrad
#' @export
surv_smoothgrad.explainer_deepsurv <- function(exp, target = "survival", instance = 1,
                                               times_input = FALSE, batch_size = 50,
                                               n = 10, noise_level = 0.1, ...) {

  # Check arguments
  assertClass(exp, "explainer_deepsurv")
  assertChoice(target, c("survival", "cum_hazard", "hazard"))
  assertIntegerish(instance, lower = 1, upper = dim(exp$input_data[[1]])[1])
  assertIntegerish(n, lower = 1)
  assertNumeric(noise_level, lower = 0)
  assertIntegerish(batch_size, lower = 1)
  assertLogical(times_input)

  # Get noise function
  noise_fun <- function(x) add_noise(x, exp$input_data, noise_level)

  result <- base_method(exp = exp,
                        instance = instance,
                        n = n,
                        model_class = "DeepSurv",
                        inputs_ref = NULL,
                        method_pre_fun = noise_fun,
                        scale_tensor = NULL,
                        n_timepoints = 1,
                        return_out = TRUE,
                        remove_time = FALSE,
                        batch_size = batch_size,
                        times_input = times_input,
                        target = target)

  result <- append(result, list(
    model_class = "DeepSurv",
    method = "Surv_SmoothGrad",
    method_args = list(
      target = target, instance = instance, times_input = times_input,
      n = n, noise_level = noise_level
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
                                              times_input = FALSE, batch_size = 50,
                                              n = 10, noise_level = 0.1, include_time = FALSE) {

  # Check arguments
  assertClass(exp, "explainer_coxtime")
  assertChoice(target, c("survival", "cum_hazard", "hazard"))
  assertIntegerish(instance, lower = 1, upper = dim(exp$input_data[[1]])[1])
  assertIntegerish(n, lower = 1)
  assertNumeric(noise_level, lower = 0)
  assertIntegerish(batch_size, lower = 1)
  assertLogical(times_input)
  assertLogical(include_time)

  # Get noise function
  noise_fun <- function(x) add_noise(x, exp$input_data, noise_level)

  result <- base_method(exp = exp,
                        instance = instance,
                        n = n,
                        model_class = "CoxTime",
                        inputs_ref = NULL,
                        method_pre_fun = noise_fun,
                        scale_tensor = NULL,
                        n_timepoints = length(exp$model$t_orig),
                        return_out = TRUE,
                        remove_time = !include_time,
                        batch_size = batch_size,
                        times_input = times_input,
                        target = target)

  result <- append(result, list(
    model_class = "CoxTime",
    method = "Surv_SmoothGrad",
    method_args = list(
      target = target, instance = instance, times_input = times_input,
      include_time = include_time, n = n, noise_level = noise_level
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
                                              times_input = FALSE, batch_size = 50,
                                              n = 10, noise_level = 0.1, ...) {

  # Check arguments
  assertClass(exp, "explainer_deephit")
  assertChoice(target, c("survival", "cif", "pmf"))
  assertIntegerish(instance, lower = 1, upper = dim(exp$input_data[[1]])[1])
  assertIntegerish(n, lower = 1)
  assertNumeric(noise_level, lower = 0)
  assertIntegerish(batch_size, lower = 1)
  assertLogical(times_input)

  # Get noise function
  noise_fun <- function(x) add_noise(x, exp$input_data, noise_level)

  result <- base_method(exp = exp,
                        instance = instance,
                        n = n,
                        model_class = "DeepHit",
                        inputs_ref = NULL,
                        method_pre_fun = noise_fun,
                        scale_tensor = NULL,
                        n_timepoints = 1,
                        return_out = TRUE,
                        remove_time = FALSE,
                        batch_size = batch_size,
                        times_input = times_input,
                        target = target)

  result <- append(result, list(
    model_class = "DeepHit",
    method = "Surv_SmoothGrad",
    method_args = list(
      target = target, instance = instance, times_input = times_input,
      n = n, noise_level = noise_level
    )
  ))
  class(result) <- c("surv_result", class(result))

  result

}


