################################################################################
#                           Survival Gradient
################################################################################

#' Calculate the Gradient of the Survival Function
#'
#' @family Attribution Methods
#' @export
surv_grad <- function(exp, target = "survival", instance = 1, times_input = FALSE,
                      batch_size = 50, dtype = "float", include_time = FALSE) {
  UseMethod("surv_grad")
}

# DeepSurv ---------------------------------------------------------------------
#'
#' @rdname surv_grad
#' @export
surv_grad.explainer_deepsurv <- function(exp, target = "survival", instance = 1,
                                         times_input = FALSE, batch_size = 50,
                                         dtype = "float", ...) {

  # Check arguments
  assertClass(exp, "explainer_deepsurv")
  assertChoice(target, c("survival", "cum_hazard", "hazard"))
  assertIntegerish(instance, lower = 1, upper = dim(exp$input_data[[1]])[1])
  assertIntegerish(batch_size, lower = 1)
  assertLogical(times_input)
  assertChoice(dtype, c("float", "double"))

  # Set dtype of all tensors
  dtype_name <- dtype
  dtype <- switch(dtype_name,
                  "float" = torch::torch_float(),
                  "double" = torch::torch_double())

  result <- base_method(exp = exp,
              instance = instance,
              n = 1,
              model_class = "DeepSurv",
              inputs_ref = NULL,
              method_pre_fun = NULL,
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
    method = "Surv_Gradient",
    method_args = list(
      target = target, instance = instance, times_input = times_input,
      dtype = dtype_name
    )
  ))
  class(result) <- c("surv_result", class(result))

  result
}


# CoxTime ----------------------------------------------------------------------
#'
#' @rdname surv_grad
#' @export
surv_grad.explainer_coxtime <- function(exp, target = "survival", instance = 1,
                                        times_input = FALSE, batch_size = 50,
                                        dtype = "float", include_time = FALSE) {
  # Check arguments
  assertClass(exp, "explainer_coxtime")
  assertChoice(target, c("survival", "cum_hazard", "hazard"))
  assertIntegerish(instance, lower = 1, upper = dim(exp$input_data[[1]])[1])
  assertIntegerish(batch_size, lower = 1)
  assertLogical(times_input)
  assertLogical(include_time)
  assertChoice(dtype, c("float", "double"))

  # Set dtype of all tensors
  dtype_name <- dtype
  dtype <- switch(dtype_name,
                  "float" = torch::torch_float(),
                  "double" = torch::torch_double())

  result <- base_method(exp = exp,
                        instance = instance,
                        n = 1,
                        model_class = "CoxTime",
                        inputs_ref = NULL,
                        method_pre_fun = NULL,
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
    method = "Surv_Gradient",
    method_args = list(
      target = target, instance = instance, times_input = times_input,
      include_time = include_time, dtype = dtype_name
    )
  ))
  class(result) <- c("surv_result", class(result))

  result
}


# DeepHit ----------------------------------------------------------------------
#'
#' @rdname surv_grad
#' @export
surv_grad.explainer_deephit <- function(exp, target = "survival", instance = 1,
                                        times_input = FALSE, batch_size = 50,
                                        dtype = "float", ...) {

  # Check arguments
  assertClass(exp, "explainer_deephit")
  assertChoice(target, c("pmf", "cif", "survival"))
  assertIntegerish(instance, lower = 1, upper = dim(exp$input_data[[1]])[1])
  assertIntegerish(batch_size, lower = 1)
  assertLogical(times_input)
  assertChoice(dtype, c("float", "double"))

  # Set dtype of all tensors
  dtype_name <- dtype
  dtype <- switch(dtype_name,
                  "float" = torch::torch_float(),
                  "double" = torch::torch_double())

  result <- base_method(exp = exp,
                        instance = instance,
                        n = 1,
                        model_class = "DeepHit",
                        inputs_ref = NULL,
                        method_pre_fun = NULL,
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
    method = "Surv_Gradient",
    method_args = list(
      target = target, instance = instance, times_input = times_input,
      dtype = dtype_name
    )
  ))
  class(result) <- c("surv_result", class(result))

  result

}
