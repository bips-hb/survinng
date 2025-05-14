
################################################################################
#                             Explain method
################################################################################

#' Explain a model
#'
#' This function is a generic method that dispatches to the appropriate
#' explain method based on the class of the model.
#'
#' @param model A model object.
#' @param data A data frame or matrix of data to explain the model.
#' @param predict_fun A function that can be used to make predictions
#' from the model. If `NULL`, the predict method of the model will be used.
#' @param model_type A string specifying the type of the survival model.
#' Possible values are "coxtime", "deephit", or "deepsurv".
#' @param baseline_hazard A data frame containing the baseline hazard. It
#' should have two columns: "time" and "hazard". This is only used for
#' "coxtime" and "deepsurv" models.
#' @param labtrans A list containing the transformation functions for the
#' time variable. It should have two elements: "transform" and
#' "transform_inv". This is highly experimental and not yet fully
#' supported.
#' @param time_bins A numeric vector specifying the time bins for the
#' "deephit" model, e.g., `c(0, 1, 2, 3)`.
#' @param preprocess_fun A function to preprocess the data before
#' making predictions, e.g., adding a time variable for a `coxtime` model.
#' This argument is highly experimental and the default values should work
#' for most cases.
#' @param postprocess_fun A function to postprocess the predictions after
#' making them. This argument is highly experimental and the default
#' values should work.
#' @param ... Unused arguments.
#'
#' @return An object of class \code{explainer} that contains the model, the
#' data, and the prediction function.
#'
#' @export
explain <- function(model,
                    data = NULL,
                    model_type = NULL,
                    baseline_hazard = NULL,
                    labtrans = NULL,
                    time_bins = NULL,
                    preprocess_fun = NULL,
                    postprocess_fun = NULL,
                    predict_fun = NULL) {
  UseMethod("explain")
}


#'
#' @rdname explain
#' @export
explain.nn_module <- function(model, data, model_type,
                              baseline_hazard = NULL,
                              labtrans = NULL,
                              time_bins = NULL,
                              preprocess_fun = NULL,
                              postprocess_fun = NULL,
                              predict_fun = NULL) {

  # Check arguments
  if (!torch::is_nn_module(model)) {
    stop("The model must be a 'torch::nn_module' object.")
  }
  assertArgData(data)
  assertCharacter(model_type)
  model_type <- tolower(model_type)
  assertChoice(model_type, c("coxtime", "deephit", "deepsurv"))
  assertFunction(predict_fun, null.ok = TRUE)
  assertFunction(preprocess_fun, null.ok = TRUE)
  assertFunction(postprocess_fun, null.ok = TRUE)

  # Create Survival model
  if (model_type == "coxtime") {
    assertDataFrame(baseline_hazard)
    assertSubset(c("time", "hazard"), colnames(baseline_hazard))
    assertList(labtrans, null.ok = TRUE)
    if (is.list(labtrans)) {
      assertSubset(c("transform", "transform_inv"), names(labtrans))
    }

    model <- CoxTime(model,
                     base_hazard = baseline_hazard,
                     labtrans = labtrans,
                     preprocess_fun = preprocess_fun,
                     postprocess_fun = postprocess_fun)
  } else if (model_type == "deephit") {
    assertNumeric(time_bins)
    model <- DeepHit(model, time_bins)
  } else if (model_type == "deepsurv") {
    assertDataFrame(baseline_hazard)
    assertSubset(c("time", "hazard"), colnames(baseline_hazard))
    assertList(labtrans, null.ok = TRUE)
    if (is.list(labtrans)) {
      assertSubset(c("transform", "transform_inv"), names(labtrans))
    }

    model <- DeepSurv(model,
                      base_hazard = baseline_hazard,
                      preprocess_fun = preprocess_fun,
                      postprocess_fun = postprocess_fun)
  } else {
    stop("Unknown model type: '", model_type, "'")
  }

  # Create input data
  if (is.list(data) & !is.data.frame(data)) {
    input_data <- data
    input_shape <- lapply(data, function(x) dim(x)[-1])
    input_names <- lapply(data, function(x) get_dimnames(x))
  } else {
    input_data <- list(data)
    input_shape <- list(dim(data)[-1])
    input_names <- list(get_dimnames(data))
  }

  res <- list(
    input_shape = input_shape,
    input_names = input_names,
    model = model,
    input_data = input_data
  )
  class(res) <- paste0("explainer_", model_type)

  res
}

#'
#' @rdname explain
#' @export
explain.extracted_survivalmodels_coxtime <- function(model, data, ...) {
  # It's very tricky to use torch in R and PyTorch within the same session
  # (the versions of LibTorch need to match). However, a session restart
  # can fix this issue: https://github.com/mlverse/torch/issues/815
  tryCatch({
    requireNamespace("torch", quietly = TRUE)
    out <- torch::torch_randn(4)
  }, error = function(e) {
    stop("[Suvinng] It seems like PyTorch is already loaded via `reticulate` in the ",
         "current session. A session restart is necessary to use `torch`. ",
         "Restart your R session to try again. \n\nOriginal error:\n", e, call. = FALSE)
  })

  # Load the underlying neural network
  num_feat <- lapply(model$input_shape, function(i) i + 1)
  net <- BaseMLP$new(num_feat, model$num_nodes, model$batch_norm,
                     model$dropout, model$activation)

  # Load the state dictionary
  state_dict <- model$state_dict
  names(state_dict) <- sub("^net.", "", names(state_dict))
  net$load_state_dict(state_dict)
  net$eval()

  # Create CoxTime module
  coxtime <- CoxTime(net, model$base_hazard, model$labtrans)

  # Create input data
  # TODO some checks here
  if (is.data.frame(data)) {
    input_data <- data[, model$input_names[[1]]]
  } else {
    input_data <- data
  }

  res <- list(
    input_shape = model$input_shape,
    input_names = model$input_names,
    model = coxtime,
    data = list(data),
    input_data = list(input_data)
  )
  class(res) <- "explainer_coxtime"

  res
}



#'
#' @rdname explain
#' @export
explain.extracted_survivalmodels_deephit <- function(model, data, ...) {
  # It's very tricky to use torch in R and PyTorch within the same session
  # (the versions of LibTorch need to match). However, a session restart
  # can fix this issue: https://github.com/mlverse/torch/issues/815
  tryCatch({
    requireNamespace("torch", quietly = TRUE)
    out <- torch::torch_randn(4)
  }, error = function(e) {
    stop("[Suvinng] It seems like PyTorch is already loaded via `reticulate` in the ",
         "current session. A session restart is necessary to use `torch`. ",
         "Restart your R session to try again. \n\nOriginal error:\n", e, call. = FALSE)
  })

  # Load the underlying neural network
  net <- BaseMLP$new(model$input_shape, model$num_nodes, model$batch_norm,
                     model$dropout, model$activation, out_features = length(model$time_bins),
                     out_bias = TRUE)

  # Load the state dictionary
  state_dict <- model$state_dict
  net$load_state_dict(state_dict)
  net$eval()

  # Create CoxTime module
  deephit <- DeepHit(net, model$time_bins)

  # Create input data
  # TODO some checks here
  if (is.data.frame(data)) {
    input_data <- data[, model$input_names[[1]]]
  } else {
    input_data <- data
  }

  res <- list(
    input_shape = model$input_shape,
    input_names = model$input_names,
    model = deephit,
    data = list(data),
    input_data = list(input_data)
  )
  class(res) <- "explainer_deephit"

  res
}


# DeepSurv ---------------------------------------------------------------------

#'
#' @rdname explain
#' @export
explain.extracted_survivalmodels_deepsurv <- function(model, data, ...) {
  # It's very tricky to use torch in R and PyTorch within the same session
  # (the versions of LibTorch need to match). However, a session restart
  # can fix this issue: https://github.com/mlverse/torch/issues/815
  tryCatch({
    requireNamespace("torch", quietly = TRUE)
    out <- torch::torch_randn(4)
  }, error = function(e) {
    stop("[Suvinng] It seems like PyTorch is already loaded via `reticulate` in the ",
         "current session. A session restart is necessary to use `torch`. ",
         "Restart your R session to try again. \n\nOriginal error:\n", e, call. = FALSE)
  })

  # Load the underlying neural network
  net <- BaseMLP$new(model$input_shape, model$num_nodes, model$batch_norm,
                     model$dropout, model$activation, out_features = 1,
                     out_bias = FALSE)

  # Load the state dictionary
  state_dict <- model$state_dict
  net$load_state_dict(state_dict)
  net$eval()

  # Create CoxTime module
  deepsurv <- DeepSurv(net, model$base_hazard, NULL)

  # Create input data
  # TODO some checks here
  if (is.data.frame(data)) {
    input_data <- data[, model$input_names[[1]]]
  } else {
    input_data <- data
  }

  res <- list(
    input_shape = model$input_shape,
    input_names = model$input_names,
    model = deepsurv,
    data = list(data),
    input_data = list(input_data)
  )
  class(res) <- "explainer_deepsurv"

  res
}

