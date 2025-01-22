# Extract models from `survivalmodels` package ---------------------------------

#' Extract model information from a `survivalmodels` object
#'
#' @rdname extract_model
#' @export
extract_model <- function(x, path = NULL, num_basehazard = 200L) {
  UseMethod("extract_model")
}

# CoxTime model ----------------------------------------------------------------
#' @rdname extract_model
#' @export
extract_model.coxtime <- function(x, path = NULL, num_basehazard = 200L) {
  assertIntegerish(num_basehazard)

  symbol_to_value <- function(a) {
    if (is.symbol(a)) {
      a <- eval.parent(a, n = 2)
    }
    a
  }

  # Extract model information
  res <- list(
    input_shape = list(ncol(x$x)),
    input_names = list(x$xnames),
    state_dict =  lapply(x$model$net$state_dict(), function(a) a$data$numpy()),
    activation =  if (is.null(x$call$activation)) "relu" else symbol_to_value(x$call$activation),
    num_nodes = if (is.null(x$call$num_nodes)) c(32L, 32L) else symbol_to_value(x$call$num_nodes),
    batch_norm = if (is.null(x$call$batch_norm)) TRUE else symbol_to_value(x$call$batch_norm),
    dropout = if (is.null(x$call$dropout)) 0 else symbol_to_value(x$call$dropout)
  )

  # Get baseline hazard
  baseline_hazard <- x$model$compute_baseline_hazards(sample = as.integer(num_basehazard))
  base_hazard <- data.frame(time = as.numeric(names(baseline_hazard)),
                            hazard = as.numeric(baseline_hazard))

  # Get time transformation
  if (isTRUE(x$call$standardize_time)) {
    if (isTRUE(x$call$log_duration)) {
      time <- log1p(x$y[, 1])
    } else {
      time <- x$y[, 1]
    }
    mean <- if (isTRUE(x$call$with_mean)) mean(time) else 0
    std <- if (isTRUE(x$call$with_std)) sd(time) else 1

    labtrans <- list(
      transform = function(a) (a - mean) / std,
      inv_transform = function(a) exp(a * std + mean) - 1
    )
  } else {
    labtrans = list(
      transform = function(a) a,
      inv_transform = function(a) a
    )
  }

  res <- c(res, list(base_hazard = base_hazard, labtrans = labtrans))

  class(res) <- "extracted_survivalmodels_coxtime"

  # Save model if path is provided
  if (!is.null(path)) {
    saveRDS(res, file = path)
  }

  res
}


# DeepHit model ----------------------------------------------------------------
#' @rdname extract_model
#' @export
extract_model.deephit <- function(x, path = NULL, ...) {

  symbol_to_value <- function(a) {
    if (is.symbol(a)) {
      a <- eval.parent(a, n = 2)
    }
    a
  }

  # Extract model information
  res <- list(
    input_shape = list(ncol(x$x)),
    input_names = list(x$xnames),
    state_dict =  lapply(x$model$net$state_dict(), function(a) a$data$numpy()),
    activation =  if (is.null(x$call$activation)) "relu" else symbol_to_value(x$call$activation),
    num_nodes = if (is.null(x$call$num_nodes)) c(32L, 32L) else symbol_to_value(x$call$num_nodes),
    batch_norm = if (is.null(x$call$batch_norm)) TRUE else symbol_to_value(x$call$batch_norm),
    dropout = if (is.null(x$call$dropout)) 0 else symbol_to_value(x$call$dropout),
    time_bins = x$model$duration_index
  )

  class(res) <- "extracted_survivalmodels_deephit"

  # Save model if path is provided
  if (!is.null(path)) {
    saveRDS(res, file = path)
  }

  res
}


# DeepSurv model ----------------------------------------------------------------
#' @rdname extract_model
#' @export
extract_model.deepsurv <- function(x, path = NULL, num_basehazard = 200L) {
  assertIntegerish(num_basehazard)

  symbol_to_value <- function(a) {
    if (is.symbol(a)) {
      a <- eval.parent(a, n = 2)
    }
    a
  }

  # Extract model information
  res <- list(
    input_shape = list(ncol(x$x)),
    input_names = list(x$xnames),
    state_dict =  lapply(x$model$net$state_dict(), function(a) a$data$numpy()),
    activation =  if (is.null(x$call$activation)) "relu" else symbol_to_value(x$call$activation),
    num_nodes = if (is.null(x$call$num_nodes)) c(32L, 32L) else symbol_to_value(x$call$num_nodes),
    batch_norm = if (is.null(x$call$batch_norm)) TRUE else symbol_to_value(x$call$batch_norm),
    dropout = if (is.null(x$call$dropout)) 0 else symbol_to_value(x$call$dropout)
  )

  # Get baseline hazard
  baseline_hazard <- x$model$compute_baseline_hazards(sample = as.integer(num_basehazard))
  base_hazard <- data.frame(time = as.numeric(names(baseline_hazard)),
                            hazard = as.numeric(baseline_hazard))

  res <- c(res, list(base_hazard = base_hazard))

  class(res) <- "extracted_survivalmodels_deepsurv"

  # Save model if path is provided
  if (!is.null(path)) {
    saveRDS(res, file = path)
  }

  res
}
