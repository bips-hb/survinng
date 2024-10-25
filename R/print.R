#' Print function for surv_result objects
#'
#' @param x An object of class "surv_result"
#' @param ... Additional arguments (not used)
#' @export
print.surv_result <- function(x, ...) {
  # Ensure it's a surv_result object
  if (!inherits(x, "surv_result")) {
    stop("The object must be of class 'surv_result'.")
  }

  cat("\nSurvival Gradient Result Summary\n")
  cat("--------------------------------\n")

  # Model information
  if (!is.null(x$model_class)) {
    cat("Model class:          ", x$model_class, "\n")
  }

  # Method information
  if (!is.null(x$method)) {
    cat("Method:               ", x$method, "\n")
  }

  # Target information
  if (!is.null(x$method_args$target)) {
    cat("Target:               ", x$method_args$target, "\n")
  }

  # Instance information
  if (!is.null(x$method_args$instance)) {
    cat("Instance:             ", x$method_args$instance, "\n")
  }

  # Timepoints (length)
  if (!is.null(x$time)) {
    cat("Number of timepoints: ", length(x$time), "\n")
  }

  # Number of competing risks (if applicable)
  if (!is.null(x$competing_risks) && x$competing_risks) {
    cat("Competing risks:      Yes\n")
  } else {
    cat("Competing risks:      No\n")
  }

  # Predicted outcomes summary (if available)
  if (!is.null(x$pred)) {
    cat("\nPrediction Summary:\n")
    cat("  - Mean prediction:     ", mean(x$pred, na.rm = TRUE), "\n")
    cat("  - Range of prediction: [", min(x$pred, na.rm = TRUE), ", ", max(x$pred, na.rm = TRUE), "]\n")
  }

  # If there are reference predictions (i.e., comparison to some reference input)
  if (!is.null(x$pred_diff)) {
    cat("\nDifference from Reference Predictions:\n")
    cat("  - Mean difference:     ", mean(x$pred_diff, na.rm = TRUE), "\n")
    cat("  - Range of difference: [", min(x$pred_diff, na.rm = TRUE), ", ", max(x$pred_diff, na.rm = TRUE), "]\n")

    if (!is.null(x$pred_diff_q1)) {
      cat("  - 1st Quartile Diff:   ", mean(x$pred_diff_q1, na.rm = TRUE), "\n")
    }

    if (!is.null(x$pred_diff_q3)) {
      cat("  - 3rd Quartile Diff:   ", mean(x$pred_diff_q3, na.rm = TRUE), "\n")
    }
  }

  # Display any warnings if applicable
  if (!is.null(x$warnings)) {
    cat("\nWarnings:\n")
    cat(x$warnings, "\n")
  }

  cat("\nEnd of Summary\n")
}


#' Custom print method for explainer objects
#'
#' This function prints a summary of the explainer object.
#'
#' @param x An object of class 'explainer_coxtime', 'explainer_deepsurv', or
#' 'explainer_deephit'.
#' @param ... Additional arguments (not used).
#' @rdname print.explainer
#' @export
print.explainer_coxtime <- function(x, ...) {
  # Check if the object is of the correct class
  if (!inherits(x, "explainer_coxtime")) {
    stop("The object is not of class 'explainer_coxtime'.")
  }

  # Start printing summary
  cat("Explainer for CoxTime model\n")
  cat("-----------------------------------\n")

  # Model information
  cat("Model Class: CoxTime\n")

  # Data summary
  input_data <- x$input_data
  n_instances <- if (is.list(input_data)) length(input_data[[1]]) else length(input_data)

  cat("Number of instances in the input data: ", n_instances, "\n")

  # Display information about timepoints
  if (!is.null(x$model$t_orig)) {
    cat("Number of timepoints in the model: ", length(x$model$t_orig), "\n")
  }

  # Additional model details
  cat("Model parameters:\n")
  cat(" - Number of features: ", length(x$input_names), "\n")

  if (!is.null(x$model$base_hazard)) {
    cat(" - Baseline hazard function: Yes\n")
  } else {
    cat(" - Baseline hazard function: No\n")
  }

  if (!is.null(x$model$preprocess_fun)) {
    cat(" - Preprocessing function applied: Yes\n")
  } else {
    cat(" - Preprocessing function applied: No\n")
  }

  # Show some of the input names (e.g., features)
  cat("\nFeatures (first 5 shown):\n")
  cat(paste(x$input_names[1:min(5, length(x$input_names))], collapse = ", "), "\n")

  # Conclude
  cat("-----------------------------------\n")
  cat("To see more details, explore the individual elements of the object.\n")

  invisible(x)
}

#' @rdname print.explainer
#' @export
print.explainer_deepsurv <- function(x, ...) {
  # Check if the object is of the correct class
  if (!inherits(x, "explainer_deepsurv")) {
    stop("The object is not of class 'explainer_deepsurv'.")
  }

  # Start printing summary
  cat("Explainer for DeepSurv model\n")
  cat("-----------------------------------\n")

  # Model information
  cat("Model Class: DeepSurv\n")

  # Data summary
  input_data <- x$input_data
  n_instances <- if (is.list(input_data)) length(input_data[[1]]) else length(input_data)

  cat("Number of instances in the input data: ", n_instances, "\n")

  # Display timepoints (DeepSurv may not have timepoints like CoxTime)
  if (!is.null(x$model$t_orig)) {
    cat("Number of timepoints in the model: ", length(x$model$t_orig), "\n")
  }

  # Additional model details
  cat("Model parameters:\n")
  cat(" - Number of features: ", length(x$input_names), "\n")

  if (!is.null(x$model$base_hazard)) {
    cat(" - Baseline hazard function: Yes\n")
  } else {
    cat(" - Baseline hazard function: No\n")
  }

  if (!is.null(x$model$preprocess_fun)) {
    cat(" - Preprocessing function applied: Yes\n")
  } else {
    cat(" - Preprocessing function applied: No\n")
  }

  # Show some of the input names (e.g., features)
  cat("\nFeatures (first 5 shown):\n")
  cat(paste(x$input_names[1:min(5, length(x$input_names))], collapse = ", "), "\n")

  # Conclude
  cat("-----------------------------------\n")
  cat("To see more details, explore the individual elements of the object.\n")

  invisible(x)
}

#' @rdname print.explainer
#' @export
print.explainer_deephit <- function(x, ...) {
  # Check if the object is of the correct class
  if (!inherits(x, "explainer_deephit")) {
    stop("The object is not of class 'explainer_deephit'.")
  }

  # Start printing summary
  cat("Explainer for DeepHit model\n")
  cat("-----------------------------------\n")

  # Model information
  cat("Model Class: DeepHit\n")

  # Data summary
  input_data <- x$input_data
  n_instances <- if (is.list(input_data)) length(input_data[[1]]) else length(input_data)

  cat("Number of instances in the input data: ", n_instances, "\n")

  # Check and print the number of competing risks
  if (!is.null(x$model$competing_risks)) {
    cat("Number of competing risks: ", x$model$competing_risks, "\n")
  } else {
    cat("Competing risks: Not specified\n")
  }

  # Display time bins (DeepHit model typically uses discretized time bins)
  if (!is.null(x$model$time_bins)) {
    cat("Number of time bins in the model: ", length(x$model$time_bins), "\n")
  }

  # Additional model details
  cat("Model parameters:\n")
  cat(" - Number of features: ", length(x$input_names), "\n")

  # If the model has a preprocessing function
  if (!is.null(x$model$preprocess_fun)) {
    cat(" - Preprocessing function applied: Yes\n")
  } else {
    cat(" - Preprocessing function applied: No\n")
  }

  # Show some of the input names (e.g., features)
  cat("\nFeatures (first 5 shown):\n")
  cat(paste(x$input_names[1:min(5, length(x$input_names))], collapse = ", "), "\n")

  # Conclude
  cat("-----------------------------------\n")
  cat("To see more details, explore the individual elements of the object.\n")

  invisible(x)
}

#' Print method for extracted pycox survival model
#'
#' @param x An object of class `extracted_survivalmodels_coxtime`,
#' `extracted_survivalmodels_deepsurv`, or `extracted_survivalmodels_deephit`.
#' @param ... Additional arguments (not used).
#' @rdname print.extracted_survivalmodels
#' @export
print.extracted_survivalmodels_coxtime <- function(x, ...) {
  cat("Extracted CoxTime Survival Model:\n\n")

  # Print input shape and names
  cat("Input Shape:\n")
  print(x$input_shape)
  cat("\nInput Names:\n")
  print(x$input_names)

  # Print model parameters
  cat("\nModel Parameters:\n")
  cat("Activation Function: ", x$activation, "\n")
  cat("Number of Nodes: ", paste(x$num_nodes, collapse = ", "), "\n")
  cat("Batch Normalization: ", x$batch_norm, "\n")
  cat("Dropout Rate: ", x$dropout, "\n")

  # Print baseline hazard
  cat("\nBaseline Hazard:\n")
  print(x$base_hazard)

  # Print time transformation functions if applicable
  if (!is.null(x$labtrans)) {
    cat("\nTime Transformation Functions:\n")
    cat("Transform Function: ", deparse(substitute(x$labtrans$transform)), "\n")
    cat("Inverse Transform Function: ", deparse(substitute(x$labtrans$inv_transform)), "\n")
  }

  invisible(x)
}

#' @rdname print.extracted_survivalmodels
#' @export
print.extracted_survivalmodels_deepsurv <- function(x, ...) {
  cat("Extracted DeepSurv Survival Model:\n\n")

  # Print input shape and names
  cat("Input Shape:\n")
  print(x$input_shape)
  cat("\nInput Names:\n")
  print(x$input_names)

  # Print model parameters
  cat("\nModel Parameters:\n")
  cat("Activation Function: ", x$activation, "\n")
  cat("Number of Nodes: ", paste(x$num_nodes, collapse = ", "), "\n")
  cat("Batch Normalization: ", x$batch_norm, "\n")
  cat("Dropout Rate: ", x$dropout, "\n")

  # Print baseline hazard
  cat("\nBaseline Hazard:\n")
  print(x$base_hazard)

  invisible(x)
}

#' @rdname print.extracted_survivalmodels
#' @export
print.extracted_survivalmodels_deephit <- function(x, ...) {
  cat("Extracted DeepHit Survival Model:\n\n")

  # Print input shape and names
  cat("Input Shape:\n")
  print(x$input_shape)
  cat("\nInput Names:\n")
  print(x$input_names)

  # Print model parameters
  cat("\nModel Parameters:\n")
  cat("Activation Function: ", x$activation, "\n")
  cat("Number of Nodes: ", paste(x$num_nodes, collapse = ", "), "\n")
  cat("Batch Normalization: ", x$batch_norm, "\n")
  cat("Dropout Rate: ", x$dropout, "\n")

  # Print time bins information
  cat("\nTime Bins:\n")
  print(x$time_bins)

  invisible(x) # Return the object invisibly
}
