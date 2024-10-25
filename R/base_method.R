

base_method <- function(exp, instance, n = 1, model_class, inputs_ref = NULL,
                        method_pre_fun = NULL, scale_tensor = NULL,
                        n_timepoints = 1, return_out = FALSE,
                        times_input = FALSE, remove_time = TRUE,
                        batch_size = 10, target = "survival",
                        num_samples = 1, dtype = torch::torch_float()) {

  # Preprocess inputs ----------------------------------------------------------
  inputs_ref_orig <- inputs_ref

  # Set dtype of the model
  exp$model$set_dtype(dtype)

  # Select and convert the inputs to tensors
  inputs <- to_tensor(exp$input_data, instance, repeats = n * num_samples, dtype = dtype)
  if (!is.null(inputs_ref)) {
    inputs_ref <- to_tensor(inputs_ref, seq_len(dim(inputs_ref[[1]])[1]), repeats = 1, dtype = dtype)
  }

  # Apply the method-specific preprocessing function (e.g. adding noise)
  if (is.null(method_pre_fun)) method_pre_fun <- identity
  inputs <- method_pre_fun(inputs)
  if (!is.null(inputs_ref)) {
    inputs_ref <- method_pre_fun(inputs_ref)
  }

  # Preprocess inputs (e.g. add the time dimension for CoxTime models)
  inputs <- lapply(inputs, exp$model$preprocess_fun)
  if (!is.null(inputs_ref)) {
    inputs_ref <- lapply(inputs_ref, exp$model$preprocess_fun)
  }

  # Combine inputs and inputs_ref according to the method
  if (!is.null(scale_tensor)) {
    # Check if dimensions match
    lapply(seq_along(inputs),
           function(i) assertTensorDim(inputs[[i]], inputs_ref[[i]],
                                       n1 = "'data'", n2 = "'data_ref'"))
    # Scale the inputs
    inputs_combined <- lapply(seq_along(inputs), function(i) {
      inputs_ref[[i]] + scale_tensor * (inputs[[i]] - inputs_ref[[i]])
    })
  } else {
    inputs_combined <- inputs
  }

  # Split inputs into batches
  batches <- split_batches(inputs_combined, batch_size, n_timepoints = n_timepoints, n = n * num_samples)

  # Calculate the method (batch-wise) ------------------------------------------
  res <- lapply(batches, function(batch) {
    message("Processing batch ", batch$idx[1], " to ", batch$idx[2], "...")

    # Calculate the gradients
    # Note: For all targets, the gradients are w.r.t. the hazard and
    # will be transformed to the target of interest afterwards
    grads <- exp$model$calc_gradients(batch$batch,
                                      target = target,
                                      return_out = return_out,
                                      use_base_hazard = TRUE)
    outs <- grads$outs[[1]]
    grads <- grads$grads

    # Since we allow models with multiple input layers, we need to
    # iterate over the gradients of each input layer
    grads <- lapply(seq_along(grads), function(i) {
      grad <- grads[[i]]

      # Model-dependent postprocessing
      if (model_class == "CoxTime") {
        # Output (batch_size * t, out_features) -> (batch_size, out_features, 1, t)
        grad <- grad$reshape(c(-1, length(exp$model$t), grad$size(-1), 1))$movedim(2, -1)

        # Remove the time dimension (only possible for CoxTime models)
        if (remove_time) {
          grad <- grad[,seq_len(dim(grad)[2] - 1), , drop = FALSE]
        }

        # Aggregate gradients for the cum. hazard or survival outcome
        # Note: This is allowed due to the linearity of the gradient operator
        if (target == "cum_hazard") {
          grad <- grad$cumsum(-1)
        } else if (target == "survival") {
          grad <- -grad$cumsum(-1) * (-outs$cumsum(-1))$exp()
        }
      } else if (model_class == "DeepSurv") {
        # Reshape grads (batch_size, input_features) --> (batch_size, input_features, 1, t)
        grad <- grad$unsqueeze(-1) * torch::torch_ones(c(dim(grad), length(exp$model$t)), dtype = dtype)
        grad <- grad$unsqueeze(-2)

        # The baseline hazard is not used in the DeepSurv model, so we need to
        # multiply the gradients and outputs with the baseline hazard
        base_haz <- torch::torch_tensor(exp$model$base_hazard$hazard, dtype = dtype)
        base_haz <- base_haz$reshape(c(rep(1, grad$dim() - 1), -1))
        grad <- grad * base_haz
        outs <- outs * base_haz

        # Aggregate gradients for the cum. hazard or survival outcome
        # Note: This is allowed due to the linearity of the gradient operator
        if (target == "cum_hazard") {
          grad <- grad$cumsum(-1)
        } else if (target == "survival") {
          grad <- -grad$cumsum(-1) * (-outs$cumsum(-1))$exp()
        }
      } else if (model_class == "DeepHit") {
        # Nothing to do here :)
      }

      # Multiply the gradients with the inputs
      if (times_input) {
        # Remove the time dimension (only possible for CoxTime models)
        feat_idx <- seq_len(dim(grad)[2])

        if (is.null(inputs_ref)) {
          grad <- grad * exp$model$postprocess_fun(batch$batch[[i]])[, feat_idx,, drop = FALSE]
        } else {
          rows <- batch$idx[1]:batch$idx[2]
          input_diff <- inputs[[i]][rows,, drop = FALSE] - inputs_ref[[i]][rows,, drop = FALSE]
          grad <- grad * exp$model$postprocess_fun(input_diff)[, feat_idx,, drop = FALSE]
        }
      }

      # Take mean over repeated values
      grad <- lapply(grad$split(batch$num$Freq, dim = 1L),
               torch::torch_sum, dim = 1, keepdim = TRUE)
      names(grad) <- batch$num$instance_id

      grad
    })
  })

  # Postprocess the results ----------------------------------------------------
  # Note: The object 'res' is a list of lists, where each list contains the
  # gradients for each input layer. We need to combine these lists into a
  # single list of gradients.

  # Get the timepoints
  timepoints <- switch(model_class,
                       CoxTime = exp$model$t_orig,
                       DeepSurv = exp$model$t_orig,
                       DeepHit = exp$model$time_bins)

  # Add the time dimension to the gradients if necessary
  if (model_class == "CoxTime" && !remove_time) {
    add_time <- TRUE
  } else {
    add_time <- FALSE
  }

  # For competing risks, get the number of events
  num_risks <- rev(dim(res[[1]][[1]][[1]]))[2]
  event_names <- paste0("Event ", seq_len(num_risks))

  # Combine the results of all batches
  result <- combine_batch_grads(res, exp$input_names, timepoints, add_time,
                                event_names, n = n * num_samples)

  # Calculate the outcomes -----------------------------------------------------
  # Calculate the predictions
  outs <- to_tensor(exp$input_data, instance, repeats = 1) |>
    lapply(FUN = exp$model$preprocess_fun) |>
    exp$model$forward(target = target, use_base_hazard = TRUE) |>
    lapply(FUN = torch::torch_squeeze, dim = c(2, 3))

  # Calculate the reference predictions
  if (!is.null(inputs_ref_orig)) {
    if (num_samples > 1) {
      data_ref <- exp$input_data
      idx <- seq_len(nrow(exp$input_data[[1]]))
      add_quantiles <- TRUE
    } else {
      data_ref <- inputs_ref_orig
      idx <- 1
      add_quantiles <- FALSE
    }
    outs_ref <- to_tensor(data_ref, idx, repeats = 1) |>
      lapply(FUN = exp$model$preprocess_fun) |>
      exp$model$forward(target = target, use_base_hazard = TRUE) |>
      lapply(FUN = function(a) {
        a <- a$squeeze(dim = c(2, 3))
        list(
          mean = torch::torch_mean(a, dim = 1, keepdim = TRUE),
          quantile = torch::torch_quantile(a, q = torch::torch_tensor(c(0.25, 0.75), dtype = dtype), dim = 1)
        )
      })

    res_pred <-  list(
      pred = as.array(outs[[1]]),
      pred_diff = as.array(outs[[1]] - outs_ref[[1]]$mean)
    )

    if (add_quantiles) {
      res_pred$pred_diff_q1 <- as.array(outs[[1]] - outs_ref[[1]]$quantile[1,])
      res_pred$pred_diff_q3 <- as.array(outs[[1]] - outs_ref[[1]]$quantile[2,])
    }
  } else {
    res_pred <- list(pred = as.array(outs[[1]]))
  }

  # Create the result object ---------------------------------------------------
  result <- list(
    res = result,
    time = timepoints,
    competing_risks = num_risks > 1
  )
  result <- append(result, res_pred)

  result
}
