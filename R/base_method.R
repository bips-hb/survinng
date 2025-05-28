

base_method <- function(exp, instance, n = 1, model_class, inputs_ref = NULL,
                        method_pre_fun = NULL, scale_fun = NULL,
                        n_timepoints = 1, return_out = FALSE,
                        times_input = FALSE, remove_time = TRUE,
                        batch_size = 10, target = "survival",
                        num_samples = 1, dtype = torch::torch_float(),
                        second_order = FALSE) {

  # Preprocess inputs ----------------------------------------------------------
  inputs_ref_orig <- inputs_ref

  # Set dtype of the model
  exp$model$set_dtype(dtype)

  # Select and convert the inputs to tensors
  inputs <- to_tensor(exp$input_data, instance, repeats = n * num_samples, dtype = dtype)
  if (!is.null(inputs_ref)) {
    inputs_ref <- to_tensor(inputs_ref, seq_len(dim(inputs_ref[[1]])[1]), repeats = 1, dtype = dtype)
  }

  # Check dimension of the data
  final_batch_size <- min(batch_size, dim(inputs[[1]])[1] * n_timepoints)
  if (batch_size < n_timepoints) {
    final_batch_size <- n_timepoints
  } else {
    final_batch_size <- n_timepoints * (final_batch_size %/% n_timepoints)
  }
  num_dims <- sum(sapply(inputs, function(x) final_batch_size * prod(dim(x)[-1])))
  if (num_dims > 2e6) {
    warning("The resulting tensor has ", num_dims, " entries in total per batch. ",
            "This may lead to runtime or memory issues. Consider using a ",
            "smaller batch size (currently ", batch_size, ") ",
            "or reducing the number of instances, samples or integration points.")
  }



  # Split inputs into batches
  batches <- Surv_BatchLoader(
    inputs = inputs,
    inputs_ref = inputs_ref,
    method_pre_fun = method_pre_fun,
    model_pre_fun = exp$model$preprocess_fun,
    scale_fun = scale_fun,
    batch_size = batch_size,
    n_timepoints = n_timepoints,
    n = n * num_samples)

  # Calculate the method (batch-wise) ------------------------------------------
  res <- lapply(seq_len(batches@total_batches), function(b) {

    # Get the current batch and update the batch loader
    batch <- get_batch(batches)
    batches <<- update(batches)

    # Calculate the gradients
    # Note: For all targets, the gradients are w.r.t. the hazard and
    # will be transformed to the target of interest afterwards
    grads <- exp$model$calc_gradients(batch$batch,
                                      target = target,
                                      return_out = return_out,
                                      use_base_hazard = TRUE,
                                      second_order = second_order)
    outs <- grads$outs[[1]]
    grads_2 <- grads$grads_2
    grads <- grads$grads

    if (length(grads) > 1) {
      stop("Calculated gradient 'grads' has more than one element. Please ",
           "create an GitHub issue")
    }
    grads <- grads[[1]]

    # Since we allow models with multiple input layers, we need to
    # iterate over the gradients of each input layer
    grads <- lapply(seq_along(grads), function(i) {
      grad <- grads[[i]]
      grad_2 <- if (second_order) grads_2[[i]] else NULL

      # Model-dependent postprocessing
      if (model_class == "CoxTime") {
        # Output (batch_size * t, in_features) -> (batch_size, in_features, 1, t)
        grad <- grad$reshape(c(-1, length(exp$model$t), grad$size()[-1], 1))$movedim(2, -1)
        out <- outs$reshape(c(outs$shape[1], rep(1, grad$dim() - outs$dim()), outs$shape[-1]))
        if (second_order) {
          # Output (batch_size * t, in_features, in_features) -> (batch_size, in_features, in_features, 1, t)
          grad_2 <- grad_2$reshape(c(-1, length(exp$model$t), grad_2$size()[-1], 1))$movedim(2, -1)
        }

        # Remove the time dimension (only possible for CoxTime models)
        if (remove_time) {
          if (!exp$model$default_preprocess) {
            warning("The model does not use the default preprocessing function. ",
                    "The time dimension will not be removed.", call. = FALSE)
          } else {
            grad <- grad[,seq_len(dim(grad)[2] - 1), , drop = FALSE]
            if (second_order) {
              grad_2 <- grad_2[, seq_len(dim(grad_2)[2] - 1), seq_len(dim(grad_2)[3] - 1), , drop = FALSE]
            }
          }
        }

        # Aggregate gradients for the cum. hazard or survival outcome
        # Note: This is allowed due to the linearity of the gradient operator
        if (target == "cum_hazard") {
          grad <- grad$cumsum(-1)
          if (second_order) {
            grad_2 <- grad_2$cumsum(-1)
          }
        } else if (target == "survival") {
          grad <- -grad$cumsum(-1) * (-out$cumsum(-1))$exp()
          if (second_order) {
            grad_2 <- -grad_2$cumsum(-1) * (-out$cumsum(-1))$exp()$unsqueeze(2)
          }
        }
      } else if (model_class == "DeepSurv") {
        # Reshape grads (batch_size, input_features) --> (batch_size, input_features, 1, t)
        time_axis <- torch::torch_ones(c(dim(grad), length(exp$model$t)), dtype = dtype)
        grad <- grad$unsqueeze(-1) * time_axis
        grad <- grad$unsqueeze(-2)
        if (second_order) {
          grad_2 <- grad_2$unsqueeze(-1) * time_axis$unsqueeze(3)
          grad_2 <- grad_2$unsqueeze(-2)
        }

        # The baseline hazard is not used in the DeepSurv model, so we need to
        # multiply the gradients and outputs with the baseline hazard
        base_haz <- torch::torch_tensor(exp$model$base_hazard$hazard, dtype = dtype)
        base_haz <- base_haz$reshape(c(rep(1, grad$dim() - 1), -1))
        out <- outs$reshape(c(outs$shape[1], rep(1, grad$dim() - outs$dim()), outs$shape[-1]))
        grad <- grad * base_haz
        out <- out * base_haz
        if (second_order) {
          grad_2 <- grad_2 * base_haz$unsqueeze(2)
        }

        # Aggregate gradients for the cum. hazard or survival outcome
        # Note: This is allowed due to the linearity of the gradient operator
        if (target == "cum_hazard") {
          grad <- grad$cumsum(-1)
          if (second_order) {
            grad_2 <- grad_2$cumsum(-1)
          }
        } else if (target == "survival") {
          grad <- -grad$cumsum(-1) * (-out$cumsum(-1))$exp()
          if (second_order) {
            grad_2 <- -grad_2$cumsum(-1) * (-out$cumsum(-1))$exp()$unsqueeze(2)
          }
        }
      } else if (model_class == "DeepHit") {
        # Nothing to do here :)
      }

      # Multiply the gradients with the inputs
      if (times_input) {
        if (is.null(inputs_ref)) {
          input <- exp$model$postprocess_fun(batch$batch[[i]])
          if (model_class == "CoxTime" && exp$model$default_preprocess) {
            # Assuming tabular input
            feat_idx <- seq_len(dim(grad)[2])
            input <- input[, feat_idx,, drop = FALSE]
          }
          grad <- grad * input
        } else {
          input_diff <- exp$model$postprocess_fun(batch$inputs[[i]] - batch$inputs_ref[[i]])
          if (model_class == "CoxTime" && exp$model$default_preprocess) {
            feat_idx <- seq_len(dim(grad)[2])
            input_diff <- input_diff[, feat_idx,, drop = FALSE]
          }
          grad <- grad * input_diff

          if (second_order) {
            input_diff <- input_diff$unsqueeze(2) * input_diff$unsqueeze(3)
            scale <- get_scale(batches, c(length(batch$idx), rep(1, length(dim(grad_2)) - 1)),
                               batch$idx, rep_time = FALSE)
            grad_2 <- grad_2 * input_diff * scale
          }
        }
      }

      # Add second order gradients to 'grad' as another feature
      if (second_order) {
        idx_diag <- torch::torch_where(torch::torch_eye(dim(grad)[2])$flatten() != 0)[[1]]
        idx <- torch::torch_where((torch::torch_ones(dim(grad)[2], dim(grad)[2])$triu() -
                                     torch::torch_eye(dim(grad)[2]))$flatten() != 0)[[1]]
        grad_2_mixed <- 2 * grad_2$flatten(start_dim = 2L, end_dim = 3L)$index_select(2, idx)
        grad_2_diag <- grad_2$flatten(start_dim = 2L, end_dim = 3L)$index_select(2, idx_diag)
        grad <- torch::torch_cat(list(grad + grad_2_diag, grad_2_mixed), dim = 2)
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

  # Get input names
  if (second_order) {
    # TODO: Currently only implemented for single input layers and 1D inputs
    feat_names <- list(list(c(
      unlist(exp$input_names),
      paste0(
        apply(combn(unlist(exp$input_names), 2), 2, paste, collapse = " x ")
      )
    )))
  } else {
    feat_names <- exp$input_names
  }

  # Combine the results of all batches
  result <- combine_batch_grads(res, feat_names, timepoints, add_time,
                                event_names, n = n * num_samples)

  # Calculate the outcomes -----------------------------------------------------
  # Calculate the predictions
  outs <- to_tensor(exp$input_data, instance, repeats = 1, dtype = dtype) |>
    exp$model$preprocess_fun() |>
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
    outs_ref <- to_tensor(data_ref, idx, repeats = 1, dtype = dtype) |>
      exp$model$preprocess_fun() |>
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
