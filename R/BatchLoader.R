
setClass("Surv_BatchLoader",
         slots = c(inputs = "list",
                   inputs_ref = "ANY",
                   method_pre_fun = "function",
                   model_pre_fun = "function",
                   scale_fun = "ANY",
                   batch_size = "numeric",
                   n_timepoints = "numeric",
                   n = "numeric",
                   total_batches = "numeric",
                   current_index = "numeric",
                   max_index = "numeric"
         )
)

Surv_BatchLoader <- function(inputs, inputs_ref, method_pre_fun, model_pre_fun,
                             scale_fun, batch_size, n_timepoints, n) {
  batch_size <- min(batch_size, dim(inputs[[1]])[1] * n_timepoints)
  if (batch_size < n_timepoints) {
    message("[survinng] Batch size is smaller than the number of timepoints. Setting ",
            "batch size (", batch_size, ") to the number of timepoints (",
            n_timepoints, ").")
    batch_size <- n_timepoints
  } else {
    batch_size <- n_timepoints * (batch_size %/% n_timepoints)
  }

  new("Surv_BatchLoader",
      inputs = if (!is.list(inputs)) list(inputs) else inputs,
      inputs_ref = if (!is.list(inputs_ref) && !is.null(inputs_ref)) list(inputs_ref) else inputs_ref,
      method_pre_fun = if (is.null(method_pre_fun)) identity else method_pre_fun,
      model_pre_fun = model_pre_fun,
      scale_fun = scale_fun,
      batch_size = batch_size,
      n_timepoints = n_timepoints,
      n = n,
      total_batches = ceiling(dim(inputs[[1]])[1] * n_timepoints / batch_size),
      current_index = 0,
      max_index = dim(inputs[[1]])[1]
  )
}

setGeneric("update", function(object) standardGeneric("update"))

setMethod("update", "Surv_BatchLoader", function(object) {
  # Get instance indices for the current batch
  instances_per_batch <- object@batch_size / object@n_timepoints
  idx <- seq_len(instances_per_batch) + object@current_index
  idx <- idx[idx <= object@max_index]

  # Update the current index
  if (object@current_index + length(idx) > object@max_index) {
    warning("[survinng] You'are trying to get more batches than available. ",
            "Resetting the current index to 0.")
    object@current_index <- 0
  } else {
    object@current_index <- object@current_index + length(idx)
  }

  invisible(object)
})


setGeneric("get_batch", function(object) standardGeneric("get_batch"))

setMethod("get_batch", "Surv_BatchLoader", function(object) {

  # Get instance indices for the current batch
  instances_per_batch <- object@batch_size / object@n_timepoints
  idx <- seq_len(instances_per_batch) + object@current_index
  idx <- idx[idx <= object@max_index]

  # Get inputs and reference inputs
  inputs <- lapply(object@inputs, function(x) x[idx, , drop = FALSE])
  if (!is.null(object@inputs_ref)) {
    inputs_ref <- lapply(object@inputs_ref, function(x) x[idx, , drop = FALSE])
  } else {
    inputs_ref <- NULL
  }

  # Apply the method-specific preprocessing function (e.g. adding noise)
  inputs <- object@method_pre_fun(inputs)
  if (!is.null(inputs_ref)) {
    inputs_ref <- object@method_pre_fun(inputs_ref)
  }

  # Preprocess inputs (e.g. add the time dimension for CoxTime models)
  inputs <- object@model_pre_fun(inputs) #lapply(inputs,)
  if (!is.null(inputs_ref)) {
    inputs_ref <- object@model_pre_fun(inputs_ref) #lapply(inputs_ref, object@model_pre_fun)
  }

  # Combine inputs and inputs_ref according to the method
  if (!is.null(object@scale_fun)) {
    # Check if dimensions match
    lapply(seq_along(inputs),
           function(i) assertTensorDim(inputs[[i]], inputs_ref[[i]],
                                       n1 = "'data'", n2 = "'data_ref'"))
    # Scale the inputs
    batch <- lapply(seq_along(inputs), function(i) {
      dim <- c(dim(inputs[[i]])[1], rep(1, length(dim(inputs[[i]])) - 1))
      scale <- get_scale(object, dim, idx)
      inputs_ref[[i]] + scale * (inputs[[i]] - inputs_ref[[i]])
    })
  } else {
    batch <- inputs
  }

  # Get the instance indices for the current batch
  num <- data.frame(table((idx - 1) %/% object@n + 1, dnn = "instance_id"))

  list(batch = batch, inputs = inputs, inputs_ref = inputs_ref, num = num,
       idx = idx)
})

setGeneric("get_scale", function(object, dim, idx, rep_time = TRUE) standardGeneric("get_scale"))

setMethod("get_scale", "Surv_BatchLoader", function(object, dim, idx, rep_time = TRUE) {
  if (!is.null(object@scale_fun)) {
    n_timepoints <- if (rep_time) object@n_timepoints else 1
    scale <- torch::torch_repeat_interleave(
      object@scale_fun(n = object@n)[(idx - 1) %% object@n + 1],
      repeats = as.integer(n_timepoints))$reshape(dim)
  } else {
    scale <- 1
  }

  scale
})
