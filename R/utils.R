checkTensor <- function(tensor, dim = NULL, null.ok = FALSE) {
  if (is.null(tensor) & null.ok) {
    return(TRUE)
  }
  if (!inherits(tensor, "torch_tensor")) {
    return(paste0("Must be a 'torch_tensor', not '",
           paste0(class(tensor), collapse = "', '"), "'"))
  }
  if (!is.null(dim)) {
    if (any(dim != dim(tensor))) {
      return(paste0("Dimensions do not match: Tensor has (",
                    paste0(dim(tensor), collapse = ", "), "), but (",
                    paste0(dim, collapse = ", "), ") is required!"))
    }
  }

  TRUE
}

checkTensorDim <- function(tensor1, tensor2,
                           name1 = "Tensor1", name2 = "Tensor2") {
  dim1 <- tensor1$shape
  dim2 <- tensor2$shape

  mess <- paste0("Missmatch in the number of dimensions: ", name1,
                 " has a shape of (*, ", paste0(dim1[-1], collapse = ", "), ")",
                 " and ", name2, " of (*, ", paste0(dim2[-1], collapse = ", "), ").")

  if (length(dim1) != length(dim2)) {
    return(mess)
  } else if (any(dim1[-1] != dim2[-1])) {
    return(mess)
  }

  TRUE
}

assertTensorDim <- function(t1, t2, n1, n2) {
  res <- checkTensorDim(t1, t2, n1, n2)
  if (!isTRUE(res)) {
    stop(res, call. = FALSE)
  }

  NULL
}

assertArgData <- function(data, null.ok = FALSE) {
  if (!is.list(data)) {
    data_list <- list(data)
  } else {
    data_list <- data
  }

  lapply(data_list, function(x) {
    assert(
      checkMatrix(x, null.ok = null.ok),
      checkArray(x, null.ok = null.ok),
      checkTensor(x, null.ok = null.ok),
      checkDataFrame(x, null.ok = null.ok),
      checkDataTable(x, null.ok = null.ok)
    )
  })

  invisible(data)
}

combine_batch_grads <- function(res, feat_names, timepoints, include_time = FALSE,
                                event_names = NULL, n = 1) {
  combine_same_instance <- function(a) {
    a <- unlist(a, recursive = FALSE)
    lapply(unique(names(a)), function(instance_id) {
      Reduce("+", a[names(a) == instance_id]) / n
    })
  }

  grads <- lapply(seq_along(res[[1]]), function(input_layer) {
    lapply(res, function(x) x[[input_layer]]) |>
      combine_same_instance() |>
      torch::torch_cat(dim = 1) |>
      set_names(feat_names = feat_names[[input_layer]],
                include_time = include_time,
                event_names = event_names,
                timepoints = timepoints)
  })
  if (length(grads) == 1) grads <- grads[[1]]

  grads
}


# Convert to torch tensor and repeat rows --------------------------------------
to_tensor <- function(x, instance, repeats = 1) {
  # Convert to list if not already
  if (!is.list(x)) {
    x <- list(x)
  }

  lapply(x, function(i) {
    if (inherits(i, "torch_tensor")) {
      res <- i[instance,, drop = FALSE]
    } else {
      res <- torch::torch_tensor(as.matrix(i[instance,,drop = FALSE]))
    }

    # Convert to tensor and repeat rows
    res <- res$repeat_interleave(repeats = as.integer(repeats), dim = 1)

    res
  })
}

# Split tensor into batches ----------------------------------------------------
split_batches <- function(inputs, batch_size, n_timepoints, n = 1) {
  # Calculate batch size
  batch_size <- max(1, batch_size) * n_timepoints
  rows_per_instance <- n * n_timepoints

  # Split inputs into batches
  total_rows <- if (is.list(inputs)) inputs[[1]]$shape[1] else inputs$shape[1]
  idx <- lapply(seq(1, total_rows, by = batch_size), function(i) {
    c(i, min(i + batch_size - 1, total_rows))
  })
  instance_idx <- rep(seq_len(total_rows %/% rows_per_instance), each = n)
  lapply(idx, function(i) {
    if (is.list(inputs)) {
      list(
        batch = lapply(inputs, function(x) x[i[1]:i[2],,drop = FALSE]),
        num = data.frame(table(
          instance_idx[(((i[1] - 1) %/% n_timepoints) + 1):(i[2] %/% n_timepoints)],
          dnn = "instance_id")),
        idx = i
      )
    } else {
      list(
        batch = inputs[i[1]:i[2],,drop = FALSE],
        num = data.frame(table(
          instance_idx[(((i[1] - 1) %/% n_timepoints) + 1):(i[2] %/% n_timepoints)],
          dnn = "instance_id")),
        idx = i
      )
    }
  })
}

# split_batches <- function(inputs, batch_size, n_timepoints, n = NULL) {
#   # Calculate batch size
#   if (!is.null(n)) {
#     batch_size <- max(1, batch_size %/% n) * n * n_timepoints
#     rows_per_instance <- n * n_timepoints
#   } else {
#     batch_size <- batch_size * n_timepoints
#     rows_per_instance <- n_timepoints
#   }
#
#   # Split inputs into batches
#   total_rows <- if (is.list(inputs)) inputs[[1]]$shape[1] else inputs$shape[1]
#   idx <- lapply(seq(1, total_rows, by = batch_size), function(i) {
#     c(i, min(i + batch_size - 1, total_rows))
#   })
#   lapply(idx, function(i) {
#     if (is.list(inputs)) {
#       list(
#         batch = lapply(inputs, function(x) x[i[1]:i[2],,drop = FALSE]),
#         num = as.integer((i[2] - i[1] + 1) / rows_per_instance),
#         idx = i
#       )
#     } else {
#       list(
#         batch = inputs[i[1]:i[2],,drop = FALSE],
#         num = as.integer((i[2] - i[1] + 1) / rows_per_instance),
#         idx = i
#       )
#     }
#   })
# }


# Add noise --------------------------------------------------------------------
add_noise <- function(inputs, orig_data, noise_level) {
  # Make sure both are lists
  if (!is.list(inputs)) inputs <- list(inputs)
  if (!is.list(orig_data)) orig_data <- list(orig_data)

  lapply(seq_along(inputs), function(i) {
    orig <- orig_data[[i]]
    names(orig) <- NULL

    # Calculate standard deviation
    std <- torch::torch_tensor(apply(orig, seq_along(dim(orig))[-1], sd))$unsqueeze(1)

    # Generate noise
    noise <- torch::torch_tensor(array(rnorm(prod(dim(inputs[[i]]))), dim = dim(inputs[[i]])))
    noise <- noise * noise_level *std

    # Add noise
    inputs[[i]] + noise
  })
}

# Set feature names ------------------------------------------------------------
set_names <- function(arr, feat_names, timepoints, include_time = FALSE,
                      event_names = NULL) {
  arr <- as.array(arr)
  if (!is.list(feat_names)) feat_names <- list(feat_names)

  # Remove event dimension if not required
  if (length(event_names) == 1) {
    event_names <- NULL
    dim(arr) <- rev(rev(dim(arr))[-2])
  }

  # Add time if required
  feat_names <- lapply(feat_names, function(a) {
    a <- if (include_time) c(a, "time") else a
    a
  })
  time_labels <- paste0(timepoints)

  # Set dimnames
  if (!is.null(event_names)) {
    dimnames(arr) <- list(NULL, unlist(feat_names, recursive = FALSE),
                          event_names, time_labels)
  } else {
    dimnames(arr) <- list(NULL, unlist(feat_names, recursive = FALSE), time_labels)
  }

  arr
}


get_dimnames <- function(x) {
  if (is.null(dimnames(x))) {
    dim_prefix <- c("X", "Y", "Z", "A", "B")[seq_len(length(dim(x)) - 1)]
    lapply(seq_along(dim(x))[-1], function(i) paste0(dim_prefix[i - 1], seq_len(dim(x)[i])))
  } else {
    dimnames(x)[-1]
  }
}
