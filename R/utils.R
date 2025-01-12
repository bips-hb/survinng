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

# Indexing along first dimension with unknown length ---------------------------
list_index <- function(x, idx) {
  if (is.list(x)) {
    res <- lapply(x, list_index, idx = idx)
  } else {
    if (inherits(x, "torch_tensor")) {
      res <- x$index_select(dim = 1, index = as.integer(idx))
    } else {
      empty_dims <- rep(list(TRUE), length(dim(x)) - 1)
      res <- do.call('[', c(list(x, idx), empty_dims, drop = FALSE))
    }
  }

  res
}


# Convert to torch tensor and repeat rows --------------------------------------
to_tensor <- function(x, instance, repeats = 1, dtype = torch::torch_float()) {
  # Convert to list if not already
  if (!is.list(x)) {
    x <- list(x)
  }

  lapply(x, function(i) {
    if (inherits(i, "torch_tensor")) {
      res <- list_index(i, instance)$to(dtype = dtype)
    } else {
      res <- torch::torch_tensor(as.array(list_index(i, instance)), dtype = dtype)
    }

    # Convert to tensor and repeat rows
    res <- res$repeat_interleave(repeats = as.integer(repeats), dim = 1)

    res
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

  # Add time if required (assuming tabular data)
  feat_names <- lapply(feat_names, function(a) {
    a <- if (include_time) c(a, "time") else a
    a
  })
  time_labels <- paste0(timepoints)

  # Set dimnames
  if (!is.null(event_names)) {
    dim_names <- list(NULL, unlist(feat_names, recursive = FALSE),
                      event_names, time_labels)
  } else {
    dim_names <- append(list(NULL, time_labels), feat_names, after = 1)
  }

  # Check dimnames
  dims_n <- unlist(lapply(dim_names, length))
  dims_arr <- dim(arr)
  for (i in seq(2, length(dim(arr)))) {
    if (dims_n[i] != dims_arr[i]) {
      # If just one dimension is missing, we assume it is the time axis
      if (dims_arr[i] - dims_n[i] == 1) {
        warning("Found one dimension less in the dimnames. Assuming it is the ",
                "time axis.", call. = FALSE)
        dim_names[[i]] <- c(dim_names[[i]], "time")
      } else {
        stop("Number of dimnames does not match the number of dimensions",
             " in the result array.", call. = FALSE)
      }
    }
  }

  # Set dimnames
  dimnames(arr) <- dim_names

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


list_stack <- function(lst, dim = 1) {
  res <- list()
  for (name in names(lst[[1]])) {
    if (is.null(lst[[1]][[name]])) {
      res[[name]] <- c()
    } else {
      if (inherits(lst[[1]][[name]], "torch_tensor")) {
        res[[name]] <- torch::torch_stack(lapply(lst, function(x) x[[name]]), dim = dim)
      } else {
        num_sub_lists <- length(lst[[1]][[name]])
        res[[name]] <- lapply(seq_len(num_sub_lists), function(i) {
          torch::torch_stack(lapply(lst, function(x) x[[name]][[i]]), dim = dim)
        })
      }
    }
  }

  res
}
