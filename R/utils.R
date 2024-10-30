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
to_tensor <- function(x, instance, repeats = 1, dtype = torch::torch_float()) {
  # Convert to list if not already
  if (!is.list(x)) {
    x <- list(x)
  }

  lapply(x, function(i) {
    if (inherits(i, "torch_tensor")) {
      res <- i[instance,, drop = FALSE]$to(dtype = dtype)
    } else {
      res <- torch::torch_tensor(as.matrix(i[instance,,drop = FALSE]), dtype = dtype)
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


list_stack <- function(lst, dim = 1) {
  res <- list()
  for (name in names(lst[[1]])) {
    if (is.null(lst[[1]][[name]])) {
      res[[name]] <- c()
    } else {
      res[[name]] <- torch::torch_stack(lapply(lst, function(x) x[[name]]), dim = dim)
    }
  }

  res
}
