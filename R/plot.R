

#' Plot method for survival attribution results
#'
#' @family Plot Methods
#' @export
plot.surv_result <- function(x, ...) {
  dat <- as.data.frame(x)

  if ("pred_diff" %in% colnames(dat)) {
    line <- geom_line(aes(y = .data$pred_diff), color = "darkgray", linetype = "dashed")
  } else {
    line <- geom_line(aes(y = .data$pred), color = "darkgray", linetype = "dashed")
  }

  ggplot(dat, aes(x = .data$time)) +
    geom_line(aes(y = .data$value, group = .data$feature, color = .data$feature)) +
    line +
    geom_hline(yintercept = 0, color = "black") +
    geom_point(aes(y = .data$value, group = .data$feature, color = .data$feature)) +
    facet_wrap(vars(.data$id), scales = "free_x", labeller = as_labeller(function(a) paste0("Instance ID: ", a))) +
    theme_minimal() +
    theme(legend.position = "bottom") +
    labs(x = "Time", y = paste0("Attribution (", x$method, ")"), color = "Feature", linetype = NULL)
}


#' Convert survival attribution results to a data frame
#'
#' @family Conversion Methods
#' @rdname as.data.frame
#' @export
as.data.frame.surv_result <- function(x, ...) {

  # Get id, time and feature names
  id <- x$method_args$instance
  time <- x$time
  feat_names <- dimnames(x$res)[-c(1, length(dim(x$res)))]
  feat_dim_labels <- paste0("feature", c("", seq_along(feat_names)[-1]))

  # If the results contain competing risks, add an event dimension
  if (x$competing_risks) {
    feat_dim_labels[length(feat_dim_labels)] <- "event"
  }
  names(feat_names) <- feat_dim_labels

  # Create data frame
  df <- expand.grid(append(list(id = id, time = time), feat_names, after = 1))
  df$value <- c(x$res)

  # Add predictions
  if (x$competing_risks) {
    df_pred <- expand.grid(id = id, event = feat_names$event, time = time)
    by <- c("id", "event", "time")
  } else {
    df_pred <- expand.grid(id = id, time = time)
    by <- c("id", "time")
  }
  df_pred$pred <- c(x$pred)
  if (!is.null(x$pred_diff)) df_pred$pred_diff <- c(x$pred_diff)
  if (!is.null(x$pred_diff_q1)) df_pred$pred_diff_q1 <- c(x$pred_diff_q1)
  if (!is.null(x$pred_diff_q3)) df_pred$pred_diff_q3 <- c(x$pred_diff_q3)

  # Merge both data frames
  df <- merge(df, df_pred, by = by)

  # Add method label
  df$method <- paste0(switch(x$method,
    "Surv_Gradient" = if (x$method_args$times_input) "Grad x Input" else "Gradient",
    "Surv_SmoothGrad" = if (x$method_args$times_input) "SmoothGrad x Input" else "SmoothGrad",
    "Surv_IntGrad" = "IntGrad",
    gsub("Surv_", "", x$method)
    ), " (", x$method_args$target, ")")

  df
}

#'
#' @family Conversion Methods
#' @exportS3Method data.table::as.data.table
#' @rdname as.data.frame
as.data.table.surv_result <- function(x, ...) {
  requireNamespace("data.table", quietly = TRUE)
  data.table::as.data.table(as.data.frame(x))
}
