

#' Plot method for survival attribution results
#'
#' @importFrom stats aggregate as.formula ave
#' @family Plot Methods
#' @export
plot.surv_result <- function(x, stacked = FALSE, normalize = FALSE, add_sum = FALSE, ...) {
  if (stacked == FALSE) {
    dat <- as.data.frame(x)

    if ("feature2" %in% colnames(dat)) {
      stop("This plot method only supports one dimensional feature attributions.")
    }

    # Normalize value
    if (normalize) {
      dat$value <- ave(
        dat$value,
        dat$id,
        dat$time,
        FUN = function(x)
          abs(x) / sum(abs(x))
      )
    }

    # Add sum of all attributions if requested
    if (add_sum) {
      dat_sum <- aggregate(as.formula(paste0(
        "value ~ ", paste0(setdiff(colnames(dat), c(
          "feature", "value"
        )), collapse = " + ")
      )), data = dat, sum)
      dat_sum$feature <- "Sum"
      dat <- rbind(dat, dat_sum)
    }

    if ("pred_diff" %in% colnames(dat)) {
      line <- geom_line(
        aes(y = .data$pred_diff),
        color = "darkgray",
        linetype = "dashed",
        linewidth = 1.15
      )
    } else {
      line <- geom_line(
        aes(y = .data$pred),
        color = "darkgray",
        linetype = "dashed",
        linewidth = 1.15
      )
    }

    if ("pred_diff_q1" %in% colnames(dat)) {
      ribbon <- geom_ribbon(
        aes(
          ymin = .data$pred_diff_q1,
          ymax = .data$pred_diff_q3
        ),
        fill = "gray",
        alpha = 0.4,
        linetype = "dashed"
      )
    } else {
      ribbon <- NULL
    }

    ggplot(dat, aes(x = .data$time)) + NULL +
      ribbon +
      geom_line(aes(
        y = .data$value,
        group = .data$feature,
        color = .data$feature
      )) +
      geom_point(aes(
        y = .data$value,
        group = .data$feature,
        color = .data$feature
      ),
      size = 0.75) +
      line +
      geom_hline(yintercept = 0, color = "black") +
      facet_wrap(vars(.data$id),
                 scales = "free_x",
                 labeller = as_labeller(function(a)
                   paste0("Instance ID: ", a))) +
      theme_minimal() +
      theme(legend.position = "bottom") +
      labs(
        x = "Time",
        y = paste0("Attribution: ", unique(dat$method)),
        color = "Feature",
        linetype = NULL
      )
  }

  if (stacked == TRUE){
    dat <- as.data.frame(x, stacked = TRUE)

    if ("feature2" %in% colnames(dat)) {
      stop("This plot method only supports one dimensional feature attributions.")
    }

    ggplot(df, aes(x = .data$time)) + NULL +
      geom_ribbon(aes(ymin = .data$min_value, ymax = cum_ref, group = .data$feature, color = .data$feature, fill = .data$feature), alpha = 0.3) +
      geom_line(aes(y = .data$cum_ref, group = .data$feature, color = .data$feature), linewidth = 0.7) +
      geom_line(aes(y = .data$pred_ref), color = "blue", linetype = "dotted", linewidth = 1.5, alpha = 0.5) +
      geom_line(aes(y = .data$pred), color = "red", linetype = "dotted", linewidth = 1.5, alpha = 0.5) +
      theme_minimal() +
      scale_colour_viridis_d() +
      scale_fill_viridis_d() +
      ylab("") +
      theme(legend.position = "bottom")
    }

}


#' Convert survival attribution results to a data frame
#'
#' @family Conversion Methods
#' @rdname as.data.frame
#' @export
as.data.frame.surv_result <- function(x, stacked = FALSE, ...) {

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

  if (stacked == TRUE) {
    # Add reference value
    df$pred_ref <- df$pred - df$pred_diff

    # Compute mean of absolute attribution values
    abs_contributions <- df %>%
      arrange(time, feature) %>%
      group_by(feature) %>%
      summarize(means = mean(abs(value), na.rm = TRUE), )

    # Order features by reverse order of mean of absolute attribution values
    custom_order <- as.character(abs_contributions[order(-abs_contributions$means), ]$feature)
    df$feature <- factor(df$feature, levels = rev(custom_order))

    # Compute cumulative sum of ordered attribution values
    df <- df %>%
      group_by(time) %>%
      arrange(time, feature) %>%
      mutate(cum_value = cumsum(value))

    # Add reference value to cumulative sum
    df$cum_ref <- df$cum_value + df$pred_ref

    # Compute minimum value for ribbon plots
    df <- df %>%
      group_by(time) %>%
      arrange(time, feature) %>%
      mutate(min_value = lag(cum_ref),
             min_value = coalesce(min_value, pred_ref))

    # Order features by mean of absolute attribution values
    df$feature <- factor(df$feature, levels = custom_order)
    df <- df[order(df$time, df$feature), ]
  }

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
