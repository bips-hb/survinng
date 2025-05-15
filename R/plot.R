
#' Plot Methods for Survival Attribution Results
#'
#' Visualize survival predictions, feature attributions, and contribution
#' percentages and force plots for survival results. The latter two are
#' specifically for GradSHAP(t) and IntGrad(t) methods.
#'
#' These functions provide a convenient way to visualize the results of survival
#' attribution methods:
#'
#' - `plot()` is a generic wrapper that dispatches to the appropriate plot type
#' based on the `type` argument.
#' - `plot_pred()` visualizes survival predictions across time for the selected
#' instances.
#' - `plot_attr()` displays time-resolved attributions over time per instance.
#'
#' The following methods are only available for `GradSHAP(t)` and `IntGrad(t)`:
#'
#' - `plot_contr()` visualizes the relative contribution of features over time,
#' optionally aggregated across instances for global insights.
#' - `plot_force()` generates force plots showing the features' effect to the
#' prediction over time.
#'
#' @param x An object of class `surv_result` containing survival attribution
#' results.
#' @param ... (unsed arguments)
#' @param type Type of plot to generate when using the generic `plot()` method. Options:
#'   - `"pred"`: plot survival predictions over time
#'   - `"attr"`: plot feature attributions over time (default)
#'   - `"contr"`: plot feature contributions percentages over time
#'   - `"force"`: plot force plots for each instance
#' @param normalize Normalization method for `plot_attr()`. Options:
#'   - `"none"` (default): no normalization
#'   - `"abs"`: normalize by the sum of absolute values
#'   - `"rel"`: normalize by the sum of values
#'  **Note:** Only recommended for visualization of `GradSHAP(t)` or `IntGrad(t)`
#'  results.
#' @param add_comp Optional vector of comparison curves to add to the
#' attribution plot (`plot_attr()` only). Options include:
#'   - `"pred"`: predicted survival curve
#'   - `"pred_ref"`: reference survival curve
#'   - `"pred_diff"`: difference between prediction and reference
#'   Default is `NULL`.
#' @param scale Scaling factor for plotting contribution percentages in
#' `plot_contr()`. Default is `0.85`.
#' @param aggregate Logical; if `TRUE`, contributions are aggregated across
#'  all instances in `plot_contr()`. If `FALSE` (default), one panel per instance
#'  is shown.
#'
#' @return A `ggplot2` object.
#' @family Plot Methods
#' @export
#' @rdname plot.surv_result
plot.surv_result <- function(x, ..., type = "attr") {


  if (type ==  "pred") {
    p <- plot_pred(x, ...)
  } else if (type == "attr") {
    p <- plot_attr(x, ...)
  } else if (type == "contr") {
    p <- plot_contr(x, ...)
  } else if (type == "force") {
    p <- plot_force(x, ...)
  } else {
    stop("Invalid type. Choose 'pred' for survival prediction, 'attr' for
         attributions, or 'contr' for contributions plots.")
  }

  p
}



#' @family Plot Methods
#' @export
#' @rdname plot.surv_result
plot_force <- function(x, ...) {
  # TODO

  NULL
}


#' @family Plot Methods
#' @export
#' @rdname plot.surv_result
plot_pred <- function(x) {
  assertClass(x, "surv_result")

  dat <- as.data.frame(x)
  dat$id <- as.factor(dat$id)

  if ("feature2" %in% colnames(dat)) {
    stop("This plot method only supports one dimensional feature attributions.")
  }

  p <- ggplot(dat, aes(x = .data$time))  +
    geom_line(aes(
      y = .data$pred,
      group = .data$id,
      color = .data$id
    )) +
    theme_minimal(base_size = 16) +
    theme(legend.position = "bottom") +
    labs(
      x = "Time",
      y = "Survival Prediction",
      color = "Instance ID",
      linetype = "Instance ID"
    )

  p
}


#' @family Plot Methods
#' @export
#' @rdname plot.surv_result
plot_attr <- function(x, normalize = "none", add_comp = NULL) {
  assertChoice(normalize, c("none", "abs", "rel"))
  assertSubset(add_comp, c("pred_ref", "pred", "pred_diff"), empty.ok = TRUE)

  dat <- as.data.frame(x)
  dat$id <- as.factor(dat$id)


  # Check if it is for one-dimensional feature attributions
  if ("feature2" %in% colnames(dat)) {
    stop("This plot method only supports one dimensional feature attributions.")
  }

  # Normalize values if requested
  if (normalize == "abs") {
    norm_fun <- function(a) abs(a) / sum(abs(a))
  } else if (normalize == "rel") {
    norm_fun <- function(a) a / sum(a)
  } else {
    norm_fun <- function(a) a
  }
  dat$value <- ave(dat$value, dat$id, dat$time, FUN = norm_fun)


  # Add comparison curves (reference, prediction, sum, difference)
  if (!is.null(add_comp)) {
    # Check if reference value is available
    if (!(x$method %in% c("Surv_GradSHAP", "Surv_Intgrad"))) {
      stop("Comparison curves are only available for GradSHAP(t) and Intgrad(t) methods.")
    }

    if (normalize != "none") {
      warning("It is not recommended to normalize values when using comparison curves.")
    }

    dat$pred_ref <- dat$pred - dat$pred_diff
    values <- stack(dat[, add_comp])
    dat_comp <- cbind(dat[rep(1:nrow(dat), times = length(add_comp)), c("id", "time", "feature")],
                      Comparison = values$ind,
                      value = values$values)
    # Create comparison curves
    comp <- geom_line(
      data = dat_comp,
      aes(
        x = .data$time,
        y = .data$value,
        group = .data$Comparison,
        linetype = .data$Comparison
      ),
      color = "grey",
      linewidth = 1
    )
  } else {
    comp <- NULL
  }

  p <- ggplot(dat, aes(x = .data$time)) +
    geom_line(aes(
      y = .data$value,
      group = .data$feature,
      color = .data$feature
    ), na.rm = TRUE) +
    geom_point(aes(
      y = .data$value,
      group = .data$feature,
      color = .data$feature
    ),
    na.rm = TRUE,
    size = 0.75) +
    comp +
    geom_hline(yintercept = 0, color = "black") +
    facet_wrap(vars(.data$id),
               scales = "free_x",
               labeller = as_labeller(function(a)
                 paste0("Instance ID: ", a))) +
    theme_minimal(base_size = 16) +
    theme(legend.position = "bottom") +
    labs(
      x = "Time",
      y = paste0("Attribution S(t|x): ", x$model_class),
      color = "Feature",
      linetype = NULL
    ) +
    scale_color_viridis_d()

  p
}

#' @family Plot Methods
#' @export
#' @rdname plot.surv_result
plot_contr <- function(x, scale = 0.85, aggregate = FALSE) {
  dat <- as.data.frame(x)

  if (!(x$method %in% c("Surv_GradSHAP", "Surv_Intgrad"))) {
    stop("Contribution plots are only available for GradSHAP(t) and Intgrad(t) methods.")
  }

  # Aggregate values if requested
  if (aggregate) {
    dat <- dat %>%
      group_by(.data$time, .data$feature) %>%
      summarize(value = mean(abs(.data$value)), pred = mean(.data$pred),
                pred_diff = mean(.data$pred_diff), method = unique(.data$method),
                .groups = "drop")
    dat$id <- "Aggregated"
  }

  dat$id <- as.factor(dat$id)
  dat$pred_ref <- dat$pred - dat$pred_diff
  integer_times <- seq(from = ceiling(min(dat$time)), to = floor(max(dat$time)))

  # Process data to compute ratios, cumulative ratios, and ymin
  dat <- dat %>%
    group_by(.data$id, .data$time) %>%
    mutate(sum = sum(abs(.data$value)),
           ratio = abs(.data$value) / abs(.data$sum) * 100) %>%
    arrange(.data$id, .data$time, desc(.data$feature)) %>%
    group_by(.data$id, .data$time) %>%
    mutate(cumulative_ratio = cumsum(.data$ratio)) %>%
    arrange(.data$id, .data$time, .data$feature) %>%
    group_by(.data$id, .data$time) %>%
    mutate(ymin = lead(.data$cumulative_ratio, default = 0))

  # Obtain average ratio over features and positions for the barchart
  avg_contribution <- dat %>%
    group_by(.data$id, .data$feature) %>%
    summarize(mean_ratio = round(mean(.data$ratio),2), .groups = "drop") %>%
    group_by(.data$id) %>%
    mutate(pos = rev(cumsum(rev(.data$mean_ratio))) * scale)

  # Set width of the bars to 10% of the time range
  bar_width <- (max(dat$time) - min(dat$time)) * 0.1

  # Generate the plot
  p <- ggplot(dat, aes(x = .data$time)) +
    # Line plot for cumulative_ratio
    geom_line(aes(y = .data$cumulative_ratio, group = .data$feature),
              color = "black") +
    # Ribbon for cumulative_ratio
    geom_ribbon(
      aes(
        ymin = .data$ymin,
        ymax = .data$cumulative_ratio,
        group = .data$feature,
        fill = .data$feature
      ),
      alpha = 0.4
    ) +
    # Bar plot for mean_ratio
    geom_bar(
      data = avg_contribution, # Unique rows for bar plot
      aes(
        x = max(dat$time) + bar_width, # Offset x to place bars next to lines
        y = .data$mean_ratio,
        fill = .data$feature
      ),
      stat = "identity",
      alpha = 0.6,
      width = 0.8 * bar_width
    ) +
    # Add percentage labels
    geom_text(
      data = avg_contribution,
      aes(
        x = max(dat$time) + bar_width,
        y = .data$pos,
        label = paste0(round(.data$mean_ratio, 1), "%")
      ),
      color = "black",
      size = 3,
      check_overlap = TRUE
    ) +
    # Facet for each instance ID
    facet_wrap(vars(.data$id),
               scales = "free_x",
               labeller = as_labeller(function(a) {
                 if (aggregate) {
                   "Global"
                 } else {
                   paste0("Instance ID: ", a)
                 }
               })) +
    # Minimal theme
    theme_minimal(base_size = 16) +
    theme(legend.position = "bottom") +
    # Suppress x-axis tick label for bar offset
    scale_x_continuous(
      breaks = integer_times,
      labels = integer_times,
      guide = guide_axis(check.overlap = TRUE)
    ) +
    scale_y_continuous(expand = c(0,0)) +
    # Labels
    labs(
      x = "Time",
      y = paste0("% Contribution: ", x$model_class),
      color = "Feature",
      fill = "Feature"
    ) +
    scale_fill_viridis_d(alpha = 0.4)

  p
}


# Helper function to process nested result structures
process_result <- function(res, prefix = NULL) {
  dims <- dim(res)
  dim_names <- dimnames(res)

  # Define meaningful prefixes based on dimensions
  if (length(dims) == 4) {
    labels <- c("Instance", "Feature", "Event", "Time")
  } else if (length(dims) == 5) {
    labels <- c("Instance", "Channel", "Height", "Width", "Time")
  } else if (length(dims) == 6) {
    labels <- c("Instance", "Channel", "Height", "Width", "Event", "Time")
  } else {
    labels <- paste0("Dim", seq_along(dims))
  }

  # Apply prefixes to dimension names
  if (is.null(dim_names)) {
    dim_names <- lapply(seq_along(dims), function(i) paste0(labels[i], seq_len(dims[i])))
  }
  names(dim_names) <- labels

  # Create data frame for this result
  df <- expand.grid(dim_names)
  df$value <- c(res)

  if (!is.null(prefix)) {
    df$input_layer <- prefix
  }
  return(df)
}


#' Convert survival attribution results to a data.frame
#'
#' This function converts the survival attribution results into a data frame
#' format. It can handle both stacked and non-stacked formats.
#'
#' @param x An object of class `surv_result` containing the survival
#' attribution results.
#' @param stacked Logical indicating whether to convert to a stacked data frame,
#' i.e., the attributions are stacked on top of each other. Default is `FALSE`.
#' @param ... Unused arguments.
#'
#' @family Conversion Methods
#' @rdname as.data.frame
#' @export
as.data.frame.surv_result <- function(x, ..., stacked = FALSE) {

  # Get id, time and feature names
  id <- x$method_args$instance
  time <- x$time

  # Iterate over nested results
  res <- lapply(seq_along(x$res), function(i) {
    arr_i <- x$res[[i]]
    feat_names <- dimnames(arr_i)[-c(1, length(dim(arr_i)))]
    feat_dim_labels <- switch(as.character(length(feat_names) - x$competing_risks),
                              "1" = c("feature"),
                              "2" = c("channel", "length"),
                              "3" = c("channel", "height", "width"),
                              paste0("feature", c("", seq_len(length(feat_names) - x$competing_risks))))

    # If the results contain competing risks, add an event dimension
    if (x$competing_risks) {
      feat_dim_labels <- c(feat_dim_labels, "event")
    }
    names(feat_names) <- feat_dim_labels

    # Create data frame
    df <- expand.grid(append(list(id = id, time = time), feat_names, after = 1))
    df$value <- c(arr_i)

    # Add predictions and quantiles
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
        arrange(.data$time, .data$feature) %>%
        group_by(.data$feature) %>%
        summarize(means = mean(abs(.data$value), na.rm = TRUE))

      # Order features by reverse order of mean of absolute attribution values
      custom_order <- as.character(abs_contributions[order(-abs_contributions$means), ]$feature)
      df$feature <- factor(df$feature, levels = rev(custom_order))

      # Compute cumulative sum of ordered attribution values
      df <- df %>%
        group_by(.data$time) %>%
        arrange(.data$time, .data$feature) %>%
        mutate(cum_value = cumsum(.data$value))

      # Add reference value to cumulative sum
      df$cum_ref <- df$cum_value + df$pred_ref

      # Compute minimum value for ribbon plots
      df <- df %>%
        group_by(.data$time) %>%
        arrange(.data$time, .data$feature) %>%
        mutate(min_value = lag(.data$cum_ref),
               min_value = coalesce(.data$min_value, .data$pred_ref))

      # Order features by mean of absolute attribution values
      df$feature <- factor(df$feature, levels = custom_order)
      df <- df[order(df$time, df$feature), ]
    }

    df
  })

  if (length(res) > 1) {
    names(res) <- paste0("Input_Layer_", seq_along(res))
  } else {
    res <- res[[1]]
  }

  res
}

#'
#' @family Conversion Methods
#' @exportS3Method data.table::as.data.table
#' @rdname as.data.frame
as.data.table.surv_result <- function(x, ..., stacked = FALSE) {

  # Get id, time and feature names
  id <- x$method_args$instance
  time <- x$time

  # Iterate over nested results
  res <- lapply(seq_along(x$res), function(i) {
    arr_i <- x$res[[i]]
    feat_names <- dimnames(arr_i)[-c(1, length(dim(arr_i)))]
    feat_dim_labels <- switch(as.character(length(feat_names) - x$competing_risks),
                              "1" = c("feature"),
                              "2" = c("channel", "length"),
                              "3" = c("channel", "height", "width"),
                              paste0("feature", c("", seq_len(length(feat_names) - x$competing_risks))))

    # If the results contain competing risks, add an event dimension
    if (x$competing_risks) {
      feat_dim_labels <- c(feat_dim_labels, "event")
    }
    names(feat_names) <- feat_dim_labels

    # Create data frame
    dt <- do.call(data.table::CJ, c(list(id = id, time = time), feat_names, after = 1))
    dt$value <- c(arr_i)

    # Add predictions and quantiles
    if (x$competing_risks) {
      dt_pred <- data.table::CJ(id = id, event = feat_names$event, time = time)
      by <- c("id", "event", "time")
    } else {
      dt_pred <- data.table::CJ(id = id, time = time)
      by <- c("id", "time")
    }
    dt_pred$pred <- c(x$pred)
    if (!is.null(x$pred_diff)) dt_pred$pred_diff <- c(x$pred_diff)
    if (!is.null(x$pred_diff_q1)) dt_pred$pred_diff_q1 <- c(x$pred_diff_q1)
    if (!is.null(x$pred_diff_q3)) dt_pred$pred_diff_q3 <- c(x$pred_diff_q3)

    # Merge both data frames
    dt <- merge(dt, dt_pred, by = by)

    # Add method label
    dt$method <- paste0(switch(x$method,
                               "Surv_Gradient" = if (x$method_args$times_input) "Grad x Input" else "Gradient",
                               "Surv_SmoothGrad" = if (x$method_args$times_input) "SmoothGrad x Input" else "SmoothGrad",
                               "Surv_IntGrad" = "IntGrad",
                               gsub("Surv_", "", x$method)
    ), " (", x$method_args$target, ")")

    if (stacked == TRUE) {
      # Add reference value
      dt$pred_ref <- dt$pred - dt$pred_diff

      # Compute mean of absolute attribution values
      abs_contributions <- dt %>%
        arrange(.data$time, .data$feature) %>%
        group_by(.data$feature) %>%
        summarize(means = mean(abs(.data$value), na.rm = TRUE))

      # Order features by reverse order of mean of absolute attribution values
      custom_order <- as.character(abs_contributions[order(-abs_contributions$means), ]$feature)
      dt$feature <- factor(dt$feature, levels = rev(custom_order))

      # Compute cumulative sum of ordered attribution values
      dt <- dt %>%
        group_by(.data$time) %>%
        arrange(.data$time, .data$feature) %>%
        mutate(cum_value = cumsum(.data$value))

      # Add reference value to cumulative sum
      dt$cum_ref <- dt$cum_value + dt$pred_ref

      # Compute minimum value for ribbon plots
      dt <- dt %>%
        group_by(.data$time) %>%
        arrange(.data$time, .data$feature) %>%
        mutate(min_value = lag(.data$cum_ref),
               min_value = coalesce(.data$min_value, .data$pred_ref))

      # Order features by mean of absolute attribution values
      dt$feature <- factor(dt$feature, levels = custom_order)
      dt <- dt[order(dt$time, dt$feature), ]
    }

    dt
  })

  if (length(res) > 1) {
    names(res) <- paste0("Input_Layer_", seq_along(res))
  } else {
    res <- res[[1]]
  }

  res
}
