x_ref <- list(as.matrix(test[2, -c(1,2)]))
x <- intgrad_cox

dat <- as.data.frame(x)
dat_shap <- as.data.frame(shap_cox)

# Add sum of all attributions if requested
add_sum <- TRUE
if (add_sum) {
  dat_sum <- aggregate(as.formula(paste0("value ~ ", paste0(setdiff(colnames(dat), c("feature", "value")), collapse = " + "))),
                       data = dat, sum)
  dat_sum$feature <- "Sum"
  dat <- rbind(dat, dat_sum)
}

if ("pred_diff" %in% colnames(dat)) {
  line <- geom_line(aes(y = .data$pred_diff), color = "darkgray", linetype = "dashed", linewidth = 1.15)
} else {
  line <- geom_line(aes(y = .data$pred), color = "darkgray", linetype = "dashed", linewidth = 1.15)
}

if ("pred_diff_q1" %in% colnames(dat)) {
  ribbon <- geom_ribbon(aes(ymin = .data$pred_diff_q1, ymax = .data$pred_diff_q3),
                        fill = "gray", alpha = 0.4, linetype = "dashed")
} else {
  ribbon <- NULL
}

ggplot(dat, aes(x = .data$time)) + NULL +
  ribbon +
  geom_line(aes(y = .data$value, group = .data$feature, color = .data$feature)) +
  geom_point(aes(y = .data$value, group = .data$feature, color = .data$feature), size = 0.75) +
  line +
  geom_hline(yintercept = 0, color = "black") +
  facet_wrap(vars(.data$id), scales = "free_x", labeller = as_labeller(function(a) paste0("Instance ID: ", a))) +
  theme_minimal() +
  theme(legend.position = "bottom") +
  labs(x = "Time", y = paste0("Attribution: ", unique(dat$method)), color = "Feature", linetype = NULL)




#######################################################################################unique()
library(dplyr)
library(reshape2)

dat$pred_ref <- dat$pred - dat$pred_diff
dat$value_ref <- dat$value + dat$pred_ref
dat <- subset(dat, feature != "Sum")

abs_contributions <- dat %>%
  arrange(time, feature) %>%
  group_by(feature) %>%
  summarize(
    means = mean(abs(value_ref), na.rm = TRUE),  # Previous feature's value_ref (feature_n)
  )
custom_order <- as.character(abs_contributions[order(-abs_contributions$means), ]$feature)
rev_sorted_dat <- dat
rev_sorted_dat$feature <- factor(rev_sorted_dat$feature, levels = rev(custom_order))
cum_dat <- rev_sorted_dat %>%
  group_by(time) %>%                 # Group by time
  arrange(time, feature) %>%         # Ensure the data is ordered by feature
  mutate(cum_value = cumsum(value))  # Compute cumulative sum of value_ref per time
cum_dat$cum_ref <- cum_dat$cum_value + cum_dat$pred_ref

min_dat <- cum_dat %>%
  group_by(time) %>%
  arrange(time, feature) %>%
  mutate(min_value = lag(cum_ref),
         min_value = coalesce(min_value, pred_ref))

sorted_dat <- min_dat
sorted_dat$feature <- factor(min_dat$feature, levels = custom_order)
sorted_dat <- sorted_dat[order(sorted_dat$time, sorted_dat$feature), ]

pred_dat <- subset(sorted_dat, feature == "trt",
                    select = c(time, feature, pred))
pred_ref_dat <- subset(sorted_dat, feature == "trt",
                   select = c(time, feature, pred_ref))
pred_dat$feature <- "prediction"
pred_ref_dat$feature <- "reference"
names(pred_ref_dat)[names(pred_ref_dat) == 'pred_ref'] <- 'cum_ref'
names(pred_dat)[names(pred_dat) == 'pred'] <- 'cum_ref'
pred_ref_dat$min_value <- pred_ref_dat$cum_ref
pred_dat$min_value <- pred_dat$cum_ref
plot_dat <- subset(sorted_dat,
                    select = c(time, feature, min_value, cum_ref))
plot_dat <- rbind(plot_dat, pred_dat, pred_ref_dat)

ggplot(plot_dat, aes(x = .data$time)) +
  geom_ribbon(aes(ymin = .data$min_value, ymax = cum_ref, group = .data$feature, color = .data$feature, fill = .data$feature), alpha = 0.3) +
  geom_line(aes(y = .data$cum_ref, group = .data$feature, color = .data$feature), linewidth = 0.7) +
  theme_minimal() +
  scale_colour_viridis_d() +
  scale_fill_viridis_d() +
  theme(legend.position = "bottom")

ggplot(sorted_dat, aes(x = .data$time)) +
  geom_ribbon(aes(ymin = .data$min_value, ymax = .data$cum_ref, group = .data$feature, color = .data$feature, fill = .data$feature), alpha = 0.3) +
  geom_line(aes(y = .data$cum_ref, group = .data$feature, color = .data$feature), linewidth = 0.7) +
  geom_line(aes(y = .data$pred, color = "prediction"), linewidth = 0.7) +
  geom_line(aes(y = .data$pred_ref, color = "reference"), linewidth = 0.7) +
  theme_minimal() +
  scale_colour_viridis_d() +
  theme(legend.position = "bottom")

ggplot(sorted_dat, aes(x = .data$time)) +
  geom_ribbon(aes(ymin = .data$min_value, ymax = cum_ref, group = .data$feature, color = .data$feature, fill = .data$feature), alpha = 0.3) +
  geom_line(aes(y = .data$cum_ref, group = .data$feature, color = .data$feature), linewidth = 0.7) +
  geom_line(aes(y = .data$pred_ref), color = "blue", linetype = "dotted", linewidth = 1.5, alpha = 0.5) +
  geom_line(aes(y = .data$pred), color = "red", linetype = "dotted", linewidth = 1.5, alpha = 0.5) +
  theme_minimal() +
  scale_colour_viridis_d() +
  scale_fill_viridis_d() +
  ylab("") +
  theme(legend.position = "bottom")



###### simulated negative data
test_dat <- dat
test_dat <- test_dat[order(sorted_dat$feature), ]
n <- length(unique(dat$time))
simulated_data <- seq(from = -0.5, to = 0.5, length.out = (n+1))
test_dat$value[1:length(simulated_data)] <- simulated_data


test_sum_dat <- test_dat %>%
  group_by(time) %>%
  arrange(time, feature) %>%
  mutate(sum_value = sum(value))

test_sum_dat$pred <- test_sum_dat$pred_ref + test_sum_dat$sum_value
test_sum_dat$value_ref <- test_sum_dat$value + test_sum_dat$pred_ref

abs_contributions <- test_sum_dat %>%
  arrange(time, feature) %>%
  group_by(feature) %>%
  summarize(
    means = mean(abs(value_ref), na.rm = TRUE),
  )
custom_order <- as.character(abs_contributions[order(-abs_contributions$means), ]$feature)
custom_order
rev_sorted_dat <- test_sum_dat
rev_sorted_dat$feature <- factor(rev_sorted_dat$feature, levels = rev(custom_order))
cum_dat <- rev_sorted_dat %>%
  group_by(time) %>%
  arrange(time, feature) %>%
  mutate(cum_value = cumsum(value))
cum_dat$cum_ref <- cum_dat$cum_value + cum_dat$pred_ref

min_dat <- cum_dat %>%
  group_by(time) %>%
  arrange(time, feature) %>%
  mutate(min_value = lag(cum_ref),
         min_value = coalesce(min_value, pred_ref))

sorted_dat <- min_dat
sorted_dat$feature <- factor(min_dat$feature, levels = custom_order)
sorted_dat <- sorted_dat[order(sorted_dat$time, sorted_dat$feature), ]

ggplot(sorted_dat, aes(x = .data$time)) +
  geom_ribbon(aes(ymin = .data$min_value, ymax = cum_ref, group = .data$feature, color = .data$feature, fill = .data$feature), alpha = 0.3) +
  geom_line(aes(y = .data$cum_ref, group = .data$feature, color = .data$feature), linewidth = 0.7) +
 # geom_line(aes(y = .data$pred_ref), color = "blue", linetype = "dotted", linewidth = 1.5, alpha = 0.5) +
#  geom_line(aes(y = .data$pred), color = "red", linetype = "dotted", linewidth = 1.5, alpha = 0.5) +
  theme_minimal() +
  scale_colour_viridis_d() +
  scale_fill_viridis_d() +
  ylab("") +
  theme(legend.position = "bottom")
