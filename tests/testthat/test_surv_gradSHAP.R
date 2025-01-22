
################################################################################
#                         surv_gradSHAP: CoxTime                               #
################################################################################

test_that("Method 'surv_gradSHAP' with CoxTime (1D model)", {
  # Preparation ----------------------------------------------------------------
  model_1d <- seq_model_1D(6)
  data <- torch_randn(10, 5)
  base_hazard <- get_base_hazard(20)
  exp_1d <- explain(model_1d, data = data, model_type = "coxtime",
                    baseline_hazard = base_hazard)

  # Test arguments -------------------------------------------------------------
  expect_error(surv_gradSHAP("no explainer"))
  res <- surv_gradSHAP(exp_1d)
  expect_error(surv_gradSHAP(exp_1d, n = "not a number"))
  expect_error(surv_gradSHAP(exp_1d, num_samples = "not a number"))
  expect_error(surv_gradSHAP(exp_1d, data_ref = "not an array"))
  expect_error(surv_gradSHAP(exp_1d, data_ref = torch_randn(1, 2)))
  expect_error(surv_gradSHAP(exp_1d, target = "wrong target"))
  expect_error(surv_gradSHAP(exp_1d, instance = "wrong instance"))
  expect_error(surv_gradSHAP(exp_1d, use_base_hazard = list("not boolean")))
  expect_error(surv_gradSHAP(exp_1d, batch_size = "not integer"))
  expect_error(surv_gradSHAP(exp_1d, dtype = "wrong dtype"))
  res <- surv_gradSHAP(exp_1d, data_ref = torch_randn(10, 5))
  res <- surv_gradSHAP(exp_1d, data_ref = list(torch_randn(10, 5)))
  res <- surv_gradSHAP(exp_1d, data_ref = as.array(torch_randn(10, 5)))

  # Test sequential 1D model ---------------------------------------------------
  res <- surv_gradSHAP(exp_1d, instance = c(1, 3))
  expect_s3_class(res, "surv_result")
  checkmate::expect_names(names(res), subset.of = c("res", "pred", "pred_diff", "time",
                                         "method", "method_args", "pred_diff_q1",
                                         "pred_diff_q3", "competing_risks", "model_class"))
  expect_equal(res$method, "Surv_GradSHAP")
  expect_equal(res$competing_risks, FALSE)
  expect_equal(res$method_args$target, "survival")
  expect_equal(res$method_args$dtype, "float")
  expect_equal(res$method_args$n, 50)
  expect_equal(dim(res$res[[1]]), c(2, 5, 20))
  expect_equal(dim(res$pred), c(2, 20))
  expect_equal(length(res$time), 20)
  checkmate::expect_array(res$res[[1]])
  checkmate::expect_array(res$pred)
  expect_vector(res$time)

  # Test sequential 1D model with other targets --------------------------------
  res <- surv_gradSHAP(exp_1d, instance = c(1, 3), target = "hazard")
  expect_equal(res$method_args$target, "hazard")
  expect_equal(res$method_args$dtype, "float")
  expect_equal(dim(res$res[[1]]), c(2, 5, 20))
  expect_equal(dim(res$pred), c(2, 20))
  expect_equal(length(res$time), 20)
  res <- surv_gradSHAP(exp_1d, instance = c(1, 3), target = "cum_hazard")
  expect_equal(res$method_args$target, "cum_hazard")
  expect_equal(res$method_args$dtype, "float")
  expect_equal(dim(res$res[[1]]), c(2, 5, 20))
  expect_equal(dim(res$pred), c(2, 20))
  expect_equal(length(res$time), 20)
  res <- surv_gradSHAP(exp_1d, instance = c(1, 3), target = "survival")
  expect_equal(res$method_args$target, "survival")
  expect_equal(res$method_args$dtype, "float")
  expect_equal(dim(res$res[[1]]), c(2, 5, 20))
  expect_equal(dim(res$pred), c(2, 20))
  expect_equal(length(res$time), 20)

  # Test sequential 1D model with other dtype ----------------------------------
  res <- surv_gradSHAP(exp_1d, instance = c(1, 3), dtype = "double")
  expect_equal(res$method_args$dtype, "double")
  expect_equal(dim(res$res[[1]]), c(2, 5, 20))
  expect_equal(dim(res$pred), c(2, 20))
  expect_equal(length(res$time), 20)
})


test_that("Method 'surv_gradSHAP' with CoxTime (Multi-modal model)", {
  # Preparation ----------------------------------------------------------------
  model_mm <- multi_modal_model(num_tabular_inputs = 4, num_outputs = 1, add_time = TRUE)
  data <- list(torch_randn(4, 3, 10, 10), torch_randn(4, 4))
  base_hazard <- get_base_hazard(12)

  # Define preprocess function to handle the time axis
  preprocess_fun <- function(x) {
    # x is a list with two tensors: image data and tabular data
    batch_size <- x[[1]]$size(1)

    # Input (batch_size, in_features) -> (batch_size * t, in_features) (replicate each row t times)
    res <- lapply(x, function(a) a$repeat_interleave(12L, dim = 1))

    # Add time to tabular input
    time <- torch::torch_vstack(replicate(batch_size, torch_tensor(base_hazard$time)$unsqueeze(-1)))
    list(res[[1]], torch::torch_cat(list(res[[2]], time), dim = -1))
  }

  exp_mm <- explain(model_mm, data = data, model_type = "coxtime",
                    baseline_hazard = base_hazard, preprocess_fun = preprocess_fun)

  # Check ----------------------------------------------------------------------
  expect_warning(expect_warning(expect_warning(
    res <- surv_gradSHAP(exp_mm, n = 5, batch_size = 500)
  )))
  expect_warning(expect_warning(expect_warning(
    res <- surv_gradSHAP(exp_mm, instance = c(1, 3), n = 5, batch_size = 500)
  )))
  expect_s3_class(res, "surv_result")

  # Test multi-modal model with custom ref dataset------------------------------
  data_ref <- list(torch_randn(4, 3, 10, 10), torch_randn(4, 4))
  expect_warning(expect_warning(expect_warning(
    res <- surv_gradSHAP(exp_mm, instance = c(1, 3), data_ref = data_ref,
                         n = 5, batch_size = 500)
  )))
  expect_s3_class(res, "surv_result")
})


################################################################################
#                         surv_gradSHAP: DeepSurv                              #
################################################################################

test_that("Method 'surv_gradSHAP' with DeepSurv (1D model)", {
  # Preparation ----------------------------------------------------------------
  model_1d <- seq_model_1D(5)
  data <- torch_randn(10, 5)
  base_hazard <- get_base_hazard(20)
  exp_1d <- explain(model_1d, data = data, model_type = "deepsurv",
                    baseline_hazard = base_hazard)

  # Test arguments -------------------------------------------------------------
  expect_error(surv_gradSHAP("no explainer"))
  res <- surv_gradSHAP(exp_1d)
  expect_error(surv_gradSHAP(exp_1d, n = "not a number"))
  expect_error(surv_gradSHAP(exp_1d, num_samples = "not a number"))
  expect_error(surv_gradSHAP(exp_1d, data_ref = "not an array"))
  expect_error(surv_gradSHAP(exp_1d, data_ref = torch_randn(1, 2)))
  expect_error(surv_gradSHAP(exp_1d, target = "wrong target"))
  expect_error(surv_gradSHAP(exp_1d, instance = "wrong instance"))
  expect_error(surv_gradSHAP(exp_1d, batch_size = "not integer"))
  expect_error(surv_gradSHAP(exp_1d, dtype = "wrong dtype"))
  res <- surv_gradSHAP(exp_1d, data_ref = torch_randn(10, 5))
  res <- surv_gradSHAP(exp_1d, data_ref = list(torch_randn(10, 5)))
  res <- surv_gradSHAP(exp_1d, data_ref = as.array(torch_randn(10, 5)))

  # Test sequential 1D model ---------------------------------------------------
  res <- surv_gradSHAP(exp_1d, instance = c(1, 3))
  expect_s3_class(res, "surv_result")
  checkmate::expect_names(names(res), subset.of = c("res", "pred", "pred_diff", "time",
                                                    "method", "method_args", "pred_diff_q1",
                                                    "pred_diff_q3", "competing_risks", "model_class"))
  expect_equal(res$method, "Surv_GradSHAP")
  expect_equal(res$competing_risks, FALSE)
  expect_equal(res$method_args$target, "survival")
  expect_equal(res$method_args$dtype, "float")
  expect_equal(res$method_args$n, 50)
  expect_equal(dim(res$res[[1]]), c(2, 5, 20))
  expect_equal(dim(res$pred), c(2, 20))
  expect_equal(length(res$time), 20)
  checkmate::expect_array(res$res[[1]])
  checkmate::expect_array(res$pred)
  expect_vector(res$time)

  # Test sequential 1D model with other targets --------------------------------
  res <- surv_gradSHAP(exp_1d, instance = c(1, 3), target = "hazard")
  expect_equal(res$method_args$target, "hazard")
  expect_equal(res$method_args$dtype, "float")
  expect_equal(dim(res$res[[1]]), c(2, 5, 20))
  expect_equal(dim(res$pred), c(2, 20))
  expect_equal(length(res$time), 20)
  res <- surv_gradSHAP(exp_1d, instance = c(1, 3), target = "cum_hazard")
  expect_equal(res$method_args$target, "cum_hazard")
  expect_equal(res$method_args$dtype, "float")
  expect_equal(dim(res$res[[1]]), c(2, 5, 20))
  expect_equal(dim(res$pred), c(2, 20))
  expect_equal(length(res$time), 20)
  res <- surv_gradSHAP(exp_1d, instance = c(1, 3), target = "survival")
  expect_equal(res$method_args$target, "survival")
  expect_equal(res$method_args$dtype, "float")
  expect_equal(dim(res$res[[1]]), c(2, 5, 20))
  expect_equal(dim(res$pred), c(2, 20))
  expect_equal(length(res$time), 20)

  # Test sequential 1D model with other dtype ----------------------------------
  res <- surv_gradSHAP(exp_1d, instance = c(1, 3), dtype = "double")
  expect_equal(res$method_args$dtype, "double")
  expect_equal(dim(res$res[[1]]), c(2, 5, 20))
  expect_equal(dim(res$pred), c(2, 20))
  expect_equal(length(res$time), 20)
})


test_that("Method 'surv_gradSHAP' with DeepSurv (Multi-modal model)", {
  # Preparation ----------------------------------------------------------------
  model_mm <- multi_modal_model(num_tabular_inputs = 5, num_outputs = 1)
  data <- list(torch_randn(4, 3, 10, 10), torch_randn(4, 5))
  base_hazard <- get_base_hazard(12)
  exp_mm <- explain(model_mm, data = data, model_type = "deepsurv",
                    baseline_hazard = base_hazard)

  # Check ----------------------------------------------------------------------
  res <- surv_gradSHAP(exp_mm, n = 5, batch_size = 200)
  res <- surv_gradSHAP(exp_mm, instance = c(1, 3), n = 5, batch_size = 200)
  expect_s3_class(res, "surv_result")

  # Test multi-modal model custom reference value ------------------------------
  data_ref <- list(torch_randn(5, 3, 10, 10), torch_randn(5, 5))
  res <- surv_gradSHAP(exp_mm, instance = c(1, 3), data_ref = data_ref,
                      n = 5, batch_size = 200)
  expect_s3_class(res, "surv_result")
})

################################################################################
#                          surv_gradSHAP: DeepHit                              #
################################################################################

test_that("Method 'surv_gradSHAP' with DeepHit (1D model)", {
  # Preparation ----------------------------------------------------------------
  time_bins <- seq(0, 1, length.out = 20)
  model_1d <- seq_model_1D(5, num_outputs = 20)
  data <- torch_randn(10, 5)
  exp_1d <- explain(model_1d, data = data, model_type = "deephit",
                    time_bins = time_bins)

  # Test arguments -------------------------------------------------------------
  expect_error(surv_gradSHAP("no explainer"))
  res <- surv_gradSHAP(exp_1d)
  expect_error(surv_gradSHAP(exp_1d, n = "not a number"))
  expect_error(surv_gradSHAP(exp_1d, num_samples = "not a number"))
  expect_error(surv_gradSHAP(exp_1d, data_ref = "not an array"))
  expect_error(surv_gradSHAP(exp_1d, data_ref = torch_randn(1, 2)))
  expect_error(surv_gradSHAP(exp_1d, target = "wrong target"))
  expect_error(surv_gradSHAP(exp_1d, instance = "wrong instance"))
  expect_error(surv_gradSHAP(exp_1d, batch_size = "not integer"))
  expect_error(surv_gradSHAP(exp_1d, dtype = "wrong dtype"))
  res <- surv_gradSHAP(exp_1d, data_ref = torch_randn(10, 5))
  res <- surv_gradSHAP(exp_1d, data_ref = list(torch_randn(10, 5)))
  res <- surv_gradSHAP(exp_1d, data_ref = as.array(torch_randn(10, 5)))

  # Test sequential 1D model ---------------------------------------------------
  res <- surv_gradSHAP(exp_1d, instance = c(1, 3))
  expect_s3_class(res, "surv_result")
  checkmate::expect_names(names(res), subset.of = c("res", "pred", "pred_diff", "time",
                                                    "method", "method_args", "pred_diff_q1",
                                                    "pred_diff_q3", "competing_risks", "model_class"))
  expect_equal(res$method, "Surv_GradSHAP")
  expect_equal(res$model_class, "DeepHit")
  expect_equal(res$competing_risks, FALSE)
  expect_equal(res$method_args$target, "survival")
  expect_equal(res$method_args$dtype, "float")
  expect_equal(res$method_args$n, 50)
  expect_equal(dim(res$res[[1]]), c(2, 5, 20))
  expect_equal(dim(res$pred), c(2, 20))
  expect_equal(length(res$time), 20)
  checkmate::expect_array(res$res[[1]])
  checkmate::expect_array(res$pred)
  expect_vector(res$time)

  # Test sequential 1D model with other targets --------------------------------
  res <- surv_gradSHAP(exp_1d, instance = c(1, 3), target = "pmf")
  expect_equal(res$method_args$target, "pmf")
  expect_equal(res$method_args$dtype, "float")
  expect_equal(dim(res$res[[1]]), c(2, 5, 20))
  expect_equal(dim(res$pred), c(2, 20))
  expect_equal(length(res$time), 20)
  res <- surv_gradSHAP(exp_1d, instance = c(1, 3), target = "cif")
  expect_equal(res$method_args$target, "cif")
  expect_equal(res$method_args$dtype, "float")
  expect_equal(dim(res$res[[1]]), c(2, 5, 20))
  expect_equal(dim(res$pred), c(2, 20))
  expect_equal(length(res$time), 20)
  res <- surv_gradSHAP(exp_1d, instance = c(1, 3), target = "survival")
  expect_equal(res$method_args$target, "survival")
  expect_equal(res$method_args$dtype, "float")
  expect_equal(dim(res$res[[1]]), c(2, 5, 20))
  expect_equal(dim(res$pred), c(2, 20))
  expect_equal(length(res$time), 20)

  # Test sequential 1D model with other dtype ----------------------------------
  res <- surv_gradSHAP(exp_1d, instance = c(1, 3), dtype = "double")
  expect_equal(res$method_args$dtype, "double")
  expect_equal(dim(res$res[[1]]), c(2, 5, 20))
  expect_equal(dim(res$pred), c(2, 20))
  expect_equal(length(res$time), 20)
})



test_that("Method 'surv_gradSHAP' with DeepHit (Multi-modal model)", {
  # Preparation ----------------------------------------------------------------
  model_mm <- multi_modal_model(num_tabular_inputs = 5, num_outputs = 10)
  data <- list(torch_randn(4, 3, 10, 10), torch_randn(4, 5))
  exp_mm <- explain(model_mm, data = data, model_type = "deephit",
                    time_bins = seq(0, 15, length.out = 10))

  # Check ----------------------------------------------------------------------
  res <- surv_gradSHAP(exp_mm)
  res <- surv_gradSHAP(exp_mm, instance = c(1, 3))
  expect_s3_class(res, "surv_result")

  # Test multi-modal model with custom reference value -------------------------
  data_ref <- list(torch_randn(5, 3, 10, 10), torch_randn(5, 5))
  res <- surv_gradSHAP(exp_mm, instance = c(1, 3), data_ref = data_ref)
  expect_s3_class(res, "surv_result")
})
