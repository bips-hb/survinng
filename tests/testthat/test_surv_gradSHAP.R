
test_that("Method 'surv_gradSHAP' with CoxTime", {
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
  res <- surv_gradSHAP(exp_1d, data_ref = torch_randn(10, 5))
  res <- surv_gradSHAP(exp_1d, data_ref = list(torch_randn(10, 5)))
  res <- surv_gradSHAP(exp_1d, data_ref = as.array(torch_randn(10, 5)))

  # Test sequential 1D model ---------------------------------------------------
  res <- surv_gradSHAP(exp_1d, instance = c(1, 3))
  expect_class(res, "surv_result")
  expect_names(names(res), subset.of = c("res", "pred", "pred_diff", "time",
                                         "method", "method_args", "pred_diff_q1",
                                         "pred_diff_q3", "competing_risks"))
  expect_equal(res$method, "Surv_GradSHAP")
  expect_equal(res$method_args$n, 50)
  expect_equal(dim(res$res), c(2, 5, 20))
  expect_equal(dim(res$pred), c(2, 20))
  expect_equal(length(res$time), 20)
  expect_array(res$res)
  expect_array(res$pred)
  expect_vector(res$time)

  # Test sequential 1D model with other targets --------------------------------
  res <- surv_gradSHAP(exp_1d, instance = c(1, 3), target = "hazard")
  expect_equal(dim(res$res), c(2, 5, 20))
  expect_equal(dim(res$pred), c(2, 20))
  expect_equal(length(res$time), 20)
  res <- surv_gradSHAP(exp_1d, instance = c(1, 3), target = "cum_hazard")
  expect_equal(dim(res$res), c(2, 5, 20))
  expect_equal(dim(res$pred), c(2, 20))
  expect_equal(length(res$time), 20)
  res <- surv_gradSHAP(exp_1d, instance = c(1, 3), target = "survival")
  expect_equal(dim(res$res), c(2, 5, 20))
  expect_equal(dim(res$pred), c(2, 20))
  expect_equal(length(res$time), 20)
})
