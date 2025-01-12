################################################################################
#                       Helper functions for tests
################################################################################

generate_survival_data <- function(n, p_cont, p_binary = 0, seed = 1) {
  # Load libraries
  requireNamespace("simsurv")

  # Set seed
  set.seed(seed)

  # Generate covariates
  x_cont <- matrix(rnorm(n * p_cont), nrow = n)
  x_binary <- matrix(rbinom(n * p_binary, 1, 0.5), nrow = n)
  x <- cbind(x_binary, x_cont)
  colnames(x) <- c(paste0("binary", 1:p_binary), paste0("cont", 1:p_cont))
  dat <- data.frame(x)


  # Generate survival data
  betas <- seq(-1, 1, length.out = p_cont + p_binary)
  names(betas) <- colnames(dat)
  surv_dat <- simsurv(dist = "weibull", lambdas = 0.1, gammas = 1.5, betas = betas,
                      x = dat, maxt = 5, seed = seed)

  # Add time and event to `dat`
  dat$time <- surv_dat$eventtime
  dat$status <- surv_dat$status

  # Set train and test data
  idx <- sample(n, 0.7 * n)
  train <- dat[idx, ]
  test <- dat[-idx, ]

  # Return
  list(train = train, test = test)
}

# Generate a sequential neural network module (1D input)
seq_model_1D <- nn_module(
  initialize = function(num_inputs, num_outputs = 1, last_act = "none") {
    self$fc1 <- nn_linear(num_inputs, 20)
    self$fc2 <- nn_linear(20, num_outputs)
    self$last_act <- last_act
  },
  forward = function(input) {
    input <- self$fc1(input)
    input <- nnf_relu(input)
    input <- self$fc2(input)

    if (self$last_act == "sigmoid") {
      input <- nnf_sigmoid(input)
    } else if (self$last_act == "softmax") {
      input <- nnf_softmax(input, dim = -1)
    }

    input
  }
)

# Generate multi-modal model (tabular + image data)
# Inputs: tabular data (numeric), image data (3x10x10)
multi_modal_model <- nn_module(
  initialize = function(num_tabular_inputs = 4, num_outputs = 1, add_time = FALSE) {
    if (add_time) {
      num_tabular_inputs <- num_tabular_inputs + 1
    }

    self$tabular_model <- nn_sequential(
      nn_linear(num_tabular_inputs, 20),
      nn_relu(),
      nn_linear(20, 10)
    )

    self$image_model <- nn_sequential(
      nn_conv2d(3, 10, kernel_size = 3),
      nn_relu(),
      nn_avg_pool2d(kernel_size = 2),
      nn_flatten(),
      nn_linear(160, 10)
    )

    self$fc1 <- nn_linear(10 + 10, 20)
    self$fc2 <- nn_linear(20, num_outputs)
  },
  forward = function(input) {
    image_input <- self$image_model(input[[1]])
    tabular_input <- self$tabular_model(input[[2]])

    input <- torch_cat(list(tabular_input, image_input), dim = -1)
    input <- self$fc1(input)
    input <- nnf_relu(input)
    input <- self$fc2(input)
    input
  }
)

get_base_hazard <- function(n) {
  # Generate baseline hazard
  time <- seq(0, 10, length.out = n)
  hazard <- exp(-(time - 5)^2 / 5)

  data.frame(time = time, hazard = hazard)

}
