
# `Survinng`: Gradient-based Explanations for Deep Learning Survival Models

<!-- badges: start -->
<a href="https://bips-hb.github.io/Survinng/"><img src="man/figures/logo.jpeg" align="right" height="120" alt="Survinng website" /></a>
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2502.04970-b31b1b.svg)](https://arxiv.org/abs/2502.04970)
[![Website](https://img.shields.io/badge/docs-📘%20Survinng%20Site-blue)](https://bips-hb.github.io/Survinng/)
[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![R-CMD-check](https://github.com/bips-hb/Survinng/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/bips-hb/Survinng/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->


The package `Survinng` provides gradient-based feature attribution methods 
for deep learning survival models. It implements novel adaptions of model-specific
XAI methods for the survival analysis domain, like **Grad(t)**, **SG(t)**, **GxI(t)**,
**IntGrad(t)**, and **GradSHAP(t)**. The whole package is designed to be compatible with the
[`survivalmodels`](https://github.com/RaphaelS1/survivalmodels) package in R, 
which is an R wrapper for the [`pycox`](https://github.com/havakv/pycox) Python package.
However, it can also be used with models from `pycox` directly and other 
survival models loaded in `torch`. Currently, the package supports the models
types `DeepSurv`/`CoxPH`, `DeepHit`, and `CoxTime`.

### 🚀 Why `Survinng`?

With `Survinng`, you get:

* ⏱ **Model-specific** and **time-resolved** feature attributions for individuals  
* ⚡️ **Fast and scalable** explanations using model gradients, especially for SHAP-like explanations  
* 🤝 Compatible with `survivalmodels` and `pycox` models  
* 📊 Easy-to-use **visualization tools** for temporal insights  
* 🔀 Support for **multimodal** inputs (e.g., tabular + image)  


Overall, this package is part of the following ICML'25 paper:  

> 📄 _Based on the ICML 2025 paper:_  
> [_Gradient-based Explanations for Deep Learning Survival Models_](https://arxiv.org/pdf/2502.04970)  
> _Sophie Hanna Langbein, Niklas Koenen, Marvin N. Wright_


## 📦 Installation

To install the latest development version directly from GitHub:

```r
# install.packages("devtools")
devtools::install_github("bips-hb/Survinng")
```

## 📖 Usage

You have a trained survival neural network model from `survivalmodels` or
`pycox` and your model input data data. Now you want to interpret individual 
data points by using the methods from the package `Survinng`, then stick to the 
following pseudo code:

```r
library(Survinng)

# Load a survival model and corresponding data
model <- ... (e.g., from survivalmodels or pycox)
data <- ... (e.g., the test set of the model)

# Create explainer object
explainer <- Survinng::explain(model, data)

# Compute feature attributions
idx <- 1 # index of the instance to explain
grad <- surv_grad(explainer, instance = idx) # Grad(t)
sg <- surv_smoothgrad(explainer, instance = idx) # SG(t)
gxi <- surv_grad(explainer, instance = idx, times_input = TRUE) # GxI(t)
ig <- surv_intgrad(explainer, instance = idx) # IntGrad(t)
shap <- surv_gradSHAP(explainer, instance = idx) # GradSHAP(t)

# Plot results
plot(shap)
```
👉 For full documentation and advanced use cases, visit the 
[📘 package website](https://bips-hb.github.io/Survinng/).

## 💻 Quick Example

```r
library(survival)
library(callr)

# Load lung dataset
data(cancer, package="survival")
data <- na.omit(cancer[, c(1, 4, 5, 6, 7, 10, 2, 3)])
train <- data[1:150, ]
test <- data[151:212, ]

# Train a DeepSurv model
ext <- callr::r(function(train) {
  library(survivalmodels)
  library(survival)
  
  # Fit the DeepSurv model
  install_pycox(install_torch = TRUE) # Requires pycox
  fit <- deepsurv(Surv(time, status) ~ ., data = train, epochs = 100,
                  early_stopping = TRUE)

  # Extract the model
  Survinng::extract_model(fit)
}, args = list(train = train))


# Create explainer
explainer <- explain(ext, data = test)

# Run GradSHAP(t)
shap <- surv_gradSHAP(explainer)

# Plot results
plot(shap)
```

## 🖥 Other Examples and Articles

- Simulation: Time-independent effects (`survivalmodels`) [→ article](https://bips-hb.github.io/Survinng/articles/Sim_time_independent.html)
- Simulation: Time-dependent effects (`survivalmodels`) [→ article](https://bips-hb.github.io/Survinng/articles/Sim_time_dependent.html)

## 📚 Citation

If you use this package in your research, please cite it as follows:

```bibtex
@article{langbein2025grad,
  title={Gradient-based Explanations for Deep Learning Survival Models},
  author={Langbein, Sophie Hanna and Koenen, Niklas and Wright, Marvin N.},
  journal={arXiv preprint arXiv:2502.04970},
  year={2025}
}
```
