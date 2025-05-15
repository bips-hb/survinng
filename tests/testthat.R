# This file is part of the standard setup for testthat.
# It is recommended that you do not modify it.
#
# Where should you do additional test configuration?
# Learn more about the roles of various files in:
# * https://r-pkgs.org/testing-design.html#sec-tests-files-overview
# * https://testthat.r-lib.org/articles/special-files.html

library(testthat)
library(Survinng)

if (Sys.getenv("TORCH_TEST", unset = 0) == 1) {
  set.seed(42)
  torch::torch_manual_seed(42)

  library(torch)

  test_check("Survinng")
}
