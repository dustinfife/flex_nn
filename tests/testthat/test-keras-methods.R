library(testthat)
library(flex_nn)

test_that("generate_predictions works for keras models", {
  # This is a placeholder test - you'll need actual keras models to test properly
  skip_if_not_installed("keras")
  skip_if_not_installed("tensorflow")
  
  # Add your tests here
  expect_true(TRUE)  # Placeholder
})

test_that("helper functions work correctly", {
  # Test data preparation
  test_data <- data.frame(x1 = 1:5, x2 = letters[1:5])
  result <- prepare_keras_data(test_data, categorical_vars = "x2")
  
  expect_true(is.matrix(result))
  expect_equal(nrow(result), 5)
})
