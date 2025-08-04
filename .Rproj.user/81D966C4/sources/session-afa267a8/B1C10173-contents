library(testthat)

options(warn = -1)
# Suppress TensorFlow warnings at the system level
Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "3")  # Suppress all TF logs except errors
Sys.setenv(PYTHONWARNINGS = "ignore")   # Suppress Python warnings
legacy_adam = tensorflow::tf$keras$optimizers$legacy$Adam()
# Helper function to check if keras/tensorflow are available
check_keras_available = function() {
  skip_if_not_installed("keras")
  skip_if_not_installed("tensorflow")
  skip_if_not_installed("reticulate")
  
  # Set environment variables BEFORE loading tensorflow/keras
  Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "3")
  Sys.setenv(PYTHONWARNINGS = "ignore")
  Sys.setenv(TF_ENABLE_ONEDNN_OPTS = "0")  # For newer TF versions
  
  # Try to load keras and check if backend is available
  tryCatch({
    library(keras)
    library(tensorflow)
    
    # Additional tensorflow logging suppression
    tryCatch({
      tensorflow::tf$get_logger()$setLevel('ERROR')
      tensorflow::tf$compat$v1$logging$set_verbosity(tensorflow::tf$compat$v1$logging$ERROR)
    }, error = function(e) {})
    
    # Simple test to see if we can create a model
    test_model = keras_model_sequential()
    TRUE
  }, error = function(e) {
    skip("Keras/TensorFlow not properly configured")
  })
}

test_that("nn function handles constant predictors correctly", {
  check_keras_available()
  
  test_data = data.frame(
    y = rnorm(10),
    x1 = rnorm(10),
    x2 = rep(5, 10)
  )
  
  # Open connection to null file
  null_con = file(nullfile(), open = "w")
  
  # Redirect stderr
  sink(null_con, type = "message")
  legacy_adam = tensorflow::tf$keras$optimizers$legacy$Adam()
  result = nn(y ~ x1 + x2, data = test_data, epochs = 2, verbose = 0, optimizer = legacy_adam)
  
  # Reset and close
  sink(type = "message")
  close(null_con)
  
  expect_equal(as.numeric(result$x_sds[2]), 1)
})

test_that("nn function handles basic regression correctly", {
  check_keras_available()
  
  # Create simple test data
  set.seed(123)
  test_data = data.frame(
    y = 1:20 + rnorm(20, 0, 0.1),
    x1 = 1:20,
    x2 = (1:20)^2
  )
  
  # Fit model with minimal epochs for speed
  result = nn(y ~ x1 + x2, data = test_data, epochs = 5, verbose = 0, optimizer = legacy_adam)
  
  # Test return structure
  expect_s3_class(result, "nn_model")
  expect_named(result, c("model", "history", "x_means", "x_sds", 
                         "formula", "x", "y", "var_names"))
  
  # Test components
  expect_s3_class(result$model, "keras.engine.sequential.Sequential")
  expect_equal(result$formula, y ~ x1 + x2)
  expect_equal(length(result$y), 20)
  expect_equal(ncol(result$x), 2)
  expect_equal(result$var_names, c("x1", "x2"))
  expect_equal(length(result$x_means), 2)
  expect_equal(length(result$x_sds), 2)
})

test_that("nn function handles custom layers correctly", {
  check_keras_available()
  
  # Create test data
  set.seed(456)
  test_data = data.frame(
    y = rnorm(15),
    x1 = rnorm(15),
    x2 = rnorm(15)
  )
  
  # Define custom layers
  custom_layers = function(model, input_dim) {
    model %>%
      layer_dense(units = 10, activation = "relu", input_shape = input_dim) %>%
      layer_dense(units = 5, activation = "relu") %>%
      layer_dense(units = 1)
  }
  
  result = nn(y ~ x1 + x2, data = test_data, 
              layers = custom_layers, epochs = 3, verbose = 0, optimizer = legacy_adam)
  
  expect_s3_class(result, "nn_model")
  expect_s3_class(result$model, "keras.engine.sequential.Sequential")
  
  # Check that model has expected architecture
  model_config = get_config(result$model)
  layer_configs = lapply(model_config$layers, function(x) x$config)
  
  # Should have 3 dense layers with specified units
  dense_layers = which(sapply(layer_configs, function(x) {
    !is.null(x$units) && x$units %in% c(10, 5, 1)
  }))
  
  # Or even simpler, just check that you have the right number of layers:
  dense_layer_count = sum(sapply(layer_configs, function(x) !is.null(x$units)))
  expect_equal(dense_layer_count, 3)  # Expecting 3 dense layers
})

test_that("nn function handles factor variables correctly", {
  check_keras_available()
  
  # Create data with factor variable
  set.seed(789)
  test_data = data.frame(
    y = rnorm(20),
    x1 = rnorm(20),
    x2 = factor(rep(c("A", "B", "C", "D"), 5))
  )
  
  result = nn(y ~ x1 + x2, data = test_data, epochs = 3, verbose = 0, optimizer = legacy_adam)
  
  expect_s3_class(result, "nn_model")
  
  # Should have expanded the factor variable
  # x1 (1 column) + x2 (3 dummy variables for 4 levels) = 4 columns
  expect_equal(ncol(result$x), 4)
  expect_equal(length(result$var_names), 4)
  expect_true(any(grepl("x2", result$var_names)))
})

test_that("nn function handles intercept removal correctly", {
  check_keras_available()
  
  set.seed(101)
  test_data = data.frame(y = rnorm(10), x1 = rnorm(10))
  
  result = nn(y ~ x1, data = test_data, epochs = 2, verbose = 0, optimizer = legacy_adam)
  
  # Should not have intercept column
  expect_false("(Intercept)" %in% result$var_names)
  expect_equal(ncol(result$x), 1)
})

test_that("nn function validates inputs correctly", {
  check_keras_available()
  
  test_data = data.frame(y = 1:5, x1 = 1:5)
  
  # Test empty predictor formula
  expect_error(nn(y ~ 1, data = test_data), 
               "No predictors found in formula")
  
  # Test missing variables
  expect_error(nn(y ~ nonexistent_var, data = test_data))
})

test_that("nn function handles normalization correctly", {
  check_keras_available()
  
  # Create data with different scales
  set.seed(202)
  test_data = data.frame(
    y = rnorm(15),
    x1 = rnorm(15, mean = 100, sd = 50),  # Large scale
    x2 = rnorm(15, mean = 0.1, sd = 0.01)  # Small scale
  )
  
  result = nn(y ~ x1 + x2, data = test_data, epochs = 2, verbose = 0, optimizer = legacy_adam)
  
  # Check that means and sds are captured correctly
  expect_equal(length(result$x_means), 2)
  expect_equal(length(result$x_sds), 2)
  expect_true(abs(result$x_means[1] - 100) < 50)  # Rough check
  expect_true(abs(result$x_means[2] - 0.1) < 0.1)   # Rough check
})

test_that("nn function handles constant predictors correctly", {
  check_keras_available()
  
  # Create data with constant predictor
  test_data = data.frame(
    y = rnorm(10),
    x1 = rnorm(10),
    x2 = rep(5, 10)  # Constant
  )
  
  # Should not error due to zero standard deviation
  expect_no_error({
    result = nn(y ~ x1 + x2, data = test_data, epochs = 2, verbose = 0, optimizer = legacy_adam)
  })
  
  result = nn(y ~ x1 + x2, data = test_data, epochs = 2, verbose = 0, optimizer = legacy_adam)
  
  # Standard deviation should be set to 1 for constant predictor
  expect_equal(as.numeric(result$x_sds[2]), 1)
})

test_that("nn function works with different loss functions and optimizers", {
  check_keras_available()
  
  set.seed(303)
  test_data = data.frame(
    y = rbinom(20, 1, 0.5),  # Binary outcome
    x1 = rnorm(20),
    x2 = rnorm(20)
  )
  
  # Custom layers for classification
  classification_layers = function(model, input_dim) {
    model %>%
      layer_dense(units = 5, activation = "relu", input_shape = input_dim) %>%
      layer_dense(units = 1, activation = "sigmoid")
  }
  
  result = nn(y ~ x1 + x2, data = test_data,
              layers = classification_layers,
              loss = "binary_crossentropy",
              optimizer = tensorflow::tf$keras$optimizers$legacy$SGD(), 
              metrics = list("accuracy"),
              epochs = 3, verbose = 0)
  
  expect_s3_class(result, "nn_model")
  expect_s3_class(result$model, "keras.engine.sequential.Sequential")
})

test_that("nn function works with all formula variations", {
  check_keras_available()
  
  set.seed(404)
  test_data = data.frame(
    y = rnorm(15),
    x1 = rnorm(15),
    x2 = rnorm(15),
    x3 = rnorm(15)
  )
  
  # Test different formula specifications
  result1 = nn(y ~ ., data = test_data, epochs = 2, verbose = 0, optimizer = legacy_adam)
  expect_equal(ncol(result1$x), 3)
  
  result2 = nn(y ~ x1 + x2, data = test_data, epochs = 2, verbose = 0, optimizer = legacy_adam)
  expect_equal(ncol(result2$x), 2)
  
  result3 = nn(y ~ x1 * x2, data = test_data, epochs = 2, verbose = 0, optimizer = legacy_adam)
  expect_true(ncol(result3$x) >= 3)  # Should include interaction
})

test_that("nn function handles small datasets", {
  check_keras_available()
  
  # Very small dataset
  test_data = data.frame(
    y = c(1, 2, 3),
    x1 = c(1, 2, 3)
  )
  
  expect_no_error({
    result = nn(y ~ x1, data = test_data, epochs = 1, batch_size = 1, 
                validation_split = 0, verbose = 0, optimizer = legacy_adam)
  })
  
  result = nn(y ~ x1, data = test_data, epochs = 1, batch_size = 1, 
              validation_split = 0, verbose = 0, optimizer = legacy_adam)
  expect_s3_class(result, "nn_model")
})

test_that("nn function preserves data types correctly", {
  check_keras_available()
  
  set.seed(505)
  test_data = data.frame(
    y = as.numeric(1:10),
    x1 = as.integer(1:10),
    x2 = as.double(rnorm(10))
  )
  
  result = nn(y ~ x1 + x2, data = test_data, epochs = 2, verbose = 0, optimizer = legacy_adam)
  
  # All should be converted to numeric matrices
  expect_true(is.matrix(result$x))
  expect_true(is.numeric(result$x))
  expect_true(is.numeric(result$y))
})

options(warn = 0)
