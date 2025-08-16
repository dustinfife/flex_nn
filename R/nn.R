#' Fit Neural Network Models Using Formula Syntax
#'
#' This function provides a formula-based interface for training neural networks
#' using Keras/TensorFlow. It handles data preprocessing, normalization, and 
#' provides flexible layer specification while maintaining compatibility with
#' flexplot's compare.fits function.
#'
#' @param formula A formula specifying the model (e.g., y ~ x1 + x2 + x3)
#' @param data A data frame containing the variables specified in the formula
#' @param epochs Integer. Number of training epochs (default: 100)
#' @param batch_size Integer. Batch size for training (default: 4)
#' @param validation_split Numeric. Fraction of data to use for validation (default: 0.2)
#' @param layers Function or NULL. A function that takes (model, input_dim) and returns
#'   the model with layers added. If NULL, uses a default two-hidden-layer architecture
#' @param optimizer Keras optimizer object. Default is optimizer_adam()
#' @param loss Character string or Keras loss function. Default is "mse"
#' @param metrics List of metrics to track during training. Default is "mean_absolute_error"
#' @param verbose Integer. Verbosity level (0 = silent, 1 = progress bar, 2 = one line per epoch)
#' @param ... Additional arguments (reserved for future use)
#' @importFrom flexplot get_terms
#' @return A list containing:
#'   \item{model}{The trained Keras model}
#'   \item{history}{Training history object}
#'   \item{x_means}{Column means used for normalization}
#'   \item{x_sds}{Column standard deviations used for normalization}
#'   \item{formula}{The original formula}
#'   \item{x}{The original predictor matrix (before scaling)}
#'   \item{y}{The original response vector}
#'   \item{var_names}{Names of predictor variables}
#'
#' @details
#' The function automatically:
#' \itemize{
#'   \item Extracts predictors and response from the formula
#'   \item Handles factor variables through model.matrix expansion
#'   \item Normalizes predictors using z-score standardization
#'   \item Stores normalization parameters for later prediction
#'   \item Creates a sequential Keras model with specified architecture
#' }
#'
#' The default layer architecture consists of:
#' \itemize{
#'   \item Input layer with 8 units and ReLU activation
#'   \item Hidden layer with 4 units and ReLU activation  
#'   \item Output layer with 1 unit (linear activation)
#' }
#'
#' For custom architectures, provide a layers function that takes the model object
#' and input dimension, then adds layers and returns the modified model.
#'
#' @examples
#' \dontrun{
#' # Load required libraries
#' library(keras)
#' library(tensorflow)
#' 
#' # Basic usage with default architecture
#' data(mtcars)
#' model1 = nn(mpg ~ hp + wt + cyl, data = mtcars, epochs = 50)
#' 
#' # Custom layer architecture with dropout
#' custom_layers = function(model, input_dim) {
#'   model %>%
#'     layer_dense(units = 64, activation = "relu", input_shape = input_dim) %>%
#'     layer_dropout(rate = 0.3) %>%
#'     layer_dense(units = 32, activation = "relu") %>%
#'     layer_dropout(rate = 0.2) %>%
#'     layer_dense(units = 16, activation = "relu") %>%
#'     layer_dense(units = 1)
#' }
#' 
#' model2 = nn(mpg ~ hp + wt + cyl, data = mtcars, 
#'             layers = custom_layers, epochs = 100)
#' 
#' # Deep network example
#' deep_layers = function(model, input_dim) {
#'   model %>%
#'     layer_dense(units = 128, activation = "relu", input_shape = input_dim) %>%
#'     layer_batch_normalization() %>%
#'     layer_dense(units = 64, activation = "relu") %>%
#'     layer_batch_normalization() %>%
#'     layer_dense(units = 32, activation = "relu") %>%
#'     layer_dense(units = 16, activation = "relu") %>%
#'     layer_dense(units = 1)
#' }
#' 
#' model3 = nn(mpg ~ ., data = mtcars, layers = deep_layers, 
#'             epochs = 200, batch_size = 8)
#' 
#' # Classification example (binary)
#' classification_layers = function(model, input_dim) {
#'   model %>%
#'     layer_dense(units = 32, activation = "relu", input_shape = input_dim) %>%
#'     layer_dense(units = 16, activation = "relu") %>%
#'     layer_dense(units = 1, activation = "sigmoid")
#' }
#' 
#' # Convert to binary outcome
#' mtcars$high_mpg = as.numeric(mtcars$mpg > median(mtcars$mpg))
#' model4 = nn(high_mpg ~ hp + wt + cyl, data = mtcars,
#'             layers = classification_layers,
#'             loss = "binary_crossentropy",
#'             metrics = list("accuracy"))
#' 
#' # Use with flexplot's compare.fits
#' library(flexplot)
#' lm_model = lm(mpg ~ hp + wt + cyl, data = mtcars)
#' compare.fits(mpg ~ hp | wt, data = mtcars, lm_model, model1$model)
#' }
#'
#' @export
#' @author Dustin Fife
#' @seealso \code{\link[keras]{keras_model_sequential}}, \code{\link[flexplot]{compare.fits}}
nn = function(formula, data, epochs = 100, batch_size = 4, validation_split = 0.2, 
              layers = NULL, optimizer = optimizer_adam(), loss = "mse", 
              metrics = list("mean_absolute_error"), verbose = 0, ...) {
  
  # Check required packages
  if (!requireNamespace("keras", quietly = TRUE)) {
    stop("Package 'keras' is required but not installed.")
  }
  if (!requireNamespace("tensorflow", quietly = TRUE)) {
    stop("Package 'tensorflow' is required but not installed.")
  }
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required but not installed.")
  }
  
  # Load required libraries
  require(keras)
  require(tensorflow)
  require(reticulate)
  
  tt = terms(formula, data = data)
  mf = model.frame(tt, data = data)
  
  # Store the original data variable names (before expansion)
  original_vars = names(data)
  response_var = all.vars(formula)[1]
  
  # Save all factor (and ordered factor) levels from raw data
  factor_levels = lapply(original_vars, function(x) {
    if (is.factor(data[,x]) || is.ordered(data[,x])) levels(data[,x]) else NULL
  })
  are_factors = which(!sapply(factor_levels, is.null))
  factor_levels = factor_levels[are_factors]
  names(factor_levels) = original_vars[are_factors]
  # Continue with your existing code...
  y = model.response(mf)
  x = model.matrix(tt, data = data)
  
  # Extract response and predictors from formula
  tt = terms(formula, data = data)
  mf = model.frame(tt, data = data)
  y = model.response(mf)
  x = model.matrix(tt, data = data)
  
  # Convert response variable to appropriate format
  y = convert_response_variable(y)
  
  # Remove intercept column if present
  if ("(Intercept)" %in% colnames(x)) {
    x = x[, !colnames(x) %in% "(Intercept)", drop = FALSE]
  }
  
  # Store variable names for later use
  var_names = colnames(x)
  
  # Force proper types
  x = as.matrix(x)
  y = as.numeric(y)
  
  # Check for empty predictor matrix
  if (ncol(x) == 0) {
    stop("No predictors found in formula. Please specify at least one predictor variable.")
  }
  
  # Normalize predictors
  x_means = colMeans(x, na.rm = TRUE)
  x_sds = apply(x, 2, sd, na.rm = TRUE)
  
  # Avoid division by zero for constant predictors
  x_sds[x_sds == 0] = 1
  
  x_scaled = scale(x, center = x_means, scale = x_sds)
  
  # Default layers function if none provided
  if (is.null(layers)) {
    layers = function(model, input_dim) {
      model %>% 
        layer_dense(units = 8, activation = "relu", input_shape = input_dim) %>%
        layer_dense(units = 4, activation = "relu") %>%
        layer_dense(units = 1)
    }
  }
  
  # Define model
  model = keras_model_sequential()
  model = layers(model, ncol(x_scaled))
  
  # Compile model
  model %>% compile(
    loss = loss,
    optimizer = optimizer,
    metrics = metrics
  )
  
  # Fit model
  history = model %>% keras::fit(
    x_scaled, y,
    epochs = epochs,
    batch_size = batch_size,
    validation_split = validation_split,
    verbose = verbose
  )
  

  

  # Attach to model
  attr(model, "factor_levels") = factor_levels
  attr(model, "var_names") = var_names
  attr(model, "response_var") = all.vars(formula)[1]  # First variable is response
  attr(model, "n_samples") = nrow(x_scaled)
  attr(model, "var_names") = var_names
  attr(model, "response_var") = all.vars(formula)[1]
  attr(model, "x_means") = x_means      
  attr(model, "x_sds") = x_sds          
  attr(model, "original_data_vars") = original_vars
  attr(model, "response_var") = response_var
  attr(model, "var_names") = var_names  # Keep this for prediction matrix reconstruction
  
  # Create result object
  result = list(
    model = model, 
    history = history, 
    x_means = x_means, 
    x_sds = x_sds, 
    formula = formula, 
    x = x, 
    y = y,
    var_names = var_names,
    original_names = original_vars
  )
  
  # Set class for S3 methods
  class(result) = c("nn_model", class(result))
  

  return(result)
}