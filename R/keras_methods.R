#' Generate Predictions for Keras Models
#'
#' S3 method for generating predictions from Keras models for use with flexplot's
#' compare.fits function.
#'
#' @param model A fitted Keras model object
#' @param re Should random effects be predicted? (Not applicable for neural networks, ignored)
#' @param pred.values Data frame containing predictor values for prediction
#' @param pred.type Type of predictions (ignored for neural networks)
#' @param report.se Should standard errors be reported? (Not supported for neural networks)
#' @return A data frame with columns "prediction" and "model"
#' @export
#' @importFrom flexplot generate_predictions
#' @method generate_predictions keras.engine.training.Model
generate_predictions.keras.engine.training.Model <- function(model, re, pred.values, pred.type, report.se) {

  # Convert pred.values to matrix format expected by Keras
  # Assumes all predictors are numeric - may need adjustment for categorical variables
  pred_matrix <- as.matrix(pred.values)
  
  # Generate predictions
  predictions <- keras::predict(model, pred_matrix)
  
  # Handle different output shapes
  if (is.matrix(predictions) && ncol(predictions) == 1) {
    predictions <- as.vector(predictions)
  } else if (is.matrix(predictions) && ncol(predictions) > 1) {
    # For multi-class classification, take the class with highest probability
    predictions <- apply(predictions, 1, which.max) - 1  # Convert to 0-based indexing
  }
  
  return(data.frame(prediction = predictions, model = "keras"))
}

#' @export
generate_predictions.keras.src.engine.sequential.Sequential = function(model, re, pred.values, pred.type, report.se) {
  
  # Get stored information from training
  var_names = attr(model, "var_names")
  x_means = attr(model, "x_means")
  x_sds = attr(model, "x_sds")
  
  # Create full prediction matrix initialized with training means
  full_pred_matrix = matrix(rep(x_means, each = nrow(pred.values)), 
                            nrow = nrow(pred.values), 
                            ncol = length(var_names))
  colnames(full_pred_matrix) = var_names
  
  # Replace with provided values (vectorized)
  common_vars = intersect(names(pred.values), var_names)
  full_pred_matrix[, common_vars] = as.matrix(pred.values[common_vars])
  
  # Normalize the full matrix
  if (!is.null(x_means) && !is.null(x_sds)) {
    full_pred_matrix = scale(full_pred_matrix, center = x_means, scale = x_sds)
  }
  
  # Generate predictions
  predictions = predict(model, full_pred_matrix)
  
  if (is.matrix(predictions) && ncol(predictions) == 1) {
    predictions = as.vector(predictions)
  }
  
  return(data.frame(prediction = predictions, model = "keras"))
}


#' Get Terms from Keras Models
#'
#' Extract predictor and response variable names from Keras models.
#' This function extends flexplot's get_terms function to work with Keras models.
#'
#' @param model A fitted Keras model object
#' @return A list with elements "predictors" and "response"
#' @export
get_terms.keras.engine.training.Model <- function(model) {
  
  # For Keras models, we need to extract variable information differently
  # This is a basic implementation - you may need to store variable names
  # during model training or pass them as attributes
  
  # Get input shape (excluding batch dimension)
  input_shape <- model$input_shape
  if (is.list(input_shape)) {
    n_predictors <- input_shape[[2]]  # Assuming single input layer
  } else {
    n_predictors <- input_shape[2]
  }
  
  # Create generic predictor names - ideally these would be stored with the model
  predictors <- paste0("X", 1:n_predictors)
  
  # For neural networks, we typically don't know the response variable name
  # This would need to be stored as a model attribute or passed separately
  response <- attr(model, "response_var") %||% "Y"
  
  return(list(predictors = predictors, response = response))
}

#' @method get_terms keras.src.engine.sequential.Sequential
#' @export
get_terms.keras.src.engine.sequential.Sequential = function(model) {
  
  # Get the original data variable names (before model.matrix expansion)
  original_data_vars = attr(model, "original_data_vars")
  response = attr(model, "response_var")
  
  if (!is.null(original_data_vars) && !is.null(response)) {
    # Remove the response variable from predictors
    predictors = setdiff(original_data_vars, response)
    return(list(predictors = predictors, response = response))
  }
  
  # Fallback if not available
  predictors = attr(model, "var_names")
  if (is.null(predictors)) {
    input_shape = model$input_shape
    n_predictors = if(is.list(input_shape)) input_shape[[2]] else input_shape[2]
    predictors = paste0("X", 1:n_predictors)
  }
  
  if (is.null(response)) response = "Y"
  
  return(list(predictors = predictors, response = response))
}

# Helper function for null coalescing
`%||%` <- function(x, y) if (is.null(x)) y else x

# Force registration of the method
.onLoad = function(libname, pkgname) {
  # Register S3 methods
  registerS3method("get_model_n", "keras.src.engine.sequential.Sequential", 
                   get_model_n.keras.src.engine.sequential.Sequential)
}

#' @importFrom flexplot get_model_n
#' @method get_model_n keras.src.engine.sequential.Sequential
#' @export
get_model_n.keras.src.engine.sequential.Sequential = function(model) {
  # Get the number of samples from stored attribute
  n_samples = attr(model, "n_samples")
  if (!is.null(n_samples)) return(n_samples)
  
  # If not available, return NULL
  return(NULL)
}


  #' Compute Variable Importance for Neural Network Models
  #'
  #' S3 method for computing variable importance using permutation importance
  #' for neural network models created with nn().
  #'
  #' @param object A fitted nn_model object
  #' @param metric Character string specifying the metric to use for importance.
  #'   Default is "mean_absolute_error". Other options include "loss", "accuracy", etc.
  #' @param ... Additional arguments (currently unused)
  #'
  #' @return A data frame with columns "variable" and "importance", sorted by
  #'   importance in descending order.
  #'
  #' @details
  #' This method uses permutation importance to assess variable importance.
  #' For each variable, it permutes that variable's values and measures the
  #' increase in prediction error. Higher increases indicate more important variables.
  #' @importFrom flexplot estimates
  #'
#' @method estimates nn_model
#' @export
estimates.nn_model = function(object, metric = NULL, return_metrics = TRUE, ...) {
  
  # Extract components from nn_model object
  model = object$model
  x_test = object$x
  y_test = object$y
  x_means = object$x_means
  x_sds = object$x_sds
  
  # Normalize test data (same as training)
  x_test_scaled = scale(x_test, center = x_means, scale = x_sds)
  
  # Get validation metrics from training history
  val_metrics = object$history$metrics
  final_epoch = length(val_metrics$val_loss)
  
  # Extract all validation metrics
  val_metric_names = names(val_metrics)[grepl("^val_", names(val_metrics))]
  val_values = sapply(val_metric_names, function(m) val_metrics[[m]][final_epoch])
  
  # Clean up metric names (remove "val_" prefix for display)
  clean_names = gsub("^val_", "", val_metric_names)
  
  # Auto-detect appropriate metric if not specified
  if (is.null(metric)) {
    metric = if ("val_accuracy" %in% val_metric_names) "val_accuracy" else "val_loss"
  }
  
  # Add "val_" prefix if not already there
  if (!grepl("^val_", metric)) {
    metric = paste0("val_", metric)
  }
  
  # Check if specified metric exists
  if (!metric %in% val_metric_names) {
    available_metrics = paste(clean_names, collapse = ", ")
    stop(paste("Metric", gsub("^val_", "", metric), "not available. Available metrics:", available_metrics))
  }
  
  baseline_score = val_metrics[[metric]][final_epoch]
  
  # Compute permutation importance using training data
  importances = purrr::map_dbl(1:ncol(x_test_scaled), function(i) {
    x_perm = x_test_scaled
    x_perm[, i] = sample(x_perm[, i])  # permute column i
    perm_score = model %>% keras::evaluate(x_perm, y_test, verbose = 0)
    
    # Get the corresponding training metric name (without val_ prefix)
    train_metric = gsub("^val_", "", metric)
    
    # For metrics where higher is better (accuracy, auc), we want baseline - permuted
    # For metrics where lower is better (loss, mse), we want permuted - baseline
    if (train_metric %in% c("accuracy", "auc", "precision", "recall", "f1_score")) {
      baseline_score - perm_score[[train_metric]]  # decrease in performance = importance
    } else {
      perm_score[[train_metric]] - baseline_score  # increase in error = importance
    }
  })
  
  # Create variable importance data frame
  importance_df = data.frame(
    variable = colnames(x_test),
    importance = importances
  )
  
  # Sort by importance (descending)
  importance_df = importance_df[order(importance_df$importance, decreasing = TRUE), ]
  rownames(importance_df) = NULL
  
  # Create metrics data frame using validation metrics
  metrics_df = data.frame(
    metric = clean_names,
    value = unlist(val_values),
    stringsAsFactors = FALSE
  )
  rownames(metrics_df) = NULL
  
  # Return structure
  if (return_metrics) {
    return(list(
      importance = importance_df,
      metrics = metrics_df,
      primary_metric = gsub("^val_", "", metric),
      primary_value = baseline_score
    ))
  } else {
    return(importance_df)
  }
}