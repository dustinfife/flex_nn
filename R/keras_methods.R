


get_sequence_of_target_variable = function(x_label, data) {
  
  data[,paste0(x_label, "_sequence")] = data[,x_label]
  x_variable = data[,x_label]
  if (is.factor(x_variable)) return(data)
  if (!is.numeric(x_variable)) return(data)
  
  unique_values = length(unique(x_variable))
  if (unique_values < 10) return(data)
  
  
  bins = quantile(x_variable, probs = seq(from=0, to=1, length.out=10)) %>% unique
  new_labels = (diff(bins)/2)+bins
  new_labels = new_labels[-length(new_labels)]
  data[,paste0(x_label, "_sequence")] = cut(x_variable, breaks=bins, include.lowest=T, new_labels)
  return(data)
  
}



# get_terms methods -------------------------------------------------------
#' @export
get_terms.keras.engine.training.Model <- function(model) {
  
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

#' @importFrom flexplot get_terms
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

#' @export
get_terms.nn_model = function(model) {
  
  nn_model = model
  model = model$model
  
  # Get the original data variable names (before model.matrix expansion)
  original_data_vars = attr(model, "original_data_vars")
  response = attr(model, "response_var")
  
  # remove DV from predictors
  if (!is.null(original_data_vars) && !is.null(response)) {
    # Remove the response variable from predictors
    predictors = setdiff(original_data_vars, response)
    return(list(predictors = predictors, response = response))
  }
  
  return(list(predictors = predictors, response = response))
}

# get_model_n methods -----------------------------------------------------
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

# get_fitted methods ------------------------------------------------------
#' @importFrom flexplot get_fitted
#' @export
get_fitted.nn_model = function(model, re=FALSE, pred.values, pred.type="response", report.se=FALSE) {
  
  # extract data
  pred.values = model$x
  
  # Separate features and target 
  X = model$x
  y = model$y
  
  # Convert to matrix and scale (X is already a matrix)
  X_scaled = scale(X)
  
  # Make predictions
  prediction = predict(model$model, X_scaled)
  return(prediction)
}

nn_predict = function(model, newdata) {
  
  # extract data
  X = newdata
  
  # Convert to matrix and scale (X is already a matrix)
  X_scaled = scale(X)
  
  # Make predictions
  prediction = predict(model, X_scaled)
  return(prediction)
}


#' @importFrom flexplot post_prediction_process_cf
#' @export
post_prediction_process_cf.nn_model = function(model1, model2=NULL, predictions, formula, re, k, pred.type="response") {
  prediction = data.frame(predictions)
  flexplot:::post_prediction_process_cf.default(model1, model2, prediction, formula, re, k, pred.type="response")
}


# Helper function for null coalescing
`%||%` <- function(x, y) if (is.null(x)) y else x

# Force registration of the method
.onLoad = function(libname, pkgname) {
  # Register S3 methods
  registerS3method("get_model_n", "keras.src.engine.sequential.Sequential", 
                   get_model_n.keras.src.engine.sequential.Sequential)
}


# Estimates Methods -------------------------------------------------------


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
estimates.nn_model = function(object, metric = NULL, return_metrics = TRUE, importance_method = "shap", ...) {
  
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
  
  # Create metrics data frame using validation metrics
  metrics_pred.values = data.frame(
    metric = clean_names,
    value = unlist(val_values),
    stringsAsFactors = FALSE
  )
  rownames(metrics_pred.values) = NULL

  if (importance_method == "shap") {
    
    require(fastshap) 

    shap_values = explain(
      object = object$model,         # your keras model
      X = object$x,               # training data, predictors only
      pred_wrapper = nn_predict,       # your predict function
      nsim = 10                 # fewer = faster, but less stable
    )
    
    shap_df = data.frame(
      value    = round(shap_values[1,], digits=4)) %>%
      arrange(desc(value))
    
    # Return structure
    if (return_metrics) {
      return(list(
        importance = shap_df,
        metrics = metrics_pred.values,
        primary_metric = gsub("^val_", "", metric),
        primary_value = baseline_score
      ))
    } else {
      return(shap_df)
    }
   
    
    
  } else {
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
  }
  
  # Create variable importance data frame
  importance_pred.values = data.frame(
    variable = colnames(x_test),
    importance = importances
  )
  
  # Sort by importance (descending)
  importance_pred.values = importance_pred.values[order(importance_pred.values$importance, decreasing = TRUE), ]
  rownames(importance_pred.values) = NULL
  

  
  # Return structure
  if (return_metrics) {
    return(list(
      importance = importance_pred.values,
      metrics = metrics_pred.values,
      primary_metric = gsub("^val_", "", metric),
      primary_value = baseline_score
    ))
  } else {
    return(importance_pred.values)
  }
}

# A custom imputer that replaces NAs with the column median
median_imputer = function(x) {
  for (col in colnames(x)) {
    if (anyNA(x[[col]])) {
      x[[col]][is.na(x[[col]])] = median(x[[col]], na.rm = TRUE)
    }
  }
  return(x)
}

megashap = function(model, 
                      X, 
                      nsim = 100,
                      predict_function = predict,
                      sample_size = NULL,
                      baseline_data = NULL,
                      verbose = FALSE) {
  feature_names = colnames(X)
  p = length(feature_names)
  n = nrow(X)
  
  if (is.null(sample_size)) sample_size = n
  if (is.null(baseline_data)) {
    baseline_data = X[sample(n, sample_size, replace = TRUE), , drop = FALSE]
  }
  
  # Initialize SHAP value accumulator
  shap_values = matrix(0, nrow = n, ncol = p)
  colnames(shap_values) = feature_names
  
  for (sim in 1:nsim) {
    if (verbose && sim %% 10 == 0) message("Simulation ", sim, " of ", nsim)
    
    # Feature order to apply this iteration
    feature_order = sample(feature_names)
    
    # Bootstrap sample from X
    X_sample = as.data.frame(lapply(X, function(x) sample(x, size = sample_size, replace = TRUE)))

    
    # Loop over features in this order
    base_data = baseline_data[rep(1:n, each = 1), , drop = FALSE]
    preds_prev = matrix(predict_function(model, newdata = base_data), nrow = n)
    
    for (feat in feature_order) {
      # Replace the current feature with actual values from X
      base_data[[feat]] = X[[feat]]
      
      # Get new predictions with that feature replaced
      preds_new = matrix(predict_function(model, newdata = base_data), nrow = n)
      
      # SHAP contribution is the difference
      shap_values[, feat] = shap_values[, feat] + (preds_new - preds_prev)
      
      # Update prediction for next iteration
      preds_prev = preds_new
    }
  }
  
  shap_vector = structure(
    colMeans(abs(shap_values)) / nsim,
    names = colnames(shap_values)
  )
  
  
  # Sort by absolute value of SHAP
  #sorted_shap = shap_vector[order(abs(shap_vector), decreasing = TRUE)]
  shap_vector
  
}

# require(tidyverse)
# X = read.csv("~/Downloads/health_depression.csv") %>% 
#   select(-c(IATTotalscores, HPLPTotalscores))
# 
# # Fit a model
# mod = lm(CESD ~ ., data = X)
# 
# # Run vectorized SHAP bootstrap
# shap_vals = megashap(mod, X, nsim = 50, verbose = TRUE)
# head(shap_vals)

safe_predict = function(object, newdata) {
  preds = predict(object, newdata)
  
  # Force to numeric vector of expected length
  if (length(preds) == 1 && nrow(newdata) > 1) {
    preds = rep(preds, nrow(newdata))
  }
  
  return(as.numeric(preds))
}



# 
# shap_vals = fastshap::explain(
#   mod,
#   X = X,
#   nsim = 50,
#   verbose = TRUE,
#   pred_wrapper = safe_predict
# )
# 
# shap_vals = fastshap::explain(mod, X = X, nsim = 50, verbose = TRUE,
#                               pred_wrapper = function(object, newdata) as.numeric(predict(object, newdata)))

