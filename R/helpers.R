#' Convert Data for Keras Prediction
#'
#' Converts a data frame to the matrix format expected by Keras models,
#' handling categorical variables appropriately.
#'
#' @param data Data frame to convert
#' @param categorical_vars Character vector of categorical variable names
#' @param encoding_info List containing encoding information for categorical variables
#' @return Matrix suitable for Keras prediction
#' @export
prepare_keras_data <- function(data, categorical_vars = NULL, encoding_info = NULL) {
  
  # Handle categorical variables
  if (is.null(categorical_vars)) return(as.matrix(data))
  
  existing_cats = intersect(categorical_vars, names(data))
  if (length(existing_cats) == 0) return(as.matrix(data))
  
  data[existing_cats] = lapply(data[existing_cats], \(x) as.numeric(as.factor(x)) - 1)
  
  # Convert to matrix
  as.matrix(data)
}

#' Set Response Variable for Keras Model
#'
#' Helper function to store the response variable name as a model attribute.
#' This is useful for the get_terms method.
#'
#' @param model Keras model object
#' @param response_var Name of the response variable
#' @return The model object with response variable attribute set
#' @export
set_response_var <- function(model, response_var) {
  attr(model, "response_var") <- response_var
  return(model)
}


#' Convert response variable to appropriate format for neural networks
#'
#' @param y Response variable from model.frame
#' @return Numeric vector suitable for neural network training
convert_response_variable = function(y) {
  
  if (is.factor(y)) {
    if (nlevels(y) > 2) {
      stop("Multi-class classification not currently supported. Please convert to binary (0/1) or use regression.")
    }
    message("Converting binary factor to 0/1 encoding for binary classification")
    return(as.numeric(y) - 1)
  }
  
  if (is.character(y)) {
    unique_vals = unique(y)
    if (length(unique_vals) > 2) {
      stop("Multi-class classification not currently supported. Please convert to binary (0/1) or use regression.")
    }
    message("Converting binary character to 0/1 encoding for binary classification")
    return(as.numeric(as.factor(y)) - 1)
  }
  
  if (is.numeric(y)) {
    unique_vals = unique(y)
    if (length(unique_vals) == 2 && all(unique_vals %in% c(0, 1))) {
      message("Detected binary classification (0/1 encoding)")
      return(y)
    }
    if (length(unique_vals) == 2) {
      message("Converting binary numeric to 0/1 encoding")
      return(as.numeric(as.factor(y)) - 1)
    }
    # Regression case - return as-is
    return(y)
  }
  
  stop("Unsupported response variable type. Please use numeric, factor, or character.")
}