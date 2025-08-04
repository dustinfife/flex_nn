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
  if (!is.null(categorical_vars)) {
    for (var in categorical_vars) {
      if (var %in% names(data)) {
        # Basic one-hot encoding - you might want to use a more sophisticated approach
        data[[var]] <- as.numeric(as.factor(data[[var]])) - 1
      }
    }
  }
  
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
