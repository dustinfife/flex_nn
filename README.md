# flex_nn

Neural Network Extensions for Flexplot

## Overview

`flex_nn` extends the functionality of the `flexplot` package to support neural network models, particularly those built with Keras/TensorFlow. It provides seamless integration with flexplot's `compare.fits()` function for visualizing neural network predictions.

## Installation

```r
# Install from GitHub (once uploaded)
# devtools::install_github("yourusername/flex_nn")

# For now, install locally
devtools::install("path/to/flex_nn")
```

## Usage

```r
library(flexplot)
library(flex_nn)
library(keras)

# Train your neural network
model <- keras_model_sequential() %>%
  layer_dense(units = 10, activation = "relu", input_shape = 3) %>%
  layer_dense(units = 1, activation = "linear")

# Compile and fit your model
model %>% compile(optimizer = "adam", loss = "mse")
# ... fit your model ...

# Set the response variable name (helpful for plotting)
model <- set_response_var(model, "outcome_variable")

# Use with flexplot's compare.fits
compare.fits(outcome ~ predictor1 | predictor2, 
             data = your_data, 
             model1 = your_lm_model, 
             model2 = model)
```

## Features

- S3 methods for Keras models that integrate with flexplot
- Automatic handling of neural network predictions
- Support for both regression and classification models
- Helper functions for data preparation

## Requirements

- R >= 3.5.0
- flexplot
- keras
- tensorflow

