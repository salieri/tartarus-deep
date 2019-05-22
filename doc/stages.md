


# Stages and Responsibilities


## Create

1.  Validate parameters


## Compile

1.  Declare inputs and outputs
1.  Verify links
    1.  Avoid circular links
    1.  Detect unresolved inputs and outputs
1.  Determine network input and output

    
## Initialize

1.  Assign (share) session
1.  Initialize/recover values

  
## Fit

1.  Train model for given number of epochs using input `X`


## Evaluate

1.  Get loss value and metrics from input `X` and known expected output `Y`


## Predict

1.  Predict output `yHat` from input `X`
