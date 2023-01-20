# Load the pba.r package:
source("C:\\Users\\user\\Downloads\\pba.r")

# Define the number of steps in plotting the P-box:
Pbox$steps = 500

# Define the aleatory parameters:
R = N(interval(471.7296, 482.5105), interval(8.0288, 23.0162))
S = N(interval(342.1467, 368.6816), interval(23.6986, 56.7473))

# Compute the Performance function g:
gf = R %-% S # MPa (Assuming uncertain correlations)
gf
gi = R %|-|% S # MPa (Assuming independence)
gi

# Plot the P-boxes for gf and gi:
lines(gf, col='blue')
lines(gi, col='green')

# Compute the failure probabilities:
gf < 0 # Failure probability under uncertain correlations
gi < 0 # Failure probability under independence assumptions