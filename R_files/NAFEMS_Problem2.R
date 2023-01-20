# Load the pba.r package:
source("C:\\Users\\user\\Downloads\\pba.r")

# Define the number of steps in plotting the P-box:
Pbox$steps = 500

# Define performance function divided by L:
func1 = function(neps,nD_p) return((150*mu*(1+neps)^2*v_s/(nD_p^2)+1.75*rho*(1+neps)*v_s^2/(-nD_p))/(-neps)^3)

# Write new P-box function to account for monotonically decreasing nature of the performance function:
phi = function(x,y) {
x <- makepbox(x)
y <- makepbox(y)
zu <- zd <- rep(0,Pbox$steps)

for (i in 1:Pbox$steps)

{
j <- i:Pbox$steps
k <- Pbox$steps:i

zd[[i]]  <- min(func1(x@d[j],y@d[k]))       
j <- 1:i
k <- i:1
zu[[i]]  <- max(func1(x@u[j],y@u[k]))       
}

# mean:
ml <- -Inf    # might be improved
mh <- Inf     # might be improved

# variance:
vl <- 0       # might be improved
vh <- Inf     # might be improved

pbox(u=zu + 0, d=zd + 0, ml = ml, mh = mh, vl=vl, vh=vh, dids=paste(x@dids,y@dids))
}

# Define the parameters:
rho = 1.225     # kg per m{3}   // density of air 
mu = 1.81e-5    # kg over m s   // dynamic viscosity of air
v_s = 0.35      # m per s       // non-negative superficial velocity of fluid

# Define the aleatory parameters:
D_p = N(interval(0.0035, 0.0038), interval(0.1184e-03, 0.4634e-03))
eps = N(interval(0.3604, 0.3866), interval(0.0169, 0.0424))
L = 5*beta(interval(112.1297,  124.9070), interval(75.1162,   83.7704))
delta_p = L %*% phi(-eps, -D_p)
delta_p

# Plot the P-box for delta_p:
lines(delta_p, col='blue')
 
# Compute failure probability bounds:
delta_p > 15250 # Pa     

# Sensitivity Analysis:
fail0 = (delta_p > 15250)
delt0 = right(fail0) - left(fail0)

D_p = N(interval(0.0035, 0.0038), interval(0.1184e-03, 0.4634e-03))
eps = N(interval(0.3604, 0.3866), interval(0.0169, 0.0424))
L = 5*beta(interval(112.1297,  124.9070), interval(75.1162,   83.7704))
delta_p = L %*% phi(-eps, -D_p)
fail_pinched = (delta_p > 15250)
delt_pinched = right(fail_pinched) - left(fail_pinched) # Pinched interval
S = 1 - (delt_pinched/delt0) # Sensitivity index
S