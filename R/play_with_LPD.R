

rm(list=ls())
# Load all relevant packages

inst_pack<-rownames(installed.packages())
if (!"fGarch"%in%inst_pack)
  install.packages("fGarch")
if (!"xts"%in%inst_pack)
  install.packages("xts")

if (!"neuralnet"%in%inst_pack)
  install.packages("neuralnet")


library(neuralnet)
library(fGarch)
library(xts)

source(paste(getwd(),"/R/neuralnet_functions.R",sep=""))
source(paste(getwd(),"/R/data_generating_functions.R",sep=""))

# Load Bitcoin data
path.dat<-paste(getwd(),"/Data/",sep="")
#path.dat<-paste(path.main,"/Exercises/Erste Woche/Data/",sep="")


load(paste(path.dat,"bitcoin.Rdata",sep=""))

#-------------------
head(dat)
tail(dat)
#-------------------
# Plot  data


# plot last, bid and ask in single figure names(dat)
par(mfrow=c(2,2))
plot(dat$Bid,col=1,main="Prices")
plot(log(dat$Bid),col=1,on=1,main="Log-prices")  #tail(dat$Bid)
plot(diff(log(dat$Bid)),col=1,on=1,main="Log-returns")
plot(log(dat$Volume),col=1,on=1,main="Log-volumes")

#----------------------
# Classic linear regression: applied to Bid-prices
# Specify target and explanatory data: we use first six lags based on above data analysis
x<-ret<-na.omit(diff(log(dat$Bid)))
x_level<-log(dat$Bid)
data_mat<-cbind(x,lag(x),lag(x,k=2),lag(x,k=3),lag(x,k=4),lag(x,k=5),lag(x,k=6))
# Check length of time series before na.exclude
dim(data_mat)
data_mat<-na.exclude(data_mat)
# Check length of time series after removal of NAs
dim(data_mat)
head(data_mat)
tail(data_mat)

#--------------------------------------------------------------------
# Specify in- and out-of-sample episodes
in_out_sample_separator<-index(dat)[nrow(dat)]

target_in<-data_mat[paste("/",in_out_sample_separator,sep=""),1]
tail(target_in)
explanatory_in<-data_mat[paste("/",in_out_sample_separator,sep=""),2:ncol(data_mat)]
tail(explanatory_in)

lm_obj<-lm(target_in~explanatory_in)

summary(lm_obj)

train<-cbind(target_in,explanatory_in)

#-------------------------------------------------------------------------------
# Scaling data for neural net

# Scaling data for the NN
maxs <- apply(data_mat, 2, max)
mins <- apply(data_mat, 2, min)
# Transform data into [0,1]
scaled <- scale(data_mat, center = mins, scale = maxs - mins)

apply(scaled,2,min)
apply(scaled,2,max)
#-----------------
# Train-test split
# Train-test split
train_set_xts <- scaled[paste("/",in_out_sample_separator,sep=""),]
test_set_xts <- scaled[paste(in_out_sample_separator,"/",sep=""),]


train_set<-as.matrix(train_set_xts)
test_set<-as.matrix(test_set_xts)
#-----------------------------------

colnames(train_set)<-paste("lag",0:(ncol(train_set)-1),sep="")
n <- colnames(train_set)
# Model: target is current bitcoin, all other variables are explanatory
f <- as.formula(paste("lag0 ~", paste(n[!n %in% "lag0"], collapse = " + ")))

# Set/fix the random seed
set.seed(1)
nn <- neuralnet(f,data=train_set,hidden=c(20,10),linear.output=F)

plot(nn)

###########################################################################################################################
###############################################################################################################
# 1. Bitcoin data: random net vs. optimized net. Use LPD_cost
# 1.1 Random initialized net: use_random_net<-T
#   LPD is approximately equivalent to discrete perturbation(derivative) of input
# 1.2 Optimized net: use_random_net<-F
#   -In general LPD is approximately equivalent to discrete perturbation(derivative) of input
#   -However setting atan_not_sigmoid<-T and linear_output<-F means that exact LPD is much closer to zero
#     and numerical errors occur in discrete LPD so that ratio is different from 1

# Settings
use_random_net<-F
# feedforward net with trhee hideen layers of dimensions 5,4 and 3
neuron_vec<-c(5,4,3)
# Activation function at output
linear_output<-F
# Atan activation function (generally better numerical convergence)
atan_not_sigmoid<-T
LPD_at_single_observation<-F
# Number iterations
epochs<-100
learning_rate<-0.5
#--
if (linear_output)
  learning_rate<-learning_rate/10


# Data
if (LPD_at_single_observation)
{
  x_train<-t(as.matrix(train_set[nrow(train_set),2:ncol(train_set)],nrow=1,ncol=6))
  y_train<-as.matrix(train_set[nrow(train_set),1])
} else
{
  x_train<-as.matrix(train_set[,2:ncol(train_set)])
  y<-y_train<-as.matrix(train_set[,1],ncol=1)
}


# Net
list_layer_size<-layer_size<-getLayerSize(x_train, y_train, neuron_vec)
list_layer_size
set.seed(1)

# Random initialization or optimal (fitted) net
if (use_random_net)
{
  parm <- initializeParameters(list_layer_size)
} else
{
  hyper_list<-list(epochs=epochs,learning_rate=learning_rate,linear_output=linear_output,atan_not_sigmoid=atan_not_sigmoid,neuron_vec=neuron_vec)

  setseed<-1

  set.seed(setseed)

  # Train model

  train_model <- trainModel(x_train, y_train, hyper_list=hyper_list)

  ts.plot(train_model$cost_hist)

  train_model$cost_hist[length(train_model$cost_hist)]

  parm<-train_model$updated_params

}

# Generate output of net
fwd_prop <- forwardPropagation(x_train, parm, layer_size,linear_output,atan_not_sigmoid)

cache<-fwd_prop
# Compute MSE
cost <- computeCost(y_train, fwd_prop)
# MSE
cost

# Perturbate input of k-th expalantory variable
k<-3
delta<-0.000001
x_train_modified<-x_train
x_train_modified[,k]<-x_train[,k]+delta

fwd_prop_modified <- forwardPropagation(x_train_modified, parm, layer_size,linear_output,atan_not_sigmoid)

cache_modified<-fwd_prop_modified

cost_modified <- computeCost(y_train, fwd_prop_modified)
cost_modified

# Discrete LPD
LPD_discret<-(cost_modified-cost)/delta

# Exact LPD
LPD_cost_exact<-LPD_cost(y_train,cache, parm, list_layer_size,linear_output,atan_not_sigmoid)$LPD_cost

# Should be one: if it is substantially different then select smaller delta>0 above...
LPD_discret/LPD_cost_exact[k]



##########################################################################################################
# 2. Linear Artificial data: LPD_cost (gradient of output)
# 2.1 Some experiments checking backpropagation, LPD and LPD_cost by comparing exact and discrete derivatives
#   -Backpropagation: gradient of MSE with respect to parameters (weights/biases)
#   -LPD_cost: gradient of MSE with respct to input data x
#   -LPD: gradient of output with respect to input data x
# Net architecture
#   With more than one hidden layer convergence is difficult
#   With a single hidden layer performance is comparable to linear regression
# Random vs. optimal net
#   LPD exact is the same as discrete LPD for random nets: this is because forecast error is not orthogonal to net output
#   For optimized net
#     -the discrete LPD on whole data-set (all rows) is heavily dependent on delta!!! This is because gradient~0
#     -on single row (single time point) exact and discrete LPD match: here abs(gradient)>0
#   Shows that discrete LPD is a bad choice when applied to all rows for optimized net!!!!!!!!!!!!!!!!!!!!
# Publication: discrete LPS can be bad for optimized nets

# Settings
use_random_net<-F
neuron_vec<-c(10,2)
neuron_vec<-c(5,4,3)
neuron_vec<-c(3)
linear_output<-F
atan_not_sigmoid<-F
LPD_at_single_observation<-F
epochs<-2000
learning_rate<-1
#--
if (linear_output)
  learning_rate<-learning_rate/10
if (!atan_not_sigmoid)
  learning_rate<-learning_rate*5


# Generate artificial data
setseed<-1
set.seed(setseed)
len<-1000
x_train<-cbind(rnorm(len),rnorm(len))
w1<-0.5
w2<--0.7
y_train<-w1*x_train[,1]+w2*x_train[,2]+rnorm(len)

x_train <- scale(x_train, center = apply(x_train,2,min), scale = apply(x_train,2,max) - apply(x_train,2,min))
y_train <- scale(y_train, center = min(y_train), scale = max(y_train) - min(y_train))

lm_obj<-lm(y_train~x_train)
summary(lm_obj)
mean(lm_obj$res^2)

# Net
list_layer_size<-layer_size<-getLayerSize(x_train, y_train, neuron_vec)
list_layer_size
set.seed(1)

# Random initialization or optimal net
if (use_random_net)
{
  parm <- initializeParameters(list_layer_size)
} else
{
  hyper_list<-list(epochs=epochs,learning_rate=learning_rate,linear_output=linear_output,atan_not_sigmoid=atan_not_sigmoid,neuron_vec=neuron_vec)

  setseed<-1

  set.seed(setseed)

  # Train model

  train_model <- trainModel(x_train, y_train, hyper_list=hyper_list)

  ts.plot(train_model$cost_hist)

  train_model$cost_hist[length(train_model$cost_hist)]

  parm<-train_model$updated_params

}

#------------
# Checks
# 1. Check gradient for backpropagation: derivative of MSE with respect to parameters
# Use random net for this because discrete estimate is poor for optimum!!!!!!!!!!!!!!!!!!!!!!!
# Discrete gradient is heavily dependent on delta for optimum!!!!!!!!!!!!!!
fwd_prop <- forwardPropagation(x_train, parm, layer_size,linear_output,atan_not_sigmoid)
cache<-fwd_prop
cost <- computeCost(y_train, fwd_prop)
cost
delta<-0.00001
parm_modified<-parm
layer<-1
j<-1
k<-1
parm_modified$W_list[[layer]][j,k]<-parm$W_list[[layer]][j,k]+delta
fwd_prop_modified <- forwardPropagation(x_train, parm_modified, layer_size,linear_output,atan_not_sigmoid)
cost_modified <- computeCost(y_train, fwd_prop_modified)
(cost_modified-cost)/delta
grads<-backwardPropagation(x_train, y_train, cache, parm, list_layer_size,linear_output,atan_not_sigmoid)
grads$dW_list[[layer]][j,k]/((cost_modified-cost)/delta)

#-------------------------
# 2. Check LPD_cost: derivative of MSE with respect to full input (all data points): problem when using optimized net because residual orthogonal to net output
# Perturbate input full data (all rows)
# Here discrete and exact LPDs do not always match for optimized net (but they match for random net)
# Note that LPD_cost applied to full data is mean(2*residual*gradient output)
#   If fit is good, then residual is independent of gradient
#   Therefore LPD_cost~0 (for full data-set)
#   And therefore LPD_discret/LPD_cost can differ from one for optimized net
k<-1
# Try any of the following (if delta is too large, then discrete proxy is poor)
delta<-0.000001
delta<-0.001
delta<-0.00015
x_train_modified<-x_train
x_train_modified[,k]<-x_train[,k]+delta
fwd_prop <- forwardPropagation(x_train, parm, layer_size,linear_output,atan_not_sigmoid)
cache<-fwd_prop
cost <- computeCost(y_train, fwd_prop)
cost
fwd_prop_modified <- forwardPropagation(x_train_modified, parm, layer_size,linear_output,atan_not_sigmoid)
cache_modified<-fwd_prop_modified
cost_modified <- computeCost(y_train, fwd_prop_modified)
cost_modified
# Discrete LPD
LPD_discret<-(cost_modified-cost)/delta
# Exact LPD
LPD_cost_exact<-LPD_cost( y_train,cache, parm, list_layer_size,linear_output,atan_not_sigmoid)$LPD_cost
# LPD_cost_exact should be small/vanishing for optimized net
LPD_cost_exact
# This ratio differs from one for optimized net!!!!!!!!!
#  For optimized net both values are close to zero...
# Select smaller delta to get aprox. one
LPD_discret/LPD_cost_exact[k]


#-------------------------
# 3. Check LPD_cost: derivative of MSE with respect to LAST DATA POINT
# Perturbate input single (last) time point
# Here discrete and exact LPDs match even for optimized net
#   Note that LPD_cost generally differs from zero at any single observation (even for optimized net)
#   Therefore the above singularity problem (vanishing LPD_cost_exact) does not apply here
x<-t(matrix(x_train[nrow(x_train),]))
y<-matrix(y_train[nrow(y_train)])
fwd_prop <- forwardPropagation(x, parm, layer_size,linear_output,atan_not_sigmoid)
cache<-fwd_prop
cost <- computeCost(y, fwd_prop)
cost
LPD_cost_exact<-LPD_cost(y, cache, parm, list_layer_size,linear_output,atan_not_sigmoid)$LPD_cost
intercept<-cache$Z_list[length(neuron_vec)+1][[1]]-(x%*%LPD_cost_exact)
k<-1
delta<-0.000001
delta<-0.001
delta<-0.00015
x_modified<-x
x_modified[,k]<-x[,k]+delta
fwd_prop_modified <- forwardPropagation(x_modified, parm, layer_size,linear_output,atan_not_sigmoid)
cache_modified<-fwd_prop_modified
cost_modified <- computeCost(y, fwd_prop_modified)
cost_modified
# Discrete LPD
LPD_discret<-(cost_modified-cost)/delta
# Both match because neither one is vanishing at last data point (they are vanishing in the mean over time)
LPD_discret/LPD_cost_exact[k]

#-------------------------
# 4. Check LPD: derivative of output (not MSE) with respect to input
# Perturbate input full data (all rows)
k<-1
delta<-0.000001
delta<-0.001
delta<-0.00015
x_train_modified<-x_train
x_train_modified[,k]<-x_train[,k]+delta
fwd_prop <- forwardPropagation(x_train, parm, layer_size,linear_output,atan_not_sigmoid)
cache<-fwd_prop
fwd_prop_modified <- forwardPropagation(x_train_modified, parm, layer_size,linear_output,atan_not_sigmoid)
# Discrete LPD
LPD_obj<-LPD( cache, parm, list_layer_size,linear_output,atan_not_sigmoid)
LPD_t<-LPD_obj$LPD_t
LPD_agg<-LPD_obj$LPD
j<-202
# At one single time point: matches even for optimized net
LPD_t[j,k]/((fwd_prop_modified$A_list[[length(fwd_prop$A_list)]]-fwd_prop$A_list[[length(fwd_prop$A_list)]])[j]/delta)
# Aggregate over all time points
LPD_agg[k]/mean((fwd_prop_modified$A_list[[length(fwd_prop$A_list)]]-fwd_prop$A_list[[length(fwd_prop$A_list)]])/delta)



#############################################################################################################
#############################################################################################################
# 3. Linear artificial data. Here we play with  LPD (not LPD_cost): replicate linear model and inferences i.e. std of parameters
# LPD leads to linear regression parameters;
#   LPD_cost (MSE=cost) identifies ratios of parameters (not the absolute values of params)
# Inference hypothesis testing
# Use generic data generating function

# Settings
use_random_net<-F
neuron_vec<-c(10,2)
neuron_vec<-c(3)
linear_output<-T
atan_not_sigmoid<-F
LPD_at_single_observation<-F
epochs<-10000
learning_rate<-0.5

#--
if (linear_output)
  learning_rate<-learning_rate/10
if (!atan_not_sigmoid)
  learning_rate<-learning_rate*5


# Data
setseed<-1
set.seed(setseed)
len<-1000
w_vec<-c(0.5,-0.7,1.9)
sigma<-1
weight_common_factor<-c(1,2,3)

data<-generate_data_func(w_vec,sigma,weight_common_factor,len)

x_train<-data$x
y_train<-data$y

lm_obj<-lm(y_train~x_train)
summary(lm_obj)
mean(lm_obj$res^2)


#---------------------
# Net
list_layer_size<-layer_size<-getLayerSize(x_train, y_train, neuron_vec)
list_layer_size
set.seed(1)
# Random initialization or optimal net
if (use_random_net)
{
  parm <- initializeParameters(list_layer_size)
} else
{
  hyper_list<-list(epochs=epochs,learning_rate=learning_rate,linear_output=linear_output,atan_not_sigmoid=atan_not_sigmoid,neuron_vec=neuron_vec)
  setseed<-1
  set.seed(setseed)
  # Train model
  train_model <- trainModel(x_train, y_train, hyper_list=hyper_list)
  ts.plot(train_model$cost_hist)
  train_model$cost_hist[length(train_model$cost_hist)]
  parm<-train_model$updated_params
}

# Check gradient at optimum: should vanish
if (F)
{
  fwd_prop <- forwardPropagation(x_train, parm, layer_size,linear_output,atan_not_sigmoid)
  cache<-fwd_prop
  backwardPropagation(x_train, y_train, cache, parm, list_layer_size,linear_output,atan_not_sigmoid)
}

# Check LPD


#-------------------
# Generate LPD for all time-points
LPD_mat<-NULL
for (i in 1:nrow(x_train))# i<-1
{
  x<-t(matrix(x_train[i,]))
#  x<-t(matrix(rep(1,ncol(x_train))))

  y<-matrix(y_train[i])
#  y<-matrix(1)
  fwd_prop <- forwardPropagation(x, parm, layer_size,linear_output,atan_not_sigmoid)
  cache<-fwd_prop
  LPD_exact<-LPD(cache, parm, list_layer_size,linear_output,atan_not_sigmoid)$LPD
# This is the same as returning LPD_t (because we use only one data-point)
  LPD_exact<-as.double(LPD(cache, parm, list_layer_size,linear_output,atan_not_sigmoid)$LPD_t)
  LPD_exact<-LPD_exact
# Here we compute the local intercept
  intercept<-cache$Z_list[length(neuron_vec)+1][[1]]-(x%*%LPD_exact)
  LPD_mat<-rbind(LPD_mat,c(intercept,LPD_exact))
}
colnames(LPD_mat)<-c("intercept",paste("w",1:ncol(x),sep=""))

#--------------------
# Plot LPD
plot.new()
ts.plot(LPD_mat,col=rainbow(ncol(x_train)+1))
abline(h=0)
apply(LPD_mat,2,mean)
# Comparison with linear regression
k<-len
j<-4
LPD_mat[,j]/lm_obj$coefficients[j]
#--------------------------
# Standarddeviation of LPD
#   This is not the ordinary standard deviation of coefficients in lm
#   Rather it is a measure for the stability/linearity of the net with respect to the corresponding explanatory
#   It assumes a single realization of the data; in contrast usual t-values assume multiple realizations
#   i.e. take the standard deviation over time for a single realization vs. take standard deviation for all time points over multiple relaizations
# A large standard deviation suggests instability of the net with respect to the explanatory
#   -Could be an identification problem; or overfitting; or non-stationary data
sd_LPD<-apply(LPD_mat,2,sd)
summary(lm_obj)
# T-value
LPD_mat[k,]/sd_LPD


#---------------------------
# Out of sample performance
len<-1000
set.seed(setseed+1)
data<-generate_data_func(w_vec,sigma,weight_common_factor,len)
x_test<-data$x
y_test<-data$y
fwd_prop<- forwardPropagation(x_test, parm, layer_size,linear_output,atan_not_sigmoid)
out<-fwd_prop$A_list[length(neuron_vec)+1][[1]]
# Neural net
mean((as.vector(y_test)-out)^2)
# Linear regression
mean((cbind(rep(1,nrow(x_test)),x_test)%*%lm_obj$coef-as.vector(y_test))^2)
# neuralnet package
nn <- neuralnet(y_train~. ,data=x_train,hidden=neuron_vec,linear.output=F)
plot(nn)
mean((y_test-predict(nn,x_test))^2)

#----------------------------
# Residual orthogonal to explanatories?
#   Yes for optimized net because the net is almost linear.
#     Gradient of MSE is residual*derivative of net output with respect to theta
#     derivative of net output is approximately x_i if net is linear
x<-x_train
y<-y_train
#  y<-matrix(1)
fwd_prop <- forwardPropagation(x, parm, layer_size,linear_output,atan_not_sigmoid)
cache<-fwd_prop

nn_output<-as.double(cache$A_list[[length(layer_size)-1]])
for (k in 1:layer_size$n_x)
{
  print(mean((y-nn_output)*x[,k]))
}


#############################################################################################################
#################################################################################################################
# 4. Non-linear net fitting non-stationary time-dependent AR(1)
#   Net is stationary in the sense that the function is deterministic: same input at different time points generates same output
#   Sinusoidal AR(1) means that process is non-stationary
# Question: can non-linearity adapt to non-stationarity and outperform linear model systematically out-of-sample?
# Outcome:
#   nlminb-optimization outperforms clearly plain backpropagation: set use_nlminb<-T
#   lags<-1: neural net pretty close to linear regression unable to learn pattern: memory too short
#   lags<-2: neural net estimated with nlminb can extract some non-stationarity:
#     out-of-sample perf midway between regression and best/true non-stationary model
#   Comparison of LPD and true ar1 illustrates that net follows periodic pattern of AR(1)
# One can try various freq settings: frequency of AR(1)-sinusoidal (slow/fast changing AR(1))
# linear_output=T or F: both extract non-stationarity
# atan activation possibly performs better in the sense that optimization seems easier (smaller in-sample MSE and out-sample MSE)

# General settings
use_random_net<-F
neuron_vec<-c(10,5)
neuron_vec<-c(5)
linear_output<-T
atan_not_sigmoid<-T
LPD_at_single_observation<-F
epochs<-10000
learning_rate<-0.1
# Number of lags in input layer: build up memory in feedforward net!!!!
#   If lags is 1 then non-linearity can possibly not be detected by net: net is similar to linear regression
#   If lags is 2 then non-linearity can be better detected because moving from positive acf to negative acf affects dependence structure more heavily
lags<-2

#--
if (linear_output)
  learning_rate<-learning_rate/10
if (!atan_not_sigmoid)
  learning_rate<-learning_rate*5

setseed<-1
set.seed(setseed)
len<-1000
# Frequency of AR(1) coefficient
freq<-5
# Generate (non-linear) data: AR(1) with time-dependent AR(1)-coefficient
data<-generate_data_non_linear_ar1_func(len,lags,freq)

x_train<-data$x_train
y_train<-data$y_train
ar1<-data$ar1
# Time dependent ar1-parameter
ts.plot(ar1)

 # Apply (false) linear model
lm_obj<-lm(y_train~x_train)
summary(lm_obj)
mean(lm_obj$res^2)

#---------------------
# Net
list_layer_size<-layer_size<-getLayerSize(x_train, y_train, neuron_vec)
list_layer_size
setseed<-13
# nlminb performs slightly better than standard backpropagation for this problem: estimated net equals best neuralnet
use_nlminb<-T
# Random initialization or optimal net
if (use_random_net)
{
  parm <- initializeParameters(list_layer_size)
} else
{
  if (use_nlminb)
  {
    layer_size<-getLayerSize(x_train, y_train, neuron_vec)

# Compute the total number of parameters: we do not distinguish weights/biases or layers here
# nlminb just takes a vector of 'unordered' parameters
    parm_len<-compute_number_parameters(layer_size)
# Initialize parameters
    set.seed(setseed)
    obj_crit<-10^90
# Proceed to 10 estmation loops and select best net
    for (i in 1:10)
    {
      parm<-parm_init<-rnorm(parm_len)/sqrt(parm_len)
      # No ordering according to layers, weights and biases are not properly identified
      parm

      x<-x_train
      y<-y_train
      # Numerical optimization in R


      nlmin_obj<-nlminb(parm_init,optimize_nlminb_net,x=x,y=y,layer_size=layer_size,linear_output=linear_output,atan_not_sigmoid=atan_not_sigmoid)

      print(nlmin_obj$objective)
      if (nlmin_obj$objective<obj_crit)
      {
        parm_opt<-nlmin_obj$par
        obj_crit<-nlmin_obj$objective
      }

    }
    print(obj_crit)
# Transform parameters back into weights and biases and arrange into layers
    parm<-translate_Parameters(list_layer_size,parm_opt)


  } else
  {
    hyper_list<-list(epochs=epochs,learning_rate=learning_rate,linear_output=linear_output,atan_not_sigmoid=atan_not_sigmoid,neuron_vec=neuron_vec)
    set.seed(setseed)
    # Train model
    train_model <- trainModel(x_train, y_train, hyper_list=hyper_list)
    ts.plot(train_model$cost_hist)
    train_model$cost_hist[length(train_model$cost_hist)]
    parm<-train_model$updated_params
  # Second round based on parm as initialization
    if (F)
    {
      parm_init<-parm
      hyper_list<-list(epochs=epochs,learning_rate=learning_rate,linear_output=linear_output,atan_not_sigmoid=atan_not_sigmoid,neuron_vec=neuron_vec,parm_init=parm_init)
      train_model <- trainModel(x_train, y_train, hyper_list=hyper_list)
      ts.plot(train_model$cost_hist)
      train_model$cost_hist[length(train_model$cost_hist)]
      parm<-train_model$updated_params

    }
  }
}



# Check gradient at optimum: should vanish
if (F)
{
  fwd_prop <- forwardPropagation(x_train, parm, layer_size,linear_output,atan_not_sigmoid)
  cache<-fwd_prop
  backwardPropagation(x_train, y_train, cache, parm, list_layer_size,linear_output,atan_not_sigmoid)
}

#-------------------
# Generate LPD for all time-points
LPD_mat<-NULL
for (i in 1:nrow(x_train))# i<-1
{
  x<-t(matrix(x_train[i,]))
  #  x<-t(matrix(rep(1,ncol(x_train))))

  y<-matrix(y_train[i])
  #  y<-matrix(1)
  fwd_prop <- forwardPropagation(x, parm, layer_size,linear_output,atan_not_sigmoid)
  cache<-fwd_prop
  LPD_obj<-LPD(cache, parm, list_layer_size,linear_output,atan_not_sigmoid)
  LPD_exact<-LPD_obj$LPD
#  LPD_t<-LPD_obj$LPD_t
  intercept<-cache$Z_list[length(neuron_vec)+1][[1]]-(x%*%LPD_exact)
  LPD_mat<-rbind(LPD_mat,c(intercept,LPD_exact))
}
colnames(LPD_mat)<-c("intercept",paste("w",1:ncol(x),sep=""))
#--------------------
# Plot LPD and compare with true time-dependent ar1
ts.plot(LPD_mat,col=rainbow(ncol(x_train)+1))
abline(h=0)
lines(ar1,lwd=3)
apply(LPD_mat,2,mean)

#---------------------------
# Out of sample performance
len<-1000
setseed<-1
# Add one to the setseed
set.seed(setseed+1)
len<-1000
freq<-5
data<-generate_data_non_linear_ar1_func(len,lags,freq)
x_test<-data$x
y_test<-data$y
fwd_prop<- forwardPropagation(x_test, parm, layer_size,linear_output,atan_not_sigmoid)
out<-fwd_prop$A_list[length(neuron_vec)+1][[1]]
# Neural net
mean((as.vector(y_test)-out)^2)
# Linear regression
mean((cbind(rep(1,nrow(x_test)),x_test)%*%lm_obj$coef-as.vector(y_test))^2)
# True/best model performance
# 1. Compute sCaling used for transforming y_test linearly into [0,1] interval:
scaling<-attributes(y_test)$`scaled:scale`
# Since unscaled data has one-step ahead MSE of 1, the true or best model has MSE of 1/scaling^2
1/scaling^2
# neuralnet package
nn <- neuralnet(y_train~. ,data=x_train,hidden=neuron_vec,linear.output=F)
plot(nn)
mean((y_test-predict(nn,x_test))^2)

#--------------------------
# Residual orthogonal to explanatories?
#   Yes for optimized net if lag=1 because then net is almost linear.
#   For lag<-10 net becomes non-linear and residual is nomore orthogonal
#     Gradient of MSE is residual*derivative of net output with respect to theta
#     derivative of net output is approximately x_i if net is linear
x<-x_train
y<-y_train
#  y<-matrix(1)
fwd_prop <- forwardPropagation(x, parm, layer_size,linear_output,atan_not_sigmoid)
cache<-fwd_prop
nn_output<-as.double(cache$A_list[[length(layer_size)-1]])
for (k in 1:layer_size$n_x)
{
  print(mean((y-nn_output)*x[,k]))
}
