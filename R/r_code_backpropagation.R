### R code from vignette source 'C:/wia_desktop/2020/teaching_unterricht/CAS BIDA 2021/Vierter Halbtag Backpropagation and XAI Opt/Rnw/slides_backpropagation'
rm(list=ls())
###################################################
### code chunk number 1: init
###################################################
# Load all relevant packages

inst_pack<-rownames(installed.packages())

if (!"neuralnet"%in%inst_pack)
  install.packages("neuralnet")


library(neuralnet)


len<-100
set.seed(1)
x<-rnorm(len)
anz_var<-10
for (i in 1:anz_var)
  x<-cbind(x,rnorm(len))
y<-x[,1]+rnorm(len)

nn <- neuralnet(y~. ,data=x,hidden=c(10,5),linear.output=F)



file = "nn.pdf"
plot(nn,rep="best")





###################################################
### code chunk number 3: init
###################################################

set.seed(1)
x<-rnorm(len)
y<-x+rnorm(len)

set.seed(100)

nn <- neuralnet(y~. ,data=x,hidden=1,linear.output=F)

#plot.new()
file = "nn_toy.pdf"
plot(nn,rep="best")



###################################################
### code chunk number 5: init
###################################################
# R-code: initialize
sigmoid <- function(x){
  return(1 / (1 + exp(-x)))
}

# We here replicate neuralnet package
# These are the optimal weights as optimized by neuralnet
b1<-nn$weights[[1]][[1]][1]
w1<-nn$weights[[1]][[1]][2]
b2<-nn$weights[[1]][[2]][1]
w2<-nn$weights[[1]][[2]][2]
b1
w1
b2
w2
parm_opt_neuralnet<-c(b1,b2,w1,w2)
names(parm_opt_neuralnet)<-c("b1","b2","w1","w2")
parm_opt_neuralnet
# Net output: chained non-linear and linear functions: complexity!!!
# Backcasting has to do with chain-rule of differentiation
net_output<-function(b1,b2,w1,w2,x)
{
  output<-sigmoid(b2+w2*sigmoid(b1+w1*x))
  return(output)
}

# Cost function: mean-square forecast error: should be minimized as a function of b1,w1,b2,w2
mse_func<-function(y,output)
{
  return(mean((y-output)^2))
}


output<-net_output(b1,b2,w1,w2,x)

tail(output)


# Performance of optimized (neuralnet-package) net: this is a ranom-net (optimized parameters depend on set.seed)
MSE_neuralnet_optimal<-mse_func(y,output)

MSE_neuralnet_optimal
# This one matches output of neural net...
len*MSE_neuralnet_optimal/2
#-------------------------------------------------
# Arbitrary initialization
# Initialize all parameters with 0.1
b1<-b2<-w1<-w2<-0.1

output<-net_output(b1,b2,w1,w2,x)

# Performance of random net with noisy initialization
MSE_init<-mse_func(y,output)

# This larger than MSE_neuralnet_optimal of optimized net: optimized net outperforms our 'arbitrary' intialized net, as expected
MSE_init


###################################################
### code chunk number 6: init
###################################################
# R-code: discrete step
# Let's increase w1 a bit: try different values of delta
delta<-0.0001
w1_new<-w1+delta

output_modified<-net_output(b1,b2,w1_new,w2,x)

# Performance of random net with modified w1
MSE_modified<-mse_func(y,output_modified)

# MSE modified has decreased marginally: better i.e.  w1_modified is better than initial w1
MSE_modified-MSE_init
# Slope or approximate derivative: increase of MSE divided by delta (increment of w1)
dw1<-(MSE_modified-MSE_init)/delta
dw1



###################################################
### code chunk number 7: init
###################################################
# R-code: discrete gradient

parm_init<-c(b1,b2,w1,w2)


# This is a discrete proxy of gradient: perturbation applied to all parameters
gradient_func<-function(parm,x,y,delta)
{

  parm_modified<-parm
  b1<-parm[1]
  b2<-parm[2]
  w1<-parm[3]
  w2<-parm[4]
  output<-net_output(b1,b2,w1,w2,x)
  MSE<-mse_func(y,output)

  gradient<-rep(NA,length(parm))
  for (i in 1:length(parm))
  {
    parm_modified<-parm
    parm_modified[i]<-parm[i]+delta
    b1<-parm_modified[1]
    b2<-parm_modified[2]
    w1<-parm_modified[3]
    w2<-parm_modified[4]
    output_modified<-net_output(b1,b2,w1,w2,x)
    MSE_modified<-mse_func(y,output_modified)
    gradient[i]<-(MSE_modified-MSE)/delta
  }
  names(gradient)<-c("b1","b2","w1","w2")
  return(gradient)
}

gradient<-gradient_func(parm_init,x,y,delta)
gradient
# Check with above result for w1
(MSE_modified-MSE_init)/delta



# up-date the parameters: go in negative direction of gradient; scale with learn-rate
# Large learn-rate: big up-dating step (learn rapidly and 'shoot over target')
# Small learn-rate: small up-dating step (learn progressively/slowly/not at all...)
learn_rate<-10
learn_rate<-0.1
learn_rate<-0.01
parm_up<-parm_init-learn_rate*gradient

b1<-parm_up[1]
b2<-parm_up[2]
w1<-parm_up[3]
w2<-parm_up[4]

output_up<-net_output(b1,b2,w1,w2,x)

# Performance of random net with up dated parameter vector
MSE_up<-mse_func(y,output_up)

# Improvement: depends on learn-rate
MSE_up-MSE_init

# This function accepts the parameter and the data and returns MSE
# Can be used for numerical optimization
optimize_toy_net<-function(parm,x,y)
{
  b1<-parm[1]
  b2<-parm[2]
  w1<-parm[3]
  w2<-parm[4]

  output<-net_output(b1,b2,w1,w2,x)

  MSE<-mse_func(y,output)

  return(MSE)
}

nlmin_obj<-nlminb(parm_init,optimize_toy_net,x=x,y=y)

par_optim<-nlmin_obj$par
names(par_optim)<-c("b1","b2","w1","w2")
par_optim
# Result from optimization
opt_mse<-nlmin_obj$objective
opt_mse
# Confirmation of forecast MSE-performance of optimized parameter
optimize_toy_net(par_optim,x,y)
# Compare with neuralnet: not as good as our own...
MSE_neuralnet_optimal
# Compare estimates....: crazy different: XAI (want to understand model)
parm_opt_neuralnet
par_optim

# The gradient should vanish at optimum: check
#   But this is not the exact gradient!!! See below
gradient_func(par_optim,x,y,delta)


###################################################
### code chunk number 10: init
###################################################
#  R-code: exact gradient
# This is the exact derivative
# Advantages
#   Exact (no numerical rounding/cancellation problems)
#   Fast: all derivatives are computed at once
# Disadvantage
#   -Net structure must be known exactly: the architecture determines the derivative
# Chain-rule: start at the output neuron and go backwards
#   Backpropagation
derivative_w1_backpropagation<-function(b1,b2,w1,w2,x)
{
# Start at output
  output<-sigmoid(b2+w2*sigmoid(b1+w1*x))
  derivative_output<-output*(1-output)
# go one layer back
  output_one_layer_back<-sigmoid(b1+w1*x)
  derivative_one_layer_back<-derivative_output*w2*output_one_layer_back*(1-output_one_layer_back)*x
# Derivative with respect to w1
  dw1<--2*mean((y-output)*derivative_one_layer_back)

  return(dw1)
}

# This is the exact expression (analytic derivative)
derivative_w1_backpropagation(b1,b2,w1,w2,x)
# Compare with above discrete derivative: the exact solution is 'better'
dw1


# At optimum the derivative should vanish
#   We here plug in the estimate of neuralnet and check this assumption
b1<-nn$weights[[1]][[1]][1]
w1<-nn$weights[[1]][[1]][2]
b2<-nn$weights[[1]][[2]][1]
w2<-nn$weights[[1]][[2]][2]

# It is small but it does not vanish...
derivative_w1_backpropagation(b1,b2,w1,w2,x)

# We here plug in our own optimum
b1<-par_optim[1]
w1<-par_optim[2]
b2<-par_optim[3]
w2<-par_optim[4]

# It is zero at w1!!!
derivative_w1_backpropagation(b1,b2,w1,w2,x)
# Discrete gradient is zero too for w1
gradient_func(par_optim,x,y,delta)
# Our optimized solution is better!


###################################################
### code chunk number 11: init
###################################################
############################################################################################################
# R-code: Comparison with linear regression
lm_obj<-lm(y~x)

summary(lm_obj)
# Markedly better than neural!!!! Why?
mean(lm_obj$res^2)
# Optimal net
opt_mse


y
min(y)
max(y)

y_shift<-y-min(y)

scaling<-max(y_shift)

y_scaled<-y_shift/scaling

y_scaled
min(y_scaled)
max(y_scaled)
# Numerical optimization in R with scaled data
nlmin_obj<-nlminb(parm_init,optimize_toy_net,x=x,y=y_scaled)

par_optim<-nlmin_obj$par
names(par_optim)<-c("b1","b2","w1","w2")
par_optim

# Mean-square at scaled data points
nlmin_obj$objective

# Mean-square for original (unscaled) data points: this is better than regression!!!
nlmin_obj$objective*scaling^2
mean(lm_obj$res^2)




