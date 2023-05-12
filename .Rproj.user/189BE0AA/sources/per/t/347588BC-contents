rm(list=ls())
###################################################
### code chunk number 1: init
###################################################
# The following code is pasted from teaching-lecture ATSF (advanced time series for finance)
# Not all results are used in presentation: selected plots
# Load packages and data (re-fresh data)
path.main<-getwd()

path.out<-paste(path.main,"/Latex/",sep="")
path.dat<-paste(path.main,"/Data/",sep="")
path.pgm<-paste(path.main,"/R/",sep="")
path.sweave<-paste(path.main,"/Rnw/",sep="")
path.results<-paste(path.main,"/results/",sep="")

# Load all relevant packages

inst_pack<-rownames(installed.packages())
if (!"fGarch"%in%inst_pack)
  install.packages("fGarch")
if (!"xts"%in%inst_pack)
  install.packages("xts")

if (!"fGarch"%in%inst_pack)
  install.packages("fGarch")
if (!"neuralnet"%in%inst_pack)
  install.packages("neuralnet")
#if (!"xtable"%in%inst_pack)
#  install.packages("xtable")
if (!"CaDENCE"%in%inst_pack)


library(xts)
library(quantmod)
library(PerformanceAnalytics)
library(xts)
library(fGarch)

source(paste(getwd(),"/R/neuralnet_functions.R",sep=""))

# Some of the computations take a couple minutes: set recompute_resultsy<-F to skip computations (results will be loaded from result-folder)
recompute_results<-F


#############################################################################################################
# Load Bitcoin data
path.dat<-paste(getwd(),"/Data/",sep="")
#path.dat<-paste(path.main,"/Exercises/Erste Woche/Data/",sep="")


load_data<-F

if (load_data)
{
  getSymbols("BTC-USD")
  BTC<-get("BTC-USD")
  colnames(BTC)<-c("open","high","low","close","volume","adjusted")
  save(BTC,file=paste(path.dat,"BTC.Rdata",sep=""))
} else
{
  load(file=paste(path.dat,"BTC.Rdata",sep=""))
}

tail(BTC)
dat<-BTC

#-------------------
# Plot  data


# plot last, bid and ask in single figure names(dat)
par(mfrow=c(2,2))
plot(dat$close,col=1,main="Prices")
plot(log(dat$close),col=1,on=1,main="Log-prices")  #tail(dat$Bid)
plot(diff(log(dat$close)),col=1,on=1,main="Log-returns")
plot(log(dat$volume),col=1,on=1,main="Log-volumes")

#----------------------
# Specify target and explanatory data: we use first six lags based on above data analysis
x<-ret<-na.omit(diff(log(dat$close)))
x_level<-log(dat$close)
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
in_out_sample_separator<-"2018-01-01"

# Use original log-returns without scaling or transformation: xts-objects
# These are used for computing trading performances
y_test_xts<-target_out<-data_mat[paste(in_out_sample_separator,"/",sep=""),1]
x_test_xts<-explanatory_out<-data_mat[paste(in_out_sample_separator,"/",sep=""),2:ncol(data_mat)]
y_train_xts<-target_out<-data_mat[paste("/",in_out_sample_separator,sep=""),1]
x_train_xts<-explanatory_out<-data_mat[paste("/",in_out_sample_separator,sep=""),2:ncol(data_mat)]
# Buy and hold benchmark: in and out-sample
bh_out<-cumsum(data_mat[paste(in_out_sample_separator,"/",sep=""),1])
bh_in<-cumsum(data_mat[paste("/",in_out_sample_separator,sep=""),1])
plot(bh_in)
plot(bh_out)

#-----------------------------------------------------------------------------
# Activation function: sigmoid (atan_not_sigmoid<-F) or atan (atan_not_sigmoid<-T)
# Atan often leads to tighter fit (smaller MSE)
# We select atan (requires fewer epochs to converge)
atan_not_sigmoid<-T

# Scaling data for neural net: depends on activation function!!!!!!!!!!!!!!!!
maxs <- apply(data_mat, 2, max)
mins <- apply(data_mat, 2, min)
# Transform data into [0,1]
scaled <- scale(data_mat, center = mins, scale = maxs - mins)
# If atan then transform to [-1,1]
if (atan_not_sigmoid)
  scaled<-2*(scaled-0.5)
apply(scaled,2,min)
apply(scaled,2,max)
#-----------------
# Train-test split
# The scaled data is uswd for parameter fitting
# The scaling depends on the activation function: in [0,1] for sigmoid and in [-1,1] for atan
# xts objects
train_set_xts <- scaled[paste("/",in_out_sample_separator,sep=""),]
test_set_xts <- scaled[paste(in_out_sample_separator,"/",sep=""),]
# as matrices
train_set<-as.matrix(train_set_xts)
test_set<-as.matrix(test_set_xts)

# These are scaled data for traing purposes: they are not xts-objects
x_train<-train_set[,-1]
y_train<-train_set[,1]

# Plot of complex net
neuron_vec<-c(100)
colnames(train_set)<-paste("lag",0:(ncol(train_set)-1),sep="")
n <- colnames(train_set)
# Model: target is current bitcoin, all other variables are explanatory
f <- as.formula(paste("lag0 ~", paste(n[!n %in% "lag0"], collapse = " + ")))

if (recompute_results)
{
  set.seed(1)
  nn <- neuralnet(close~close.1+close.2+close.3+close.4+close.5+close.6,data=train_set,hidden=neuron_vec,linear.output=F)
  plot(nn,rep="best")
}

#------------------------------------------------------
#-------------------------------------------------------
# We use our own package for computing feeforward nets
# The reason is: other packages have not implemented the XAI-tool
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
# R-code: illustration LPD
# Feedforward net with single hidden layer and 4 neurons: one can add layers and neurons simply by changing the vector
neuron_vec<-c(4)
# Explanatory data for training
x_train<-as.matrix(train_set[,2:ncol(train_set)])
# Target for training
y<-y_train<-as.matrix(train_set[,1],ncol=1)

# Output of neural net: linear or non-linear
# We can use linear also for unscaled data: if linear_output==T the scaling could generally be skipped/ignored
# Scaling generally helps optimization
linear_output<-T

# Set-up neural net
hidden_neurons<-neuron_vec

list_layer_size<-layer_size<-getLayerSize(x_train, y_train, hidden_neurons)

learning_rate<-1
epochs<-10000
#--
if (linear_output)
  learning_rate<-learning_rate/10
#--
# Set-up all hyperparameters for optimization
hyper_list<-list(epochs=epochs,learning_rate=learning_rate,linear_output=linear_output,atan_not_sigmoid=atan_not_sigmoid,neuron_vec=neuron_vec)

#------------------
# Train/Learn/Optimize
setseed<-10
set.seed(setseed)
if (recompute_results)
{
# Use backpropagation
  train_model <- trainModel(x_train, y_train, hyper_list=hyper_list)
  save(train_model,file=paste(path.results,"train_model_",neuron_vec,"_",atan_not_sigmoid,sep=""))
} else
{
  load(file=paste(path.results,"train_model_",neuron_vec,"_",atan_not_sigmoid,sep=""))
}
# Check convergence
ts.plot(train_model$cost_hist)
cost_simple_backprop<-train_model$cost_hist[length(train_model$cost_hist)]
# Retrieve optimized parameters
updated_params<-train_model$updated_params

#------------------------------------------------------------------
# Compute LPD
# In-sample LPD
fwd_prop <- forwardPropagation(x_train, updated_params, layer_size,linear_output,atan_not_sigmoid)
cache<-fwd_prop
output<-fwd_prop$A_list[[length(fwd_prop$A_list)]]
# MSE of optimized net
cost <- computeCost(y_train, fwd_prop)
cost

# Select arbitrary observation
t<-13
# Select any explanatory variable
k<-2
# Slightly perturbate the variable
delta<-0.0001
x_train_modified<-x_train
x_train_modified[t,k]<-x_train[t,k]+delta
# Re-run net with perturbation at input
fwd_prop_modified <- forwardPropagation(x_train_modified, updated_params, layer_size,linear_output,atan_not_sigmoid)
cache_modified<-fwd_prop_modified
output_modified<-fwd_prop_modified$A_list[[length(fwd_prop$A_list)]]
# Compute discrete partial derivative
(output_modified[t]-output[t])/delta


# Compute exact LPD: always preferable to discrtete proxy!!!
LPD_obj<-LPD( cache, updated_params, list_layer_size,linear_output,atan_not_sigmoid)

in_sample_LPD_t<-LPD_obj$LPD_t

# Should match the above discrete proxy
in_sample_LPD_t[t,k]

# Set-up LPD
LPD_t<-LPD_obj$LPD_t


#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
# R-code: LPD vs. regression

mean_LPD<-apply(LPD_t,2,mean)
lm_obj<-lm(y_train~x_train)
summary(lm_obj)

# Compare standarddeviations of regression vs. NN
mean(lm_obj$residuals^2)
cost

# Compare mean-LPD and regression coeffs
mean_LPD
lm_obj$coefficients[-1]



#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
# R-code: plot LPD
ts.plot(in_sample_LPD_t,col=rainbow(ncol(in_sample_LPD_t)))

#--------------------------------------------------------------------
# Out-of-sample LPD
x_test<-as.matrix(test_set[,2:ncol(test_set)])
y_test<-as.matrix(test_set[,1],ncol=1)
# Out-of-sample output of optimized net
fwd_prop <- forwardPropagation(x_test, updated_params, layer_size,linear_output,atan_not_sigmoid)
cache<-fwd_prop
output<-fwd_prop$A_list[[length(fwd_prop$A_list)]]
# MSE of optimized net out-sample (generally slightly larger than in-sample)
cost <- computeCost(y_test, fwd_prop)
cost

# Out-of-sample LPD
LPD_obj<-LPD( cache, updated_params, list_layer_size,linear_output,atan_not_sigmoid)

out_sample_LPD_t<-LPD_obj$LPD_t

# Set-up LPD
LPD_t<-out_sample_LPD_t


par(mfrow=c(1,2))
colo<-rainbow(ncol(LPD_t))
ts.plot(LPD_t,col=colo)
bh_plot<-as.double((bh_out-min(bh_out))/(max(bh_out)-min(bh_out))*(max(LPD_t)-min(LPD_t))+min(LPD_t))
lines(bh_plot)
k<-3
ts.plot(LPD_t[,k],col=colo[k])
bh_plot<-as.double((bh_out-min(bh_out))/(max(bh_out)-min(bh_out))*(max(LPD_t[,k])-min(LPD_t[,k]))+min(LPD_t[,k]))
lines(bh_plot)



#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
# R-code: random LPDs
# Train 10 different random nets
setseed<-10
set.seed(setseed)
anzsim<-9

if (recompute_results)
{
  train_model <- trainModel(x_train, y_train, hyper_list=hyper_list)

  fwd_prop <- forwardPropagation(x_train, updated_params, layer_size,linear_output,atan_not_sigmoid)
  cache<-fwd_prop
  updated_params<-train_model$updated_params

  # Compute exact LPD: always preferable to discrtete proxy!!!
  LPD_obj<-LPD( cache, updated_params, list_layer_size,linear_output,atan_not_sigmoid)

  LPD_t<-LPD_obj$LPD_t

  LPD_array<-array(dim=c(anzsim+1,dim(LPD_t)))
  LPD_array[1,,]<-LPD_t

  for (i in 1:anzsim)
  {
  # Use backpropagation
    setseed<-setseed+1
    set.seed(setseed)

    train_model <- trainModel(x_train, y_train, hyper_list=hyper_list)

    fwd_prop <- forwardPropagation(x_train, updated_params, layer_size,linear_output,atan_not_sigmoid)
    cache<-fwd_prop
    updated_params<-train_model$updated_params

  # Compute exact LPD: always preferable to discrtete proxy!!!
    LPD_obj<-LPD( cache, updated_params, list_layer_size,linear_output,atan_not_sigmoid)


    LPD_t<-LPD_obj$LPD_t
    LPD_array[1+i,,]<-LPD_t
  }
  save(LPD_array,file=paste(path.results,"random_lpd_in_sample_simple_net_without_inter",sep=""))

} else
{
  load(file=paste(path.results,"random_lpd_in_sample_simple_net_without_inter",sep=""))

}

dim(LPD_array[,,1])
dim(x_train)

# select an explanatory
k<-1
colo<-rainbow(ncol(LPD_array[1,,]))
ts.plot(t(LPD_array[,,k]),col=colo[k])


par(mfrow=c(2,3))
# LPD
for (k in 1:6)
{
  mplot<-t(LPD_array[,,k])
  dim(mplot)
  rownames(mplot)<-rownames(x_train)
  mplot<-as.xts(mplot)
  print(plot(mplot,col=colo[k],lwd=1,main=paste("Lag ",k-1,sep="")))
}




#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
# R-code: mean LPD
# Train 10 different random nets

dim(LPD_array)

# select an explanatory
k<-1
colo<-rainbow(ncol(LPD_array[1,,]))
ts.plot(apply(LPD_array[,,k],2,mean),col=colo[k])

mean_LPD<-NULL
for (k in 1:dim(LPD_array)[3])
  mean_LPD<-cbind(mean_LPD,apply(LPD_array[,,k],2,mean))

ts.plot(mean_LPD,col=colo)

#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
# R-code: STD LPD
# Train 10 different random nets

dim(LPD_array)

# select an explanatory
k<-2
colo<-rainbow(ncol(LPD_array[1,,]))
ts.plot(sqrt(apply(LPD_array[,,k],2,var)),col=colo[k])

std_LPD<-NULL
for (k in 1:dim(LPD_array)[3])
  std_LPD<-cbind(std_LPD,sqrt(apply(LPD_array[,,k],2,var)))

ts.plot(std_LPD,col=colo)
bh_plot<-as.double((bh_in-min(bh_in))/(max(bh_in)-min(bh_in))*(max(std_LPD)-min(std_LPD))+min(std_LPD))
lines(bh_plot)
mplot<-std_LPD
rownames(mplot)<-rownames(x_train)
mplot<-as.xts(mplot)
plot(mplot,col=colo,lwd=1,main="Vola randpm LPDs all lags")



# Variance is leading by one day draw-downs
returns<-c(0,diff(bh_plot))
cor_mat<-NULL
max_lead<-5
for (i in 1:max_lead)
{
  cor_mat<-rbind(cor_mat,cor(cbind(std_LPD[1:(nrow(std_LPD)+1-i),],returns[(i):length(returns)]))[1:dim(LPD_array)[3],dim(LPD_array)[3]+1]
)
}
rownames(cor_mat)<-paste("Lead ",-1+1:max_lead,sep="")
# Negative correlation at lead one day
cor_mat

#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
# R-code: LPD with nlminb
neuron_vec<-c(4)
# Explanatory data for training
x_train<-as.matrix(train_set[,2:ncol(train_set)])
# Target for training
y<-y_train<-as.matrix(train_set[,1],ncol=1)

# Set-up neural net
hidden_neurons<-neuron_vec

list_layer_size<-layer_size<-getLayerSize(x_train, y_train, hidden_neurons)

#------------------
# Train/Learn/Optimize
setseed<-10
set.seed(setseed)
# Use nlminb for net optimization
layer_size<-getLayerSize(x_train, y_train, neuron_vec)

# Compute the total number of parameters: we do not distinguish weights/biases or layers here
# nlminb just takes a vector of 'unordered' parameters
parm_len<-compute_number_parameters(layer_size)
obj_crit<-10^90
# Proceed to  estimation
parm<-parm_init<-rnorm(parm_len)/sqrt(parm_len)
# No ordering according to layers, weights and biases are not properly identified
parm
x<-x_train
y<-y_train

if (recompute_results)
{
# Initialize parameters
  obj_crit<-10^90
# Proceed to 10 estimation loops and select best in-sample net
  for (i in 1:10)
  {
    parm<-parm_init<-rnorm(parm_len)/sqrt(parm_len)
# No ordering according to layers, weights and biases are not properly identified
    parm

    x<-x_train
    y<-y_train
# Numerical optimization in R
    nlmin_obj<-nlminb(parm_init,optimize_nlminb_net,x=x,y=y,layer_size=layer_size,linear_output=linear_output,atan_not_sigmoid=atan_not_sigmoid)
#                      control=list(iter.max=10))
    print(nlmin_obj$objective)
    if (nlmin_obj$objective<obj_crit)
    {
      parm_opt<-nlmin_obj$par
      obj_crit<-nlmin_obj$objective
    }
  }
  save(nlmin_obj,file=paste(path.results,"nlmin_obj_",neuron_vec,sep=""))
} else
{
  load(file=paste(path.results,"nlmin_obj_",neuron_vec,sep=""))
  parm_opt<-nlmin_obj$par
  obj_crit<-nlmin_obj$objective
}


# Slightly better fit than backprop above
obj_crit
cost_simple_backprop

# Transform parameters back into weights and biases and arrange into layers
updated_params<-translate_Parameters(list_layer_size,parm_opt)


#------------------------------------------------------------------
# Compute LPD
# In-sample LPD
fwd_prop <- forwardPropagation(x_train, updated_params, layer_size,linear_output,atan_not_sigmoid)
cache<-fwd_prop
output<-fwd_prop$A_list[[length(fwd_prop$A_list)]]
# MSE of optimized net
cost <- computeCost(y_train, fwd_prop)
cost

# Compute exact LPD
LPD_obj<-LPD( cache, updated_params, list_layer_size,linear_output,atan_not_sigmoid)

in_sample_LPD_t<-LPD_obj$LPD_t

ts.plot(in_sample_LPD_t,col=rainbow(ncol(in_sample_LPD_t)))

#--------------------------------------------------------------------
# Out-of-sample LPD
x_test<-as.matrix(test_set[,2:ncol(test_set)])
y_test<-as.matrix(test_set[,1],ncol=1)
# Out-of-sample output of optimized net
fwd_prop <- forwardPropagation(x_test, updated_params, layer_size,linear_output,atan_not_sigmoid)
cache<-fwd_prop
output<-fwd_prop$A_list[[length(fwd_prop$A_list)]]
# MSE of optimized net out-sample (generally slightly larger than in-sample)
cost <- computeCost(y_test, fwd_prop)
cost

# Out-of-sample LPD
LPD_obj<-LPD( cache, updated_params, list_layer_size,linear_output,atan_not_sigmoid)

out_sample_LPD_t<-LPD_obj$LPD_t


par(mfrow=c(2,1))
mplot<-as.xts(in_sample_LPD_t)
plot(mplot,col=rainbow(ncol(in_sample_LPD_t)),lwd=1,main="LPD BTC In-Sample")
mplot<-as.xts(out_sample_LPD_t)
plot(mplot,col=rainbow(ncol(in_sample_LPD_t)),lwd=1,main="LPD BTC Out-of-Sample")

#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
# R-code: LPD complex net with backprop
# Feedforward net with hidden layer of 100 neurons
neuron_vec<-100
# Explanatory data for training
x_train<-as.matrix(train_set[,2:ncol(train_set)])
# Target for training
y<-y_train<-as.matrix(train_set[,1],ncol=1)

# Set-up neural net
hidden_neurons<-neuron_vec

list_layer_size<-layer_size<-getLayerSize(x_train, y_train, hidden_neurons)

learning_rate<-0.1
epochs<-10000
#--
if (linear_output)
  learning_rate<-learning_rate/10
#--
hyper_list<-list(epochs=epochs,learning_rate=learning_rate,linear_output=linear_output,atan_not_sigmoid=atan_not_sigmoid,neuron_vec=neuron_vec)

#------------------
# Train/Learn/Optimize
setseed<-10
set.seed(setseed)

if (recompute_results)
{

# Use backpropagation
  train_model <- trainModel(x_train, y_train, hyper_list=hyper_list)
  save(train_model,file=paste(path.results,"train_model_",neuron_vec,sep=""))
} else
{
  load(file=paste(path.results,"train_model_",neuron_vec,sep=""))
}

# Check convergence
ts.plot(train_model$cost_hist)
# MSE marginally worse than simple net
train_model$cost_hist[length(train_model$cost_hist)]
cost_simple_backprop

# Retrieve optimized parameters
updated_params<-train_model$updated_params

#------------------------------------------------------------------
# Compute LPD
# In-sample LPD
fwd_prop <- forwardPropagation(x_train, updated_params, layer_size,linear_output,atan_not_sigmoid)
cache<-fwd_prop
output<-fwd_prop$A_list[[length(fwd_prop$A_list)]]
# MSE of optimized net
cost <- computeCost(y_train, fwd_prop)
cost

# Compute exact LPD
LPD_obj<-LPD( cache, updated_params, list_layer_size,linear_output,atan_not_sigmoid)

in_sample_LPD_t<-LPD_obj$LPD_t



ts.plot(in_sample_LPD_t,col=rainbow(ncol(in_sample_LPD_t)))

#--------------------------------------------------------------------
# Out-of-sample LPD
x_test<-as.matrix(test_set[,2:ncol(test_set)])
y_test<-as.matrix(test_set[,1],ncol=1)
# Out-of-sample output of optimized net
fwd_prop <- forwardPropagation(x_test, updated_params, layer_size,linear_output,atan_not_sigmoid)
cache<-fwd_prop
output<-fwd_prop$A_list[[length(fwd_prop$A_list)]]
# MSE of optimized net out-sample (generally slightly larger than in-sample)
cost <- computeCost(y_test, fwd_prop)
cost

# Out-of-sample LPD
LPD_obj<-LPD( cache, updated_params, list_layer_size,linear_output,atan_not_sigmoid)

out_sample_LPD_t<-LPD_obj$LPD_t

ts.plot(out_sample_LPD_t,col=rainbow(ncol(out_sample_LPD_t)))
bh_plot<-as.double((bh_out-min(bh_out))/(max(bh_out)-min(bh_out))*(max(out_sample_LPD_t)-min(out_sample_LPD_t))+min(out_sample_LPD_t))
lines(bh_plot)


ts.plot(out_sample_LPD_t,col=rainbow(ncol(out_sample_LPD_t)))
bh_plot<-as.double((bh_out-min(bh_out))/(max(bh_out)-min(bh_out))*(max(out_sample_LPD_t)-min(out_sample_LPD_t))+min(out_sample_LPD_t))
lines(bh_plot)

colo<-rainbow(ncol(in_sample_LPD_t))
par(mfrow=c(2,1))
mplot<-as.xts(in_sample_LPD_t)
plot(mplot,col=rainbow(ncol(in_sample_LPD_t)),lwd=1,main="LPD BTC In-Sample")
for (i in 1:ncol(mplot))
  mtext(paste("Lag ",i-1,sep=""),line=-i,col=colo[i])
mplot<-as.xts(out_sample_LPD_t)
plot(mplot,col=rainbow(ncol(in_sample_LPD_t)),lwd=1,main="LPD BTC Out-of-Sample")
for (i in 1:ncol(mplot))
  mtext(paste("Lag ",i-1,sep=""),line=-i,col=colo[i])


#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
# R-code: LPD complex net with nlminb
# Use nlminb for fitting the net
setseed<-10
set.seed(setseed)
# Use nlminb for net optimization
layer_size<-getLayerSize(x_train, y_train, neuron_vec)

# Compute the total number of parameters: we do not distinguish weights/biases or layers here
# nlminb just takes a vector of 'unordered' parameters
parm_len<-compute_number_parameters(layer_size)
obj_crit<-10^90
# Proceed to  estimation
parm<-parm_init<-rnorm(parm_len)/sqrt(parm_len)
# No ordering according to layers, weights and biases are not properly identified
parm
x<-x_train
y<-y_train
# Convergence is slow: fix upper limit for number of iterations
maxiter<-20

# Numerical optimization in R
if (recompute_results)
{

  nlmin_obj<-nlminb(parm_init,optimize_nlminb_net,x=x,y=y,layer_size=layer_size,linear_output=linear_output,atan_not_sigmoid=atan_not_sigmoid,control=list(iter.max=maxiter))

  save(nlmin_obj,file=paste(path.results,"nlmin_obj_",neuron_vec,sep=""))
} else
{
  load(file=paste(path.results,"nlmin_obj_",neuron_vec,sep=""))
}

nlmin_obj$objective
parm_opt<-nlmin_obj$par


# Transform parameters back into weights and biases and arrange into layers
updated_params<-translate_Parameters(list_layer_size,parm_opt)

#------------------------------------------------------------------
# Compute LPD
# In-sample LPD
fwd_prop <- forwardPropagation(x_train, updated_params, layer_size,linear_output,atan_not_sigmoid)
cache<-fwd_prop
output<-fwd_prop$A_list[[length(fwd_prop$A_list)]]
# MSE of optimized net
cost <- computeCost(y_train, fwd_prop)
cost

# Compute exact LPD
LPD_obj<-LPD( cache, updated_params, list_layer_size,linear_output,atan_not_sigmoid)

in_sample_LPD_t<-LPD_obj$LPD_t

# Comparison with linear regression
apply(in_sample_LPD_t,2,mean)
lm_obj$coefficients[2:length(lm_obj$coefficients)]


ts.plot(in_sample_LPD_t,col=rainbow(ncol(in_sample_LPD_t)))

#--------------------------------------------------------------------
# Out-of-sample LPD
x_test<-as.matrix(test_set[,2:ncol(test_set)])
y_test<-as.matrix(test_set[,1],ncol=1)
# Out-of-sample output of optimized net
fwd_prop <- forwardPropagation(x_test, updated_params, layer_size,linear_output,atan_not_sigmoid)
cache<-fwd_prop
output<-fwd_prop$A_list[[length(fwd_prop$A_list)]]
# MSE of optimized net out-sample (generally slightly larger than in-sample)
cost <- computeCost(y_test, fwd_prop)
cost

# Out-of-sample LPD
LPD_obj<-LPD( cache, updated_params, list_layer_size,linear_output,atan_not_sigmoid)

out_sample_LPD_t<-LPD_obj$LPD_t

ts.plot(out_sample_LPD_t,col=rainbow(ncol(out_sample_LPD_t)))
bh_plot<-as.double((bh_out-min(bh_out))/(max(bh_out)-min(bh_out))*(max(out_sample_LPD_t)-min(out_sample_LPD_t))+min(out_sample_LPD_t))
lines(bh_plot)


#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
# R-code: LPD vs. regression

mean_LPD<-apply(LPD_t,2,mean)
lm_obj<-lm(y_train~x_train)
summary(lm_obj)

# Compare standarddeviations of regression vs. NN
mean(lm_obj$residuals^2)
cost

# Compare mean-LPD and regression coeffs
mean_LPD
lm_obj$coefficients[-1]





#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# This code is pasted from XAI-paper
#---------------------------------------------------------------------------------
#------------------------------------------------------------------------------------

source(paste(path.pgm,"functions_XAI_paper.r",sep=""))
# Scaled data leads to stronger dynamics in the LPD: with single 100-layer or 3,2 layer
use_scaled_data<-T
# Unscaled data leads to EqMA(6): more interesting!!!!
use_scaled_data<-F
# Up to peak of BTC end 2017
in_out_sample_separator<-"2018-01-01"


# Load bitcoin and prepare data for usage with neuralnet package
BTC_obj<-load_BTC_func(in_out_sample_separator,use_scaled_data,load_data)

x_train<-BTC_obj$x_train
x_test<-BTC_obj$x_test
y_train<-BTC_obj$y_train
y_test<-BTC_obj$y_test
train_set<-BTC_obj$train_set
test_set<-BTC_obj$test_set


head(x_train)
tail(x_train)
head(y_train)
tail(y_train)
head(x_test)
tail(x_test)
head(y_test)
tail(y_test)

anzsim<-100


#-------------------------------------------------------------------------
# Settings
# Linear output
#   Sigmoid output would be a problem for unscald data which can be negative
#   Either use linear_output<-T (then the output neuron is linear) or atan_not_sigmoid<-T (atan function can be negative in contrast to sigmoid)
linear_output<-T
# Sigmoid activation
atan_not_sigmoid<-F
# Optimization settings
epochs<-200
learning_rate<-0.1
if (linear_output)
  learning_rate<-learning_rate/10
if (!atan_not_sigmoid)
  learning_rate<-learning_rate*5
hyper_list<-list(epochs=epochs,learning_rate=learning_rate,linear_output=linear_output,atan_not_sigmoid=atan_not_sigmoid,neuron_vec=neuron_vec)

#-----------------
# Linear regression: lag6 most important
lm_obj<-lm(y_train~x_train)
summary(lm_obj)
mean(lm_obj$res^2)
mse_regression_out_sample<-mean((as.double(y_test)-lm_obj$coefficients[1+1:ncol(x_test)]%*%t(x_test))^2)
mse_regression_out_sample
#------------------
# Net unregularized and regularized
list_layer_size<-layer_size<-getLayerSize(x_train, y_train, neuron_vec)
list_layer_size
set.seed(1)

# Number of random nets (each setseed leads to another net)
anzsim<-anzsim

# Train model
# No regularization
lambda<-0.

if (recompute_results)
{
  setseed<-0


  mplot_sign_in<-mplot_prop_in<-mplot_sign_out<-mplot_prop_out<-NULL#<-cumsum(y_test)
  out_mse_in<-out_mse_out<-out_reg_all<-NULL
  pb <- txtProgressBar(min = 1, max = anzsim, style = 3)
  LPD_array_in_sample<-array(dim=c(anzsim,dim(x_train)+c(0,1)))
  LPD_array_out_sample<-array(dim=c(anzsim,dim(x_test)+c(0,1)))
  Hessian_array_in_sample<-array(dim=c(anzsim,dim(x_train)))
  Hessian_array_out_sample<-array(dim=c(anzsim,dim(x_test)))
  for (i in 1:anzsim)#i<-1
  {
# Change seed for each pass-through
    setseed<-setseed+1


    compute_mse_original<-compute_net_func(x_train, y_train,x_test,y_test, hyper_list,lambda,setseed,layer_size)

    updated_params<-compute_mse_original$train_model$updated_params
# In-sample
    x<-x_train
    y<-y_train

    compute_obj<-compute_trade_perf_LPD_Hessian_func(x,y, updated_params, layer_size,linear_output,atan_not_sigmoid)

    output<-compute_obj$output
    perf_sign<-compute_obj$perf_sign
    perf_prop<-compute_obj$perf_prop
    LPD_with_intercept<-compute_obj$LPD_with_intercept
    QPD<-compute_obj$QPD
    out_mse_in<-cbind(out_mse_in,output)
    mplot_sign_in<-cbind(mplot_sign_in,cumsum(perf_sign))
    mplot_prop_in<-cbind(mplot_prop_in,cumsum(perf_prop))
    LPD_array_in_sample[i,,]<-LPD_with_intercept
    Hessian_array_in_sample[i,,]<-QPD
# Out-of-sample: use in-sample net as estimated above
    x<-x_test
    y<-y_test

    compute_obj<-compute_trade_perf_LPD_Hessian_func(x,y, updated_params, layer_size,linear_output,atan_not_sigmoid)

    output<-compute_obj$output
    perf_sign<-compute_obj$perf_sign
    perf_prop<-compute_obj$perf_prop
    LPD_with_intercept<-compute_obj$LPD_with_intercept
    QPD<-compute_obj$QPD
    out_mse_out<-cbind(out_mse_out,output)
    mplot_sign_out<-cbind(mplot_sign_out,cumsum(perf_sign))
    mplot_prop_out<-cbind(mplot_prop_out,cumsum(perf_prop))
    LPD_array_out_sample[i,,]<-LPD_with_intercept
    Hessian_array_out_sample[i,,]<-QPD

    setTxtProgressBar(pb, i)



  }
  close(pb)
  save(mplot_sign_in,file=paste(path.results,"bit_sign_trade_mse_in",sep=""))
  save(mplot_prop_in,file=paste(path.results,"bit_prop_trade_mse_in",sep=""))
  save(out_mse_in,file=paste(path.results,"bit_out_mse_in",sep=""))
  save(LPD_array_in_sample,file=paste(path.results,"LPD_array_in_sample",sep=""))
  save(Hessian_array_in_sample,file=paste(path.results,"Hessian_array_in_sample",sep=""))
  save(mplot_sign_out,file=paste(path.results,"bit_sign_trade_mse_out",sep=""))
  save(mplot_prop_out,file=paste(path.results,"bit_prop_trade_mse_out",sep=""))
  save(out_mse_out,file=paste(path.results,"bit_out_mse_out",sep=""))
  save(LPD_array_out_sample,file=paste(path.results,"LPD_array_out_sample",sep=""))
  save(Hessian_array_out_sample,file=paste(path.results,"Hessian_array_out_sample",sep=""))


} else
{
  load(file=paste(path.results,"bit_sign_trade_mse_in",sep=""))
  load(file=paste(path.results,"bit_prop_trade_mse_in",sep=""))
  load(file=paste(path.results,"bit_out_mse_in",sep=""))
  load(file=paste(path.results,"LPD_array_in_sample",sep=""))
  load(file=paste(path.results,"Hessian_array_in_sample",sep=""))
  load(file=paste(path.results,"bit_sign_trade_mse_out",sep=""))
  load(file=paste(path.results,"bit_prop_trade_mse_out",sep=""))
  load(file=paste(path.results,"bit_out_mse_out",sep=""))
  load(file=paste(path.results,"LPD_array_out_sample",sep=""))
  load(file=paste(path.results,"Hessian_array_out_sample",sep=""))
}


par(mfrow=c(1,1))
mplot<-cbind(cumsum(as.double(y_test)),as.matrix(mplot_sign_out))
colo<-rainbow(ncol(mplot))
plot(mplot[,1],main=paste("Log-perf (sign-rule): ",anzsim," random nets (colored) vs. buy-and-hold (black)"),col=colo[1],type="l",axes=F,xlab="",ylab="",ylim=c(min(mplot),max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[,i],col=colo[i])
lines(mplot[,1],col="black",lwd=4)
lines(apply(mplot[,2:ncol(mplot)],1,mean),col="blue",lwd=3,lty=1)
  #  mtext(colnames(mplot)[1],line=-1,col=colo[1])
  #  mtext(paste("Ideal lowpass with cutoff-length ",per_m,sep=""),line=-2,col=colo[3])
axis(1,at=1:nrow(mplot),labels=index(y_test))#length(index(data_obj$y_test))
axis(2)
box()





# Sharpe-ratio: annualize with sqrt(12) (monthly data)



# Aggregate performance (of all random nets): regularized performs slightly worse
perf_agg_sign_mse<-apply(mplot_sign_out,1,mean)
perf_agg_prop_mse<-apply(mplot_prop_out,1,mean)
# Annualized Sharpe ratios
sharpe_mat<-sqrt(365)*matrix(c(mean(as.double(y_test))/sd(as.double(y_test)),mean(diff(perf_agg_sign_mse),na.rm=T)/sd(diff(perf_agg_sign_mse),na.rm=T),mean(diff(perf_agg_prop_mse),na.rm=T)/sd(diff(perf_agg_prop_mse),na.rm=T)),nrow=1)
colnames(sharpe_mat)<-c("Buy-and-hold","Sign","Proportional")
sharpe_mat
mplot<-cbind(cumsum(as.double(y_test)),as.double(perf_agg_sign_mse),as.double(perf_agg_prop_mse))
par(mfrow=c(1,1))
plot(mplot[,1],main=paste("Out-of-sample performances: NN (red) vs. buy-and-hold (black)"),col="black",type="l",axes=F,xlab="",ylab="",ylim=c(min(mplot[,1:2]),max(mplot[,1:2])))
lines(mplot[,2],col="red")
#  lines(mplot[,3],col="blue")
mtext(paste("Buy and hold: Sharpe-ratio ",round(sharpe_mat[1,1],2)),line=-1,col="black")
mtext(paste("NN: Sharpe-ratio ",round(sharpe_mat[1,2],2)),line=-2,col="red")
#  mtext(paste("Proportionality-rule: Sharpe-ratio ",round(sharpe_mat[1,3],2)),line=-3,col="blue")
axis(1,at=1:nrow(mplot),labels=index(y_test))
axis(2)
box()




par(mfrow=c(1,1))
LPD_t_agg<-LPD_sd<-NULL
# With or without intercept
with_intercept<-T
start_i<-ifelse(with_intercept,1,2)
if (with_intercept)
{
  start_i<-1
  main_text<-paste("Mean out-of-sample LPD of ",anzsim," random nets: explanatories (colored) and intercept (black)")
} else
{
  start_i<-2
  main_text<-paste("Mean out-of-sample LPD of ",anzsim," random nets")
}

for (i in start_i:dim(LPD_array_out_sample)[3])
{
# Mean  LPD
  LPD_t_agg <-cbind(LPD_t_agg,apply(LPD_array_out_sample[,,i],2,mean))
# Standarddeviation of LPD
  LPD_sd <-cbind(LPD_sd,sqrt(apply(LPD_array_out_sample[,,i],2,var)))

}
mplot_mean<-LPD_t_agg
mplot_sd<-LPD_sd
if (with_intercept)
{
  colnames(mplot_mean)<-colnames(mplot_sd)<-c("intercept",colnames(x_test))
} else
{
  colnames(mplot_mean)<-colnames(mplot_sd)<-colnames(x_test)
}
mplot_all<-mplot_all_lpd_btc<-mplot_mean
lpd_btc<-mplot_all
# Compute minimum correlation of random LPD and mean-LPD for each explanatory
mat_min_cor_LPD_btc<-NULL
for (i in 1:(layer_size$n_x))
  mat_min_cor_LPD_btc  <-c(mat_min_cor_LPD_btc,mean(cor(lpd_btc[,i],t(LPD_array_out_sample[,,i]))))
mat_min_cor_LPD_btc<-t(as.matrix(mat_min_cor_LPD_btc,nrow=1))
# Add row of cross correlation across mean LPDs
mat_min_cor_LPD_btc<-rbind(cor(lpd_btc)[2,2:ncol(lpd_btc)],mat_min_cor_LPD_btc)
colnames(mat_min_cor_LPD_btc)<-paste("Lag ",1:layer_size$n_x,sep="")
rownames(mat_min_cor_LPD_btc)<-c("Correlation across mean LPDs","Correlation of mean and random LPDs")


# This plot is without intercept
mplot<-mplot_all[,2:ncol(mplot_all)]
main_text<-paste("Out-of-sample LPD")
#  par(mfrow=c(1,2))
colo<-rainbow(ncol(mplot))
colnames(mplot)<-paste("Lag ",1:ncol(mplot),sep="")


plot(mplot[,1],main=main_text,col=colo[1],type="l",axes=F,xlab="",ylab="",ylim=c(min(mplot)-0.001,max(mplot)))
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
  #  mtext(paste("Ideal lowpass with cutoff-length ",per_m,sep=""),line=-2,col=colo[3])
axis(1,at=1:nrow(mplot),labels=index(y_test))#length(index(data_obj$y_test))
axis(2)
box()









par(mfrow=c(2,1))
LPD_t_agg<-NULL
  # Select explanatory:
  #   k=0: intercept
  #   k>0: lag k bitcoin
k<-6
mplot<-t(LPD_array_out_sample[,,k+1])
dim(mplot)
colo<-rainbow(ncol(mplot))
plot(mplot[,1],main=paste("LPDs of ",anzsim," random nets for explanatory ",k,sep=""),col=colo[1],type="l",axes=F,xlab="",ylab="",ylim=c(min(0,min(mplot)),max(mplot)))
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
  #  mtext(colnames(mplot)[1],line=-1,col=colo[1])
  #  mtext(paste("Ideal lowpass with cutoff-length ",per_m,sep=""),line=-2,col=colo[3])
axis(1,at=1:nrow(mplot),labels=index(y_test))#length(index(data_obj$y_test))
axis(2)
box()
me<-apply(LPD_array_out_sample[,,k+1],2,mean)
st<-sqrt(apply(LPD_array_out_sample[,,k+1],2,var))
mplot<-cbind(me+2*st,me,me-2*st)
colo<-rainbow(ncol(mplot))
colnames(mplot)<-c("upper two-sigma","mean","lower two-sigma")
plot(mplot[,1],main=paste("Mean and two-sigma band of above LPD realizations"),col=colo[1],type="l",axes=F,xlab="",ylab="",ylim=c(min(mplot),max(mplot)))
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
  #  mtext(colnames(mplot)[1],line=-1,col=colo[1])
  #  mtext(paste("Ideal lowpass with cutoff-length ",per_m,sep=""),line=-2,col=colo[3])
axis(1,at=1:nrow(mplot),labels=index(y_test))#length(index(data_obj$y_test))
axis(2)
box()



sharpe_vec<-sqrt(365)*apply(apply(mplot_sign_out,2,diff),2,mean)/sqrt(apply(apply(mplot_sign_out,2,diff),2,var))
# In case of multiple identical performances: use first only
best_p<-which(sharpe_vec==max(sharpe_vec))[1]
# In case of multiple identical performances: use first only
worst_p<-which(sharpe_vec==min(sharpe_vec))[1]
par(mfrow=c(1,2))
mplot<-LPD_array_out_sample[best_p,,-1]
colnames(mplot)<-colnames(x_test)
colo<-rainbow(ncol(mplot))

#-----------------------------------
# Risk management
# Exploiting LPD

par(mfrow=c(1,1))
# k=1,...,6 are lagged BTC-returns
# Note: we use lag-one for quantiles based on QPD
k<-1
# A trade per week
quant_s<-7

quantile_select<-quantile_select_f<-1-1/quant_s
# Rolling window of length 10*quant_s: we then have 10 observations above quantile (sufficiently precise for quantile estimation)
length_roll_quantile<-10*quant_s

# Trading rule: exit market based on rolling quantile of QPD and of BTC
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ATTENTION: solutions are a bit lengthy to compute
#   Therefore they are saved and loaded
#   The name of the solution includes quantile_select and length_roll_quantile but NOT LAG k
#   If k changes one has to run the code by hand anew or set recompute_results<-T
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
QPD_array_out_sample<-Hessian_array_out_sample

mplot_obj<-generate_quantile_QPD_adjusted_performance_func(QPD_array_out_sample,k,y_test,quantile_select,length_roll_quantile,path.results,recompute_results,x_test)

# Upper quantile of QPD
roll_quant_up<-mplot_obj$roll_quant_up

mplot_all<-mplot_obj$mplot_all
# This will be used for further scrutiny/analysis below
weight_trade_analysis_QPD_low<-mplot_obj$weight_trade_low
weight_trade_analysis_QPD_up<-mplot_obj$weight_trade_up

# Check empirical quantiles: should correspond roughly to 1/quant_s
if (F)
{
  sum(na.exclude(!weight_trade_analysis_QPD_up))/length(na.exclude(weight_trade_analysis_QPD_up))
  sum(na.exclude(!weight_trade_analysis_QPD_low))/length(na.exclude(weight_trade_analysis_QPD_low))
  1/quant_s
}

# Select buy-and-hold, perf with lower and upper quantiles as well as lagged QPD
mplot<-mplot_all[,c(1:3,6)]
weight_up<-mplot_all[,ncol(mplot_all)-2]
weight_low<-mplot_all[,ncol(mplot_all)-1]
mplot<-t(t(mplot)-mplot[1,])
sharpe_vec<-sqrt(365)*apply(apply(mplot,2,diff),2,mean)/sqrt(apply(apply(mplot,2,diff),2,var))
colnames(mplot)<-c(paste("Buy-and-hold: Sharpe ratio ",round(sharpe_vec[1],2),sep=""),paste("QPD < 1/",quant_s,"-quantile: Sharpe ratio ",round(sharpe_vec[2],2),sep=""),paste("QPD > 1-1/",quant_s,"-quantile: Sharpe ratio ",round(sharpe_vec[3],2),sep=""),paste("(Scaled and shifted) QPD lag ",k,sep=""))
colo<-c("black","blue","red","brown","darkgreen")
# Scale and shift QPD so that it fits with BTC-performances
mplot[,ncol(mplot)]<-(mplot[,ncol(mplot)]-min(mplot[,ncol(mplot)]))*(max(mplot[,1])-min(mplot[,1]))/(max(mplot[,ncol(mplot)])-min(mplot[,ncol(mplot)]))+min(mplot[,1])



# Plot with QPD-based risk-management strategy
plot(mplot[,1],main=paste("Buy-and-hold (",colo[1],") vs QPD-RM (red)",sep=""),col=colo[1],type="l",axes=F,xlab="",ylab="",ylim=c(min(mplot),max(mplot)),lwd=2)
# weight_low is based on lagged bitcoin data: therefore we use which(!weight_low)-1 (instead of which(!weight_low))
abline(v=which(!weight_up)-1,col="grey",lty=3)
lines(mplot[,3],col=colo[3],lwd=2)
lines(mplot[,4],col=colo[5])
mtext(colnames(mplot)[3],col=colo[3],line=-2,lwd=2)
mtext(colnames(mplot)[1],col=colo[1],line=-1,lwd=2)
mtext(colnames(mplot)[ncol(mplot)],col=colo[5],line=-3,lwd=2)
axis(1,at=1:nrow(mplot),labels=rownames(mplot))#length(index(data_obj$y_test))
axis(2)
box()


# Plot with QPD of lag-one BTC and upper 1-1/7 quantile
Hessian_t_agg<-NULL

for (i in 1:dim(Hessian_array_out_sample)[3])
  Hessian_t_agg <-cbind(Hessian_t_agg,apply(Hessian_array_out_sample[,,i],2,mean))
# Add lagged upper-quantile of QPD: lag has no effect on time-shift but is used to ignore current extreme data,
#   see description/explanation in function weight_trade_func (in file functions_XAI_paper.r)
mplot<-cbind(Hessian_t_agg,lag(roll_quant_up))
colnames(mplot)<-c(colnames(x_test),"upper quantile")
# Check the quantile-rule
if (F)
{
  which(!weight_trade_analysis_QPD_up)
# The lagged quantile is used in order to ignore possible current outliers: this does not impact shift/delay, see weight_trade_func function for details
  which(mplot[,1]>mplot[,ncol(mplot)])
}

par(mfrow=c(1,1))
colnames(mplot)<-paste("Lag ",1:ncol(mplot),sep="")
mplot<-as.matrix(mplot)
plot(mplot[,1],main=paste("Out-of-sample QPD (green) and rolling q(1-1/7) quantile (blue)"),col="darkgreen",type="l",axes=F,xlab="",ylab="")
lines(mplot[,ncol(mplot)],col="blue")
  #  mtext(colnames(mplot)[1],line=-1,col=colo[1])
  #  mtext(paste("Ideal lowpass with cutoff-length ",per_m,sep=""),line=-2,col=colo[3])
axis(1,at=1:nrow(mplot),labels=index(y_test))#length(index(data_obj$y_test))
axis(2)
box()



#--------------------------------------------
# Analysis of critical time points identified by LPD (LPD<lower quantile)
perf_w_up<-na.exclude(y_test[!weight_trade_analysis_QPD_up])
perf_w_low<-na.exclude(y_test[!weight_trade_analysis_QPD_low])
# The effect is not in sign-prediction accuracy (there are even more positives than negatives)
sum(perf_w_up<0)/length(perf_w_up)
sum(perf_w_up>0)/length(perf_w_up)
# But the effect is on mean or drift during critical time points identified by LPD: -0.55% per day
mean(perf_w_up)
# vs. +0.14% per day for BTC overall
mean(y_test)

# Analysis of auspicious time points identified by LPD (LPD>upper quantile)
perf_w_up<-na.exclude(y_test[!weight_trade_analysis_QPD_up])
# The effect is not in sign-prediction accuracy (there are even more positives than negatives)
sum(perf_w_up<0)/length(perf_w_up)
sum(perf_w_up>0)/length(perf_w_up)
# But the effect is on mean or drift during critical time points identified by LPD: -0.55% per day
mean(perf_w_up)
# vs. +0.14% per day for BTC overall
mean(y_test)

# Analysis of neutral time points identified by LPD (lower quantile<LPD<upper quantile)
perf_w_neut<-na.exclude(y_test[weight_trade_analysis_QPD_up&weight_trade_analysis_QPD_low])
# The effect is not in sign-prediction accuracy (there are even more positives than negatives)
sum(perf_w_neut<0)/length(perf_w_neut)
sum(perf_w_neut>0)/length(perf_w_neut)
# But the effect is on mean or drift during critical time points identified by LPD: -0.55% per day
mean(perf_w_neut)
# vs. +0.14% per day for BTC overall
mean(y_test)




# Plot of cumsum: drift is negative and systematic for selected critical time points identified by LPD
par(mfrow=c(2,2))
ts.plot(cumsum(perf_w_up),ylab="Cumulated log-performance",main="Large QPD (non-linear)")
ts.plot(cumsum(perf_w_low),ylab="Cumulated log-performance",main="Small QPD (linear)")
ts.plot(cumsum(perf_w_neut),ylab="Cumulated log-performance",main="Mid QPD ")
#ts.plot(cumsum(perf_w_up_20),ylab="Performance during critical time points",xlab="Critical time points selected by low values of LPD")

mat_crit<-t(c(sum(perf_w_up>0)/length(perf_w_up),mean(perf_w_up),sd(perf_w_up)/sqrt(length(perf_w_up))))
mat_crit<-rbind(mat_crit,c(sum(perf_w_neut>0)/length(perf_w_neut),mean(perf_w_neut),sd(perf_w_neut)/sqrt(length(perf_w_neut))))
mat_crit<-rbind(mat_crit,c(sum(perf_w_low>0)/length(perf_w_low),mean(perf_w_low),sd(perf_w_low)/sqrt(length(perf_w_low))))
# Use only those future returns which coincide with trading orders
y_t<-y_test[weight_trade_analysis_QPD_low|weight_trade_analysis_QPD_up]
# Concatenate performances for exceedances of upper quantile and (negative) exceedances of lower quantile and compute t-statistic: significant
aggul<-c(perf_w_up,-perf_w_low)
tstat<-sqrt(length(aggul))*mean(aggul)/sd(aggul)


mat_crit<-rbind(mat_crit,c(sum(y_t>0)/length(y_t),mean(y_t),sd(y_t)/sqrt(length(y_t))))
colnames(mat_crit)<-c("Proportion of positive signs","Average next days' returns","Standard deviations")
rownames(mat_crit)<-c("Critical time points","Neutral time points","Auspicious time points","All time points")




###################################################
### code chunk number 11: tab_vola
###################################################
a_m<-matrix(c(paste(round(100*mat_crit[,1],1),"%",sep=""),paste(round(100*mat_crit[,2],3),"%",sep="")),ncol=2)
colnames(a_m)<-c("Prop. Positive sign","Average return")
rownames(a_m)<-rownames(mat_crit)

a_m
