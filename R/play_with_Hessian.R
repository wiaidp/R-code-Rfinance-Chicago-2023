
rm(list=ls())
# Load all relevant packages

inst_pack<-rownames(installed.packages())
if (!"fGarch"%in%inst_pack)
  install.packages("fGarch")
if (!"xts"%in%inst_pack)
  install.packages("xts")

if (!"neuralnet"%in%inst_pack)
  install.packages("neuralnet")

# Use iml package for classic XAI approaches see: https://cran.r-project.org/web/packages/iml/vignettes/intro.html

library(neuralnet)
library(fGarch)
library(xts)

source(paste(getwd(),"/R/neuralnet_functions.R",sep=""))
source(paste(getwd(),"/R/data_generating_functions.R",sep=""))
#######################################################################
######################################################################################################################
# 4. Discrete Hessian:
#   Identifiability/variance: use (derivative of) LPD_cost for identification
#   Non-linearity: use (derivative of) LPD for non-linearity



# Settings
use_random_net<-F
neuron_vec<-c(10,5,2)
#neuron_vec<-c(1)
linear_output<-F
atan_not_sigmoid<-T
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
len<-100
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
  if (F)
  {
    parm$W_list[[1]]<-matrix(rep(1,ncol(x_train)),nrow=1)
    parm$W_list[[2]]<-1
    parm$b_list[[1]]<-0
    parm$b_list[[2]]<-0


  }
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


fwd_prop <- forwardPropagation(x_train, parm, layer_size,linear_output,atan_not_sigmoid)
cache<-fwd_prop
LPD_obj<-LPD(cache, parm, list_layer_size,linear_output,atan_not_sigmoid)
LPD_exact<-LPD_obj$LPD
LPD_t<-LPD_obj$LPD_t
ts.plot(LPD_t)
dA_list<-LPD_obj$dA_list
LPD_forward_obj<-LPD_forward(cache, parm, list_layer_size,linear_output,atan_not_sigmoid)
dA_list_forward<-LPD_forward_obj$dA_list_forward

# Hessian_diag computes the whole diagonal of the Hessian matrix
Hess_obj<-Hessian_diag(dA_list,cache, parm, list_layer_size,linear_output,atan_not_sigmoid,dA_list_forward)


k<-1
delta<-0.00001
#delta<-0.001
#delta<-.0001
x_train_modified<-x_train
x_train_modified[,k]<-x_train[,k]+delta
fwd_prop <- forwardPropagation(x_train, parm, layer_size,linear_output,atan_not_sigmoid)
LPD_obj<-LPD(fwd_prop, parm, list_layer_size,linear_output,atan_not_sigmoid)
LPD_t<-LPD_obj$LPD_t
LPD_exact<-LPD_obj$LPD
fwd_prop_modified <- forwardPropagation(x_train_modified, parm, layer_size,linear_output,atan_not_sigmoid)
LPD_obj_mod<-LPD(fwd_prop_modified, parm, list_layer_size,linear_output,atan_not_sigmoid)
LPD_t_mod<-LPD_obj_mod$LPD_t
LPD_exact_mod<-LPD_obj_mod$LPD

Hess_obj$Hessian_t[,k]/((LPD_t_mod[,k]-LPD_t[,k])/delta)

# Hessian_t computes the complete Hessian (all partial derivatives) for a single time point t
t<-10

Hess_t_obj<-Hessian_t(dA_list,cache, parm, list_layer_size,linear_output,atan_not_sigmoid,dA_list_forward,t)

# Complete Hessain at time point t
Hess_t_obj$Hessian
# Difference of diagonal elements should vanish at time point t
diag(Hess_t_obj$Hessian)-Hess_obj$Hessian_t[t,]


ts.plot(LPD_t,col=rainbow(ncol(LPD_t)))
ts.plot(Hess_obj$Hessian_t,col=rainbow(ncol(LPD_t)))

# This function computes the Hessian with respect to x_j,x_k i.e. off-diagonal are possible, too
k<-k
j<-k

Hess_jk_obj<-Hessian_kj(dA_list,cache, parm, list_layer_size,linear_output,atan_not_sigmoid,dA_list_forward,k,j)

Hess_jk_obj$Hessian_t/((LPD_t_mod[,k]-LPD_t[,k])/delta)

# Should be exactly one
Hess_jk_obj$Hessian_t/Hess_obj$Hessian_t[,k]

# Check off-diagonal elements at time point t of complete Hessian above

k<-2
j<-3

Hess_jk_obj<-Hessian_kj(dA_list,cache, parm, list_layer_size,linear_output,atan_not_sigmoid,dA_list_forward,k,j)

Hess_t_obj$Hessian[k,j]-Hess_jk_obj$Hessian_t[t,]



#------------------------------------------------------------------------------------------------
# Compute entire Hessian matrix

Hess_mat<-matrix(ncol=list_layer_size$n_x,nrow=list_layer_size$n_x)
for (j in 1:list_layer_size$n_x)
{
  for (k in j:list_layer_size$n_x)
  {
    Hess_jk_obj<-Hessian_kj(dA_list,cache, parm, list_layer_size,linear_output,atan_not_sigmoid,dA_list_forward,k,j)
    Hess_mat[k,j]<-Hess_mat[j,k]<-Hess_jk_obj$Hessian
  }
}

solve(Hess_mat)


