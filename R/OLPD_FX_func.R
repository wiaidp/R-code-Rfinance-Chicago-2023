# Topic: regression vs. simple feedforward based on nn
#   Application to bitcoin
# First week

rm(list=ls())


# Die folgenden packages muss man beim ersten Mal installieren (falls noch nicht installiert...)
if (F)
{
  install.packages("neuralnet")
  install.packages("fGarch")
  install.packages("xts")
  install.packages("Quandl")
}
library(neuralnet)
library(fGarch)
library(xts)
library(Quandl)

source(paste(getwd(),"/R/data_load_functions.R",sep=""))


data_from_IB<-T
hour_of_day<-"16:00"
reload_sp500<-F
path.dat<-"C:\\wia_desktop\\2020\\Projekte\\IB\\daily_pick\\Data\\IB\\"

data_load_obj<-data_load_gzd_trading_func(data_from_IB,hour_of_day,reload_sp500,path.dat)

mydata<-data_load_obj$mydata
xts_data_mat<-data_load_obj$xts_data_mat

colnames(xts_data_mat)

asset_sel<-"USDCHF"
asset_sel<-"EURCHF"
asset_sel<-"EURUSD"
x<-ret<-diff(log(xts_data_mat[,asset_sel]))
is.xts(x)

#------------------
# 3.b
data_mat<-cbind(x,lag(x),lag(x,k=2),lag(x,k=3),lag(x,k=4),lag(x,k=5),lag(x,k=6))
# Check length of time series before na.exclude
dim(data_mat)
data_mat<-na.exclude(data_mat)
# Check length of time series after removal of NAs
dim(data_mat)
head(data_mat)
tail(data_mat)

#--------------------------------------------------------------------
# 3.c&d Specify in- and out-of-sample episodes
in_out_sample_separator<-"2015-06-01"

target_in<-data_mat[paste("/",in_out_sample_separator,sep=""),1]
tail(target_in)
explanatory_in<-data_mat[paste("/",in_out_sample_separator,sep=""),2:ncol(data_mat)]
tail(explanatory_in)

target_out<-data_mat[paste(in_out_sample_separator,"/",sep=""),1]
head(target_out)
tail(target_out)
explanatory_out<-data_mat[paste(in_out_sample_separator,"/",sep=""),2:ncol(data_mat)]
head(target_out)
tail(explanatory_out)

train<-cbind(target_in,explanatory_in)
test<-cbind(target_out,explanatory_out)
head(test)
tail(test)
nrow(test)

#-------------------------------------------------------------------------------
# 4. Neural net fitting

# 4.a
# Scaling data for the NN
maxs <- apply(data_mat, 2, max)
mins <- apply(data_mat, 2, min)
# Transform data into [0,1]
scaled <- scale(data_mat, center = mins, scale = maxs - mins)

apply(scaled,2,min)
apply(scaled,2,max)
#-----------------
# 4.b
# Train-test split
train_set <- scaled[paste("/",in_out_sample_separator,sep=""),]
test_set <- scaled[paste(in_out_sample_separator,"/",sep=""),]

train_set<-as.matrix(train_set)
test_set<-as.matrix(test_set)
#-----------------------------------
# 4.c

colnames(train_set)<-paste("lag",0:(ncol(train_set)-1),sep="")





# Original linear parameter data
#   Generate new data from original data
#   New data: in each time point compute the parameters of the exact infinitesimal linear regression model
OLPD_func<-function(x,delta,epsilon,nn)
{
  try_data_list<-try(out_original<-compute(nn,x)$net.result,silent=T)

  if(class(try_data_list)[1]=="try-error")
  {
    data_list<-vector(mode="list")
    print("Neural net singular")
    effect<-NULL
    return(list(effect=effect))

  } else
  {



    # For each explanatory...
    for (i in 1:ncol(x))#i<-1
    {
      # y will be the original explanatory plus an infinitesimal perturbation of i-th explanatory
      y<-x
      y[,i]<-y[,i]+delta*x[,i]

    # Generate infinitesimally perturbated output
      out_i <-compute(nn,y)$net.result

      if (i==1)
      {
        effect<-(out_i-out_original)/(delta*x[,i])
      } else
      {
        effect<-c(effect,(out_i-out_original)/(delta*x[,i]))
      }
      # Collect for each explanatory the perturbated data and the corresponding nn-output
      #    }
    }

  # Fit the regression to the noiseless perturbated data: as many observations as unknowns i.e. zero-residual
    return(list(effect=effect))
  }
}





n <- colnames(train_set)
# Model: target is current bitcoin, all other variables are explanatory
f <- as.formula(paste("lag0 ~", paste(n[!n %in% "lag0"], collapse = " + ")))

tail(train_set)

set.seed(1)

neuron_vec<-c(3,2)

nn <- neuralnet(f,data=train_set,hidden=neuron_vec,linear.output=F)

plot(nn)


# Induce infinitesimal perturbations to data and fit regression to output

delta<-1.e-5
epsilon<-1.e-4

pb <- txtProgressBar(min = 1, max = (nrow(train_set)-1), style = 3)

# Rougher (out-sample data)
data<-test_set
# Smoother (in-sample data)
data<-train_set


for (i in 1:(nrow(data)))
{
  x<-matrix(data[i,2:ncol(data)],nrow=1)
  colnames(x)<-colnames(data)[2:ncol(data)]
  OLPD_obj<-OLPD_func(x,delta,epsilon,nn)

  if (i==1)
  {
    OLPD_mat<-OLPD_obj$effect
  } else
  {
    OLPD_mat<-rbind(OLPD_mat,OLPD_obj$effect)
  }
  setTxtProgressBar(pb, i)

}
close(pb)

returnsh<-na.exclude(ret)[paste("/",in_out_sample_separator,sep="")]
returns<-returnsh[ncol(data):(length(returnsh))]

OLPD_mat<-reclass(OLPD_mat,returns)

nrow(OLPD_mat)
length(returns)


par(mfrow=c(2,1))
plot(OLPD_mat,col=rainbow(ncol(OLPD_mat)))
plot(cumsum(returns))
#--------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------



set.seed(1)
# Net architecture
#neuron_vec<-c(36,18)
neuron_vec<-c(24,12)
#neuron_vec<-c(12,6)
#neuron_vec<-c(6,3)
#neuron_vec<-c(4,2)
# Number of random nets
anz_real<-100
# Evaluate net in-sample
data<-train_set
data_xts<-as.xts(data)
# Evaluate net out-of-sample
data<-test_set

# Array of sensitivities
arr_sens<-array(dim=c(anz_real,nrow(data),ncol(data)-1))
pb <- txtProgressBar(min = 1, max = anz_real, style = 3)

compute_results<-F

if (compute_results)
{
  # Loop over all random-nets
  for (j in 1:anz_real)#j<-1
  {

    nn <- neuralnet(f,data=train_set,hidden=neuron_vec,linear.output=F)


    for (i in 1:(nrow(data)))#i<-2
    {
      x<-matrix(data[i,2:ncol(data)],nrow=1)
      colnames(x)<-colnames(data)[2:ncol(data)]

      OLPD_obj<-OLPD_func(x,delta,epsilon,nn)

      if (i==1)
      {
        OLPD_mat<-OLPD_obj$effect
      } else
      {
# If the neural net is NOT singular then we add the last row
        if (length(OLPD_obj$effect)>0)
        {
          OLPD_mat<-rbind(OLPD_mat,OLPD_obj$effect)
        } else
        {
# Otherwise we duplicate last row
          OLPD_mat<-rbind(OLPD_mat,OLPD_mat[nrow(OLPD_mat),])
        }
      }

    }
    OLPD_mat<-reclass(OLPD_mat,data_xts)
    arr_sens[j,,]<-OLPD_mat
    setTxtProgressBar(pb, j)
  }
  close(pb)

  # Compute aggregate/mean sensitivities


  OLPD_mat_mean<-apply(arr_sens,c(2,3),mean)
  OLPD_mat_var<-apply(arr_sens,c(2,3),var)

  save(OLPD_mat_mean,file=paste(getwd(),"/Output/OLPD_mat_mean_",asset_sel,"_",neuron_vec[1],"_",neuron_vec[2],sep=""))
  save(OLPD_mat_var,file=paste(getwd(),"/Output/OLPD_mat_var_",asset_sel,"_",neuron_vec[1],"_",neuron_vec[2],sep=""))
} else
{
  load(file=paste(getwd(),"/Output/OLPD_mat_mean_",asset_sel,"_",neuron_vec[1],"_",neuron_vec[2],sep=""))
  load(file=paste(getwd(),"/Output/OLPD_mat_var_",asset_sel,"_",neuron_vec[1],"_",neuron_vec[2],sep=""))

}
rownames(OLPD_mat_mean)<-rownames(OLPD_mat_var)<-rownames(data)
OLPD_mat_mean<-as.xts(OLPD_mat_mean)
OLPD_mat_var<-as.xts(OLPD_mat_var)
colo<-rainbow(ncol(OLPD_mat_mean))

par(mfrow=c(1,1))
plot(OLPD_mat_mean,main="Mean sensitivities: partial derivative",col=colo,ylim=quantile(OLPD_mat_mean)[c(1,5)])
for (i in 1:ncol(OLPD_mat_mean))
  mtext(paste("Lag ",i,sep=""),col=colo[i],line=-i)



OLPD_mat_lower<-OLPD_mat_mean-2.2*sqrt(OLPD_mat_var)#/sqrt(anz_real)
OLPD_mat_upper<-OLPD_mat_mean+2.2*sqrt(OLPD_mat_var)#/sqrt(anz_real)

rownames(OLPD_mat_lower)<-rownames(OLPD_mat_upper)<-rownames(data)
OLPD_mat_lower<-as.xts(OLPD_mat_lower)
OLPD_mat_upper<-as.xts(OLPD_mat_upper)
colo<-rainbow(ncol(OLPD_mat_lower))
par(mfrow=c(2,2))
# select a particular explanatory
select_exp<-1
mplot<-cbind(OLPD_mat_mean[,select_exp],OLPD_mat_lower[,select_exp],OLPD_mat_upper[,select_exp])
plot(mplot,main=paste("Confidence interval for sensitivity of ",colnames(data)[select_exp+1],sep=""),col=rep(colo[select_exp],3),lty=c(1,2,2))
select_exp<-2
mplot<-cbind(OLPD_mat_mean[,select_exp],OLPD_mat_lower[,select_exp],OLPD_mat_upper[,select_exp])
plot(mplot,main=paste("Confidence interval for sensitivity of ",colnames(data)[select_exp+1],sep=""),col=rep(colo[select_exp],3),lty=c(1,2,2))
select_exp<-3
mplot<-cbind(OLPD_mat_mean[,select_exp],OLPD_mat_lower[,select_exp],OLPD_mat_upper[,select_exp])
plot(mplot,main=paste("Confidence interval for sensitivity of ",colnames(data)[select_exp+1],sep=""),col=rep(colo[select_exp],3),lty=c(1,2,2))
select_exp<-6
mplot<-cbind(OLPD_mat_mean[,select_exp],OLPD_mat_lower[,select_exp],OLPD_mat_upper[,select_exp])
plot(mplot,main=paste("Confidence interval for sensitivity of ",colnames(data)[select_exp+1],sep=""),col=rep(colo[select_exp],3),lty=c(1,2,2))

quantile(OLPD_mat_mean[,select_exp])[c(2,4)]



