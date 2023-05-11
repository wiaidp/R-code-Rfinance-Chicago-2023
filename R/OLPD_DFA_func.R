

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# RUN OLPD_FX_func.r first!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

library(MDFA)
# This is used for implementing rolling quantiles
library(PerformanceAnalytics)





# Filter function: applies a filter b to a series x which can be xts or double
#   If x is xts then time ordering of b is reversed
filt_func<-function(x,b)
{
  L<-nrow(b)
  if (is.matrix(x))
  {
    length_time_series<-nrow(x)
  } else
  {
    if (is.vector(x))
    {
      length_time_series<-length(x)
    } else
    {
      print("Error: x is neither a matrix nor a vector!!!!")
    }
  }
  if (is.xts(x))
  {
    yhat<-x[,1]
  } else
  {
    yhat<-rep(NA,length_time_series)
  }
  for (i in L:length_time_series)#i<-L
  {
    # If x is an xts object then we cannot reorder x in desceding time i.e. x[i:(i-L+1)] is the same as  x[(i-L+1):i]
    #   Therefore, in this case, we have to revert the ordering of the b coefficients.
    if (is.xts(x))
    {
      if (ncol(b)>1)
      {
        yhat[i]<-as.double(sum(apply(b[L:1,]*x[i:(i-L+1),],1,sum)))
      } else
      {
        yhat[i]<-as.double(b[L:1,]%*%x[i:(i-L+1)])#tail(x) x[(i-L+1):i]
      }
    } else
    {
      if (ncol(b)>1)
      {
        yhat[i]<-as.double(sum(apply(b[1:L,]*x[i:(i-L+1),],1,sum)))
      } else
      {
        yhat[i]<-as.double(as.vector(b)%*%x[i:(i-L+1)])#tail(x) x[(i-L+1):i]
      }
    }
  }
  #  names(yhat)<-index(x)#index(yhat)  index(x)
  #  yhat<-as.xts(yhat,tz="GMT")
  return(list(yhat=yhat))
}


# No cheating: real-time
# xyz<-std_mean
# quantile_select<-0.5
weight_trade_func<-function(xyz,quantile_select,length_roll_quantile)
{
# Old code based on median: newer code below is more general since it relies on quantiles
  if (F)
  {
    roll_median<-xyz
# First possibility: from 1 to t i.e. 1:i for i=1,2,...,T
    for (i in 1:length(xyz))
    {
      roll_median[i]<-as.double(median(xyz[1:i]))
    }
# Second possibility: rolling window of length length_roll_median
    roll_median<-rollmedian(xyz,k=length_roll_median)
  }

# Third possibility: more general i.e. arbitrary quantile
  roll_quant<-apply.rolling(xyz,width=length_roll_quantile,FUN="quantile",p=quantile_select)
  if (F)
  {
# This is a check: last element of tail should match median since p=0.5
    tail(apply.rolling(xyz,width=151,FUN="quantile",p=0.5))
    median(xyz[(length(xyz)-151+1):length(xyz)])
  }
# Weight is one if xyz smaller than lagged quantile
# Note that since xyz[i] is not in lagged roll_quant[i] : therefore even if p=1 it could happen that weight_trade[i] is F
  weight_trade<-(xyz<=lag(roll_quant))
  return(list(weight_trade=weight_trade))

}





data_mat_level<-xts_data_mat[,asset_sel]
# Diffs
data_mat_unscaled<-data_mat<-diff(data_mat_level)
# For convenience we scale the data (no centering though...)
#   Scaling has no effect on estimation but it facilitates interpretations (one can set apply_scaling<-T or F leading to identical performances)

apply_scaling<-F

if (apply_scaling)
{
  for (i in 1:ncol(data_mat_unscaled))
    data_mat[,i]<-data_mat_unscaled[,i]/as.double(sqrt(var(data_mat_unscaled[,i],na.rm=T)))
} else
{
  data_mat<-data_mat_unscaled
}

#---------------------------------------------
# Settings (hyperparameters)
#---------------------------------------------
# Settings (hyperparameters)
in_sample<-"2019-01-01"
# Target
periodicity<-2.5
# Nowcast (Lag=1), Backcast (Lag>1) and Forecast (Lag<1)
Lag<-0#-periodicity/2

#-----------------------------------------------
# Some graphs

plot(data_mat[paste(in_sample,"/",sep="")])

if (ncol(data_mat)>1)
{
# Cross-sectional correlation as a function of time
  cor(na.exclude(data_mat))

  cor_mat<-NULL
  for (j in 0:1)
  {
    for (i in 0:9)
    {
      cor_mat<-rbind(cor_mat,cor(na.exclude((data_mat)[paste("20",j,i,sep="")]))[1,2:3])
    }
  }
  ts.plot(cor_mat,lty=1:2,main="Yearly Correlations between explanatory (interest rate differentials) and EURUSD: returns")
abline(h=0)
}
#-----------------------------------------------
# Format data
# Specify in-sample data matrix
data_in<-data_mat[paste("/",in_sample,sep="")]
tail(data_in)
# Lag explanatory
data_matrix_in_sample<-as.matrix(na.exclude(cbind(data_in[,1],(data_in))))
tail(data_matrix_in_sample)
data_matrix_full<-(na.exclude(cbind(data_mat[,1],(data_mat))))
data_matrix_unscaled_full<-(na.exclude(cbind(data_mat_unscaled[,1],(data_mat_unscaled))))
tail(data_matrix_full)
is.xts(data_matrix_full)

#--------------------------------------------
# Estimate spectrum
weight_func_multi<-spec_comp(nrow(data_matrix_in_sample), data_matrix_in_sample, 0)$weight_func
# Resolution of frequency-grid
K<-nrow(weight_func_multi)-1
# White noise
weight_func_multi<-cbind(rep(1,600),rep(1,600))
# Resolution of frequency-grid
K<-nrow(weight_func_multi)-1
# Specify target signal
cutoff<-pi/periodicity
Gamma<-(0:(K))<=K*cutoff/pi+1.e-9
# Filter length: L/K should be 'small'
L<-30
# Estimate filter coefficients
mdfa_obj_multi<-MDFA_mse(L,weight_func_multi,Lag,Gamma)$mdfa_obj
# Filter coefficients
b_multi<-mdfa_obj_multi$b
dimnames(b_multi)[[2]]<-colnames(data_mat)
dimnames(b_multi)[[1]]<-paste("Lag ",0:(L-1),sep="")#dim(b_mat)
apply(b_multi,2,sum)

# First 'toggle breakpoint': interpreting the filter coefficients
head(b_multi)
ts.plot(b_multi,lty=1:3)


# Filtering multi: used for computing time domain MSEs
yhat_multi<-filt_func(data_matrix_full[,2:ncol(data_matrix_full)],b_multi)$yhat
# Trading: note that first column in data_matrix_unscaled_full is leading (by one day)
perf_multi<-na.exclude(lag(sign(yhat_multi))*data_matrix_unscaled_full[,1])
perf_agg<-perf_multi
yhat_agg<-yhat_multi

par(mfrow=c(1,1))
plot(cumsum(perf_multi))
#events<-xts("In sample/out-of-sample",as.Date(in_sample))
#addEventLines(events,srt=90,pos=2,col="red")
sqrt(250)*mean(perf_multi[paste(in_sample,"/",sep="")])/sqrt(var(perf_multi[paste(in_sample,"/",sep="")]))

par(mfrow=c(3,1))
mplot<-na.exclude(cbind(OLPD_mat_mean,cumsum(perf_multi),sqrt(OLPD_mat_var)))
plot(mplot[,1:ncol(OLPD_mat_mean)],main="Mean sensitivities: partial derivative",col=colo,ylim=quantile(OLPD_mat_mean)[c(1,5)])
for (i in 1:ncol(OLPD_mat_mean))
  mtext(paste("Lag ",i,sep=""),col=colo[i],line=-i)
plot(mplot[,(ncol(OLPD_mat_mean)+1)],main="DFA")
plot(mplot[,(ncol(OLPD_mat_mean)+2):ncol(mplot)],col=colo,main="Standard deviations of sensitivities")

# Use either (absolute values of) sensitivities or standard deviations of sensitivities
use_sensitivities<-T
if (!use_sensitivities)
{
# Compute mean of standard deviations of sensitivities
  std_mean<-apply(sqrt(OLPD_mat_var),1,mean)
  std_mean<-reclass(std_mean,OLPD_mat_var)
# Select quantile for escaping market (note that p=1 that does not ensure that one stays in market because the quantile is lagged)
  quantile_select<-0.95
} else
{
# Compute mean of absolute values of sensitivities
  std_mean<-apply(abs(OLPD_mat_mean),1,mean)
  std_mean<-reclass(std_mean,OLPD_mat_mean)
# Select quantile for escaping market (note that p=1 that does not ensure that one stays in market because the quantile is lagged)
  length_roll_quantile<-61
  quantile_select<-0.9
  length_roll_quantile<-250
  quantile_select<-0.8
  length_roll_quantile<-151
  quantile_select<-0.9
}


weight_trade<-weight_trade_func(std_mean,quantile_select,length_roll_quantile)$weight_trade

par(mfrow=c(2,2))
sharpe<-sqrt(250)*mean(perf_multi[index(weight_trade)],na.rm=T)/sqrt(var(perf_multi[index(weight_trade)],na.rm=T))
plot(cumsum(perf_multi[index(na.exclude(weight_trade))]),main=paste("DFA: sharpe=",round(sharpe,3),sep=""))
perf_weight<-na.exclude(lag(weight_trade)*perf_multi)
sharpe<-sqrt(250)*mean(perf_weight,na.rm=T)/sqrt(var(perf_weight,na.rm=T))
plot(cumsum(perf_weight),main=paste("Long only: sharpe=",round(sharpe,3),sep=""))
# Add: short on trading strategy when vola high
#   Not to be recommended when trading strategy OK overall...
perf_weight_long_short<-na.exclude(lag(weight_trade)*perf_multi+lag(weight_trade-1)*perf_multi)
sharpe<-sqrt(250)*mean(perf_weight_long_short,na.rm=T)/sqrt(var(perf_weight_long_short,na.rm=T))
plot(cumsum(perf_weight_long_short),main=paste("Long/short: sharpe=",round(sharpe,3),sep=""))


