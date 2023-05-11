


transform_OLPD_back_original_data_func<-function(data_xts,data_mat,OLPD_scaled_mat,lm_obj,data)
{
  # Make xts-object (not trivial in this case because of monthly dates...)
  OLPD_mat<-data_xts
  for (i in 1:nrow(OLPD_scaled_mat))
    OLPD_mat[i,]<-OLPD_scaled_mat[i,]
  OLPD_scaled_mat<-OLPD_mat
  is.xts(OLPD_mat)
  colnames(OLPD_mat)<-c("intercept",colnames(data_xts)[2:ncol(data_xts)])

  # Transform back to original log-returns: the regression weights are not affected in this case because target and explanatory are scaled by the same constant: we nevertheless apply the (identity) scaling to be able to work in more general settings
  for (j in 2:ncol(OLPD_mat))
    OLPD_mat[,j]<- OLPD_scaled_mat[,j]*(max(data_mat[,1])-min(data_mat[,1]))/(max(data_mat[,j])-min(data_mat[,j]))
  # The intercept is affected
  #   -We center the intercept: variations about its mean value
  #   -We scale these variations: divide by scale of transformed and multiply by scale of log-returns
  #   -Add intercept from original regression
  OLPD_mat[,1]<-(OLPD_scaled_mat[,1]-mean(OLPD_scaled_mat[,1],na.rm=T))*((max(data_mat[,1])-min(data_mat[,1]))/(max(data[,1])-min(data[,1]))) +lm_obj$coefficients[1]

  return(list(OLPD_mat=OLPD_mat,OLPD_scaled_mat=OLPD_scaled_mat))
}




OLPD_func<-function(x,delta,nn)
{
  try_data_list<-try(out_original<-predict(nn,x),silent=T)

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
      out_i <-predict(nn,y)

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
    # Virtual intercept: output of neural net minus linear regression part
    virt_int<-out_original-as.double(x%*%effect)
    effect<-c(virt_int,effect)


    # Fit the regression to the noiseless perturbated data: as many observations as unknowns i.e. zero-residual
    return(list(effect=effect))
  }
}





#number_neurons<-neuron_vec
estimate_nn<-function(train_set,number_neurons,data_mat,test_set,f)
{
  nn <- neuralnet(f,data=train_set,hidden=number_neurons,linear.output=F)


  # In sample performance
  predicted_scaled_in_sample<-nn$net.result[[1]]
  # Scale back from interval [0,1] to original log-returns
  predicted_nn_in_sample<-predicted_scaled_in_sample*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
  # In-sample MSE
  MSE.in.nn<-mean(((train_set[,1]-predicted_scaled_in_sample)*(max(data_mat[,1])-min(data_mat[,1])))^2)

  # Out-of-sample performance
  # Compute out-of-sample forecasts
  if (nrow(test_set)>1)
  {
    pr.nn <- predict(nn,as.matrix(test_set[,2:ncol(test_set)]))
    predicted_scaled<-pr.nn
  # Results from NN are normalized (scaled)
  # Descaling for comparison
    predicted_nn <- predicted_scaled*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
    test.r <- test_set[,1]*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
  # Calculating MSE
    MSE.out.nn <- mean((test.r - predicted_nn)^2)
  } else
  {
    MSE.out.nn<-predicted_nn<-NULL
  }
  # Compare in-sample and out-of-sample
  MSE_nn<-c(MSE.in.nn,MSE.out.nn)
  return(list(MSE_nn=MSE_nn,predicted_nn=predicted_nn,predicted_nn_in_sample=predicted_nn_in_sample,nn=nn))

}



trading_func<-function(predicted_mat,target_out,target_in,period_sharpe,long_only,use_in_samp)
{

  sharpe_vec<-rep(NA,ncol(predicted_mat))
  for (i in 1:ncol(predicted_mat))#i<-1
  {


    # Note that loss and mean-squared error are identical in our case (since we selected MSE as performance measure)
    #   In-sample (training) and out-of-sample (validation) MSEs
    #   Slight overfitting visible for epochs>100
    # Go long or short depending on sign of forecast
    #   We do not need to lag the signal here since the forecast is based on (already) lagged data
    if (use_in_samp)
    {
      target_trade<-target_in
    } else
    {
      target_trade<-target_out
    }
    if (long_only)
    {
      perf<-((sign(predicted_mat[,i])+1)/2)*target_trade
    } else
    {
      perf<-sign(predicted_mat[,i])*target_trade
    }
    if (i==1)
    {
      perf_mat<-perf
    } else
    {
      perf_mat<-cbind(perf_mat,perf)
    }
    sharpe_vec[i]<-sqrt(period_sharpe)*mean(perf,na.rm=T)/sqrt(var(perf,na.rm=T))

  }

  return(list(perf_mat=perf_mat,sharpe_vec=sharpe_vec))
}








OLPD_calculate_func<-function(in_out_sample_separator,data_mat,use_in_samp,x_level,ret,neuron_vec,select_exp_vec,sharpe_period,quant_th,compute_results,anz_real,asset_sel,long_only)
{



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

  lm_obj<-lm(target_in~explanatory_in)

  summary(lm_obj)




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
  train_set_xts <- scaled[paste("/",in_out_sample_separator,sep=""),]
  test_set_xts <- scaled[paste(in_out_sample_separator,"/",sep=""),]
  index(train_set_xts)

  train_set<-as.matrix(train_set_xts)
  test_set<-as.matrix(test_set_xts)
  #-----------------------------------
  # 4.c

  colnames(train_set)<-paste("lag",0:(ncol(train_set)-1),sep="")





  # Original linear parameter data
  #   Generate new data from original data
  #   New data: in each time point compute the parameters of the exact infinitesimal linear regression model






  n <- colnames(train_set)
  # Model: target is current bitcoin, all other variables are explanatory
  f <- as.formula(paste("lag0 ~", paste(n[!n %in% "lag0"], collapse = " + ")))

  tail(train_set)

  set.seed(1)


  nn <- neuralnet(f,data=train_set,hidden=neuron_vec,linear.output=F)

  #plot(nn)


  # Induce infinitesimal perturbations to data and fit regression to output

  delta<-1.e-5


  if (use_in_samp)
  {
  # Smoother (in-sample data)
    data<-train_set
    data_xts<-train_set_xts
  } else
  {
  # Rougher (out-sample data)
    data<-test_set
    data_xts<-test_set_xts
  }

  pb <- txtProgressBar(min = 1, max = (nrow(data)-1), style = 3)


  for (i in 1:(nrow(data)))
  {
    x<-matrix(data[i,2:ncol(data)],nrow=1)
    colnames(x)<-colnames(data)[2:ncol(data)]
    OLPD_scaled_obj<-OLPD_func(x,delta,nn)

    if (i==1)
    {
      OLPD_scaled_mat<-OLPD_scaled_obj$effect
    } else
    {
      OLPD_scaled_mat<-rbind(OLPD_scaled_mat,OLPD_scaled_obj$effect)
    }
    setTxtProgressBar(pb, i)

  }
  close(pb)

  OLPD_mat_obj<-transform_OLPD_back_original_data_func(data_xts,data_mat,OLPD_scaled_mat,lm_obj,data)

  OLPD_mat<-OLPD_mat_obj$OLPD_mat
  is.xts(OLPD_mat)
  index(OLPD_mat)<-index(data_xts)


  par(mfrow=c(2,1))
  plot(OLPD_mat,col=rainbow(ncol(OLPD_mat)))
  for (i in 1:ncol(OLPD_mat))
    mtext(colnames(OLPD_mat)[i],col=rainbow(ncol(OLPD_mat))[i],line=-i)
  plot(x_level[index(OLPD_mat)])


  #--------------------------------------------------------------------------------------------------
  #----------------------------------------------------------------------------------------------------



  set.seed(1)
  #neuron_vec<-c(4,2)
  # Number of random nets
  if (use_in_samp)
  {
    # Smoother (in-sample data)
    data<-train_set
    data_xts<-train_set_xts
  } else
  {
    # Rougher (out-sample data)
    data<-test_set
    data_xts<-test_set_xts
  }

  # Array of sensitivities
  arr_sens<-array(dim=c(anz_real,nrow(data),ncol(data)))
  if (anz_real>1)
    pb <- txtProgressBar(min = 1, max = anz_real, style = 3)

  neuron_name<-NULL
  for (i in 1:length(neuron_vec))
    neuron_name<-paste(neuron_name,"_",neuron_vec[i],sep="")

  if (compute_results)
  {
    # Loop over all random-nets
    predicted_mat<-NULL
    for (j in 1:anz_real)#j<-1
    {

      nn_obj<-estimate_nn(train_set,neuron_vec,data_mat,test_set,f)

      nn<-nn_obj$nn
      if (use_in_samp)
      {
        predicted_nn<-nn_obj$predicted_nn_in_sample
      } else
      {
        predicted_nn<-nn_obj$predicted_nn
      }

      predicted_mat<-cbind(predicted_mat,predicted_nn)

      for (i in 1:(nrow(data)))#i<-2
      {
  # data is out-of-sample or in-sample depending on use_in_samp=F/T
        x<-matrix(data[i,2:ncol(data)],nrow=1)
        colnames(x)<-colnames(data)[2:ncol(data)]

        OLPD_obj<-OLPD_func(x,delta,nn)

        if (i==1)
        {
          OLPD_scaled_mat<-OLPD_obj$effect
        } else
        {
  # If the neural net is NOT singular then we add the last row
          if (length(OLPD_obj$effect)>0)
          {
            OLPD_scaled_mat<-rbind(OLPD_scaled_mat,OLPD_obj$effect)
          } else
          {
  # Otherwise we duplicate last row
            OLPD_scaled_mat<-rbind(OLPD_scaled_mat,OLPD_scaled_mat[nrow(OLPD_scaled_mat),])
          }
        }

      }
      OLPD_mat_obj<-transform_OLPD_back_original_data_func(data_xts,data_mat,OLPD_scaled_mat,lm_obj,data)

      OLPD_mat<-OLPD_mat_obj$OLPD_mat
      # Replace NAs by means of columns
      for (i in 1:ncol(OLPD_mat))
        OLPD_mat[is.na(OLPD_mat[,i]),i]<-mean(OLPD_mat[,i],na.rm=T)



      OLPD_mat<-reclass(OLPD_mat,data_xts)
      arr_sens[j,,]<-OLPD_mat
      if (anz_real>1)
        setTxtProgressBar(pb, j)
    }
# Aggregates and saves are done only if more than one random net
    if (anz_real>1)
    {
      close(pb)

    # Compute aggregate/mean sensitivities


      OLPD_mat_mean<-apply(arr_sens,c(2,3),mean)
      OLPD_mat_var<-apply(arr_sens,c(2,3),var)
      save(OLPD_mat_mean,file=paste(getwd(),"/Output/OLPD_mat_mean_",asset_sel,neuron_name,sep=""))
      save(OLPD_mat_var,file=paste(getwd(),"/Output/OLPD_mat_var_",asset_sel,neuron_name,sep=""))
      save(predicted_mat,file=paste(getwd(),"/Output/predicted_mat_",asset_sel,neuron_name,sep=""))
    } else
    {
      OLPD_mat_mean<-apply(arr_sens,c(2,3),mean)
      OLPD_mat_var<-0*OLPD_mat_mean
    }
  } else
  {
    load(file=paste(getwd(),"/Output/OLPD_mat_mean_",asset_sel,neuron_name,sep=""))
    load(file=paste(getwd(),"/Output/OLPD_mat_var_",asset_sel,neuron_name,sep=""))
    load(file=paste(getwd(),"/Output/predicted_mat_",asset_sel,neuron_name,sep=""))

  }

  period_sharpe<-sharpe_period

  trading_obj<-trading_func(predicted_mat,target_out,target_in,period_sharpe,long_only,use_in_samp)

  perf_mat<-trading_obj$perf_mat
  sharpe_vec<-trading_obj$sharpe_vec
  perf_agg<-apply(perf_mat,1,mean)
  names(perf_agg)<-as.character(index(perf_mat))
  perf_agg<-as.xts(perf_agg)
  index(perf_agg)<-index(perf_mat)
  nrow(OLPD_mat_mean)
  nrow(data_xts)


  rownames(OLPD_mat_mean)<-rownames(OLPD_mat_var)<-as.character(index(data_xts))
  OLPD_mat_mean<-as.xts(OLPD_mat_mean)
  OLPD_mat_var<-as.xts(OLPD_mat_var)
  index(OLPD_mat_mean)<-index(OLPD_mat_var)<-index(data_xts)
  colnames(OLPD_mat_mean)<-colnames(OLPD_mat_var)<-c("intercept",colnames(data_xts)[2:ncol(data_xts)])

  colo<-rainbow(ncol(OLPD_mat_mean))

  par(mfrow=c(1,1))
  plot(OLPD_mat_mean,main="Mean sensitivities: partial derivative",col=colo,ylim=quantile(na.exclude(OLPD_mat_mean))[c(1,5)])
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
  for (i in 1:length(select_exp_vec))
  {
    select_exp<-select_exp_vec[i]
    mplot<-cbind(OLPD_mat_mean[,select_exp],OLPD_mat_lower[,select_exp],OLPD_mat_upper[,select_exp])
    print(plot(mplot,main=paste("Confidence interval for sensitivity of ",colnames(OLPD_mat_mean)[select_exp],sep=""),col=rep(colo[select_exp],3),lty=c(1,2,2)))
  }


  summary(lm_obj)

  apply(na.exclude(OLPD_mat_mean),2,mean)

  lm_obj$coef/apply(na.exclude(OLPD_mat_mean),2,mean)


  colo<-rainbow(ncol(OLPD_mat_lower))
  par(mfrow=c(2,2))
  # select a particular explanatory
  for (i in 1:length(select_exp_vec))
  {
    select_exp<-select_exp_vec[i]

    mplot<-cbind(OLPD_mat_mean[,select_exp])
    print(plot(mplot,ylim=c(min(na.exclude(OLPD_mat_mean)),max(na.exclude(OLPD_mat_mean))),main=paste("Confidence interval for sensitivity of ",colnames(OLPD_mat_mean)[select_exp],sep=""),col=rep(colo[select_exp],3),lty=c(1,2,2)))
  }


# Here we generate plots for two-sided quantile trespassings of LPD
  one_sided<-F

  generate_perf_plot_func_obj<-generate_perf_plot_func(select_exp_vec,OLPD_mat_mean,sharpe_period,perf_agg,x_level,ret,colo,data,one_sided,long_only)

  list_plot<-generate_perf_plot_func_obj$list_plot

# Here we generate plots for one-sided quantile trespassings of variance of LPD
  one_sided<-T

  generate_perf_plot_func_obj<-generate_perf_plot_func(select_exp_vec,OLPD_mat_var,sharpe_period,perf_agg,x_level,ret,colo,data,one_sided,long_only)

  list_plot_var<-generate_perf_plot_func_obj$list_plot

  return(list(list_plot=list_plot,list_plot_var=list_plot_var,OLPD_mat_mean=OLPD_mat_mean,OLPD_mat_var=OLPD_mat_var))

}







#OLPD_mat<-OLPD_mat_var

generate_perf_plot_func<-function(select_exp_vec,OLPD_mat,sharpe_period,perf_agg,x_level,ret,colo,data,one_sided,long_only)
{
  list_plot<-vector(mode="list")

  for (i in 1:length(select_exp_vec))#i<-4
  {
    list_plot_in_loop<-vector(mode="list")


    select_exp<-select_exp_vec[i]

    mplot<-na.exclude(cbind(OLPD_mat[,select_exp]))
    # Two-sided: use quant_th/2
    #threshold<-0.05
    # Two-sided: use quant_th/2
    if (one_sided)
    {
# We take one-sided upper quantile without median centering
      threshold_upper<-quantile(mplot,probs=1-quant_th)
      threshold_lower<--(10^99)
# Time points of trespassings
      tp_last<-index(mplot)[which(mplot<threshold_lower|mplot>threshold_upper)]
    } else
    {
# We take two-sided quantile about median centering
      threshold_upper<-quantile(mplot-median(mplot),probs=1-quant_th/2)
      threshold_lower<-quantile(mplot-median(mplot),probs=quant_th/2)
# Time points of trespassings
      tp_last<-index(mplot)[which((mplot-median(mplot))<threshold_lower|(mplot-median(mplot))>threshold_upper)]
    }
    q1<-plot(mplot,ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),
             main=paste("Confidence interval for sensitivity of ",colnames(OLPD_mat)[select_exp],sep=""),col=rep(colo[select_exp],3),lty=c(1,2,2))
    sharpe<-sqrt(sharpe_period)*mean(ret[index(mplot)],na.rm=T)/sd(ret,na.rm=T)

    # Buy and hold
    q2<-plot(x_level[index(mplot)],col=1,on=1,main=paste("Buy-and-hold, sharpe=",round(sharpe,3),sep=""))  #tail(dat$Bid)
    events<-xts(rep("",length(tp_last)),tp_last)
    #    q2<-addEventLines(events, srt=90, pos=2,col=rep(colo[select_exp],3))
    #    q2<-lines(x_level[index(mplot)])
    par(mfrow=c(2,1))
    q1
    q2

    # Active 2: Stay out of buy-and-hold if LPD beyond threshold
    list_plot_in_loop[[1]]<-q1
    list_plot_in_loop[[2]]<-q2
    mplot<-cbind(OLPD_mat[,select_exp])
    ret_perf<-ret[index(mplot)]
    # Lag for signal: 0 is OK, -1 is cheating
    lag_sig<-0
    if (long_only)
    {
      ret_perf[tp_last]<-0
    } else
    {
      ret_perf[tp_last]<--ret_perf[tp_last]
    }
    sharpe_ret<-sqrt(sharpe_period)*mean(ret_perf,na.rm=T)/sd(ret_perf,na.rm=T)
    perf<-cumsum(na.exclude(ret_perf))+as.double(x_level[index(mplot)][1])
    q4<-plot(perf,col=1,on=1,main=paste("Active 2, sharpe=",round(sharpe_ret,3),sep=""))  #tail(dat$Bid)
    events<-xts(rep("",length(tp_last)),tp_last)
    q4<-addEventLines(events, srt=90, pos=2,col=rep(colo[select_exp],3))
    q4<-lines(perf)


    # Active 1: Neural net aggregate
    perf_nn_only<-cumsum(na.exclude(perf_agg))+as.double(x_level[index(mplot)][1])
    sharpe<-sqrt(sharpe_period)*mean(perf_agg,na.rm=T)/sd(perf_agg,na.rm=T)
    q3<-plot(perf_nn_only,col=1,on=1,main=paste("Active 1 (NN), sharpe=",round(sharpe,3),sep=""))  #tail(dat$Bid)
    events<-xts(rep("",length(tp_last)),tp_last)

    #    q3<-addEventLines(events, srt=90, pos=2,col=rep(colo[select_exp],3))
    #    q3<-lines(perf)
    perf_agg_act<-perf_agg
    # Active 3: Neural net aggregate and stay out if LPF beyond quantile
    if (long_only)
    {
      perf_agg_act[tp_last]<-0
    } else
    {
      perf_agg_act[tp_last]<--perf_agg_act[tp_last]

    }
    sharpe<-sqrt(sharpe_period)*mean(perf_agg_act,na.rm=T)/sd(perf_agg_act,na.rm=T)
    perf_nn_lpd<-cumsum(na.exclude(perf_agg_act))+as.double(x_level[index(mplot)][1])

    q5<-plot(perf_nn_lpd,col=1,on=1,main=paste("Active 3, sharpe=",round(sharpe,3),sep=""))  #tail(dat$Bid)
    events<-xts(rep("",length(tp_last)),tp_last)
    q5<-addEventLines(events, srt=90, pos=2,col=rep(colo[select_exp],3))
    q5<-lines(perf_nn_lpd)
    #    q5<-lines(perf)


    par(mfrow=c(2,2))
    #    q1
    q2
    q3
    q4
    q5
    list_plot_in_loop[[3]]<-q3
    list_plot_in_loop[[4]]<-q4
    list_plot_in_loop[[5]]<-q5

    names(list_plot_in_loop)<-c("LPD","buy-and_hold","NN","LPD buy-and-hold","LPD NN")
    list_plot[[i]]<-list_plot_in_loop
  }

  names_list_plot<-NULL
  for (i in 1:length(select_exp_vec))
    names_list_plot<-c(names_list_plot,ifelse(select_exp_vec[i]==1,"intercept",colnames(data)[i]))

  names(list_plot)<-names_list_plot
  return(list(list_plot=list_plot))
}
