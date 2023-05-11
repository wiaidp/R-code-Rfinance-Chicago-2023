generate_data_func<-function(w_vec,sigma,weight_common_factor,len)
{
  common_factor<-rnorm(len)
  x<-weight_common_factor[1]*common_factor+rnorm(len)
  if (length(w_vec)>1)
  {
    for (i in 2:length(w_vec))
    {
      x<-cbind(x,weight_common_factor[i]*common_factor+rnorm(len))
    }
  }
  cor(x)
  y<-sigma*rnorm(len)
  for (i in 1:length(w_vec))
    y<-y+w_vec[i]*x[,i]

  x <- scale(x, center = apply(x,2,min), scale = apply(x,2,max) - apply(x,2,min))
  y <- scale(y, center = min(y), scale = max(y) - min(y))
  return(list(x=x,y=y))
}



generate_data_non_linear_ar1_func<-function(len,lags,freq)
{


  eps<-rnorm(len+lags+1)
  ar1<-cos((1:(len+lags+1))*freq*pi/len)
  y<-rep(0,len+lags+1)

  for (i in 2:(len+lags+1))
  {
    y[i]<-ar1[i]*y[i-1]+eps[i]
  }



  ts.plot(y)


  x_train<-matrix(y[1:(len)],ncol=1)
  if (lags>1)
  {
    for (i in 2:lags)
    {
      x_train<-cbind(x_train,matrix(y[i-1+1:(len)],ncol=1))
    }
  }
  y_train<-matrix(y[lags+1:len],ncol=1)

  x_train <- scale(x_train, center = apply(x_train,2,min), scale = apply(x_train,2,max) - apply(x_train,2,min))
  y_train <- scale(y_train, center = min(y_train), scale = max(y_train) - min(y_train))
  return(list(x_train=x_train,y_train=y_train,ar1=ar1))
}

