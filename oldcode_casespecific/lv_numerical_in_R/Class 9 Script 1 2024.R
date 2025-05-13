# For details on how to use the function ode() from package deSolve, refer to Class 7 Script 1

library(deSolve)

# Set the number of species in the community
number.of.species<-200

# This function returns the derivative of each species abundance. r is a vector and A is a matrix.
# The function %*% is R's shortcut for matrix multiplication: A%*%n is a vector whose i-th component is Sum_j A_ij * n_j
LV<-function(time,abundances,parameters){
  r<-parameters$r
  A<-parameters$A
  mu<-parameters$mu
  dndt<-abundances*(r-A%*%abundances)+mu
  dndt[abundances<0]<-0
  return(list(dndt))
}

#number of species
S<-number.of.species

#set the list of time values we'll output the numerical solution of the growth curve at
time_range<-1:5000		

#trait values
set.seed(0)				## sets the seed of the random number generator. Useful for repeatable results.
trait<-seq(0,1,length=S)		## generates S trait values between 0 and 1, in regular increasing order. 


#trait differences
d<-as.matrix(dist(trait,diag=TRUE,upper=TRUE))	## matrix with the absolute trait differences |x_i - x_j| between all pairs of species 

#initial abundances
initial_abundances<-rep(1,S)	## vector with S identical elements 1. To make them random, set them to runif(S)	

#immigration rates
mu<-rep(0,S)			## alternatives: mu<-rep(0,S); mu<-rep(.001,S); 
mu<-.001*runif(S)

#intrinsic growth rates
r<-rep(1,S)			## vector with S identical elements 1. Alternatives: r<-rep(1,S); r<-trait*(1-trait)

#competition coefficients
w<-0.15				        ## defines width of competition curve. Alternatives: w<-0.15; w<-seq(.01,.2,length=S)	
A<-abs(exp(-(d/w)^4)+1*rnorm(S^2,0,.1))	## competition matrix - monotonic function of trait differences;
## the second term adds a random perturbation to each coefficient if you want it
A[A<0]=0; A[A>1]=1                      ## set any negative competition coefficients to zero (these might arise when we add noise, but aren't meaningful), set max(A)=1 (for same reason)

times<-c(50,100,500,5000)	## times at which R will plot snapshots of the community

#now use the ODE solver in package deSolve.  
output_LV<-ode(initial_abundances,time_range,func=LV,parms=list(r=r,A=A,mu=mu))

#extract the relevant quantities
abunds1<-output_LV[times[1],-1]		## abundances of all species at a specified time
abunds2<-output_LV[times[2],-1]
abunds3<-output_LV[times[3],-1]	
abunds4<-output_LV[times[4],-1]

#set plot parameters
if(names(dev.cur())!='null device') dev.off(); dev.new(width=12,height=12)	## tells R to kill the current plot device and open a new one
par(mfrow=c(2,2),mar=c(2,2,2,2),oma=c(2,2,2,0))					## sets the margins of the new plot device

#plot final abundance by trait
plot(trait,abunds1,type="h",pch=20,las=1,main=paste("Time = ",times[1]))
plot(trait,abunds2,type="h",pch=20,las=1,main=paste("Time = ",times[2]))
plot(trait,abunds3,type="h",pch=20,las=1,main=paste("Time = ",times[3]))
plot(trait,abunds4,type="h",pch=20,las=1,main=paste("Time = ",times[4]))	
title(outer=TRUE,xlab="Trait",ylab="Abundance",main="Lotka-Volterra multiple species",line=.9,cex.lab=1.3)