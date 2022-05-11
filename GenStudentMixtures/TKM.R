trimkmeans <- function(data,k,trim=0.1, scaling=FALSE, runs=100, points=NULL,
                       countmode=runs+1, printcrit=FALSE,
                       maxit=2*nrow(as.matrix(data))){
  data <- as.matrix(data)
  n <- nrow(data)
  nin <- ceiling((1-trim)*n)
  if (scaling) data <- scale(data)
  crit <- Inf
  oldclass <- iclass <- optclass <- rep(0,n)
  disttom <- rep(0,n)
  for (i in 1:runs){
    if ((i/countmode)==round(i/countmode)) cat("Iteration ",i,"\n")
    if (is.null(points))
      means <- data[sample(n,k),,drop=FALSE]
    else
      means <- points
    wend <- FALSE
    itcounter <- 0
    while(!wend){
      itcounter <- itcounter+1
      for (j in 1:n){
        dj <- rep(0,k)
        for (l in 1:k)
          dj[l] <- sum((data[j,]-means[l,])^2)
        iclass[j] <- which.min(dj)
        disttom[j] <- min(dj)
      }
      iclass[order(disttom)[(nin+1):n]] <- 0
      if (itcounter>=maxit | identical(oldclass,iclass)) wend <- TRUE
      else{
        for (l in 1:k){
          if (sum(iclass==l)==0) means[l,] <- data[iclass==0,,drop=FALSE][1,]
          else{
            if (sum(iclass==l)>1){
              if (dim(means)[2]==1)
                means[l,] <- mean(data[iclass==l,])
              else
                means[l,] <- colMeans(data[iclass==l,])
            }
            else means[l,] <- data[iclass==l,]
          }
        }
        oldclass <- iclass
      }
    }
    newcrit <- sum(disttom[iclass>0])
    if (printcrit) cat("Iteration ",i," criterion value ",
                       newcrit/nin,"\n")
    if (newcrit<=crit){
      optclass <- iclass
      crit <- newcrit
      optmeans <- means
    }
  }
  optclass[optclass==0] <- k+1
  out <- list(classification=optclass,means=optmeans,
              criterion=crit/nin,disttom=disttom,ropt=sort(disttom)[nin],
              k=k,trim=trim,runs=runs,scaling=scaling)
  class(out) <- "tkm"
  out
}