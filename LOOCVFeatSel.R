library(MASS) #para calculo do nr. efetivo de parâmetros

#Calcula nova matriz de aniquilação ao incrementar variável
incrementM =  function(Mprev, Fnew,lambda){
  aux = Mprev %*% Fnew
  scale = as.numeric(lambda + crossprod(Fnew,aux))
  adjust = (aux%*% crossprod(Fnew,Mprev))/scale
  Mnew = Mprev - adjust  
  return(Mnew)
}

#Calcula erro de LOOCV para uma matriz de projeção M e vetor de saída Y
looMSE = function(M,y){
  N = nrow(M)
  aux = diag(N)*(diag(M)^-2) # mais rápido que diag(diag(M)^-2))
  mse = 1/N * crossprod(y,M) %*% aux %*% M %*% y  
  return(as.numeric(mse))
}

#Função principal
#Detemrina o melhor subconjunto.
#Testa todas as variáveis ou até subconjuntos de $lim$ elementos.
#Pode assumir uma matriz de aniquilação M inicial.
rankFeatures = function(X,Y, lim = -1, M = NULL,lambda = 0){  
  #Conversão das variáveis
  ##################################
  if(!is.matrix(X))
    X = as.matrix(X)

  if(!is.vector(Y))
    Y = as.vector(as.matrix(Y))
  
  if(lim == -1 || lim > ncol(X))
    lim = ncol(X)
  ##################################
  N = nrow(X)
  if(!is.matrix(M))
    M = diag(nrow = N, ncol=N)  
  selected_features = c()
  candidates = colnames(X)
  i = 0  
  while(length(candidates) && i<lim)
  {    
    best = list(name="",error=Inf,M=NULL)    
    for(feat in candidates){
      Mcand = incrementR(M,X[,feat],lambda)
      error = looMSE(Mcand,Y)      
      improvement = error<best$error      
      if(is.na(improvement)){
        print("Degeneration")
        return(-1)
      }else if(improvement){
        best$name = feat
        best$error = error
        best$M = Mcand
      }
    }    
    i = i+1
    selected_features[best$name] = best$error
    M = best$M    
    candidates = candidates[candidates != best$name]
  }
  return(selected_features)
}

#Extra: Calcula o numero efetivo de parâmetros,
#considerando regularização L2
dfreedom = function(X,lambda)
{
  if(!is.matrix(X))
    X = as.matrix(X)
  P = ncol(X)
  A = lambda*diag(P) + t(X)%*%X
  P = X%*%ginv(A)%*%t(X)
  return(sum(diag(P)))
}