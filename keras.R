library(keras)
library(caret)

#Carrega banco de dados (din,dout)
load(db.RData)

#Carrega lista de melhores subconjuntos
load(sets.RData)

N = nrow(din)
P = ncol(din)

########### Embaralha ###########
set.seed(57); # repetibilidade
shuffleindexes = sample(N)
din <- din[shuffleindexes,]
dout <- dout[shuffleindexes]
#################################

######################## Particiona os dados ###########################
trainIndex <- createDataPartition(dout, p = .7, list = FALSE, times = 1)
db = list()
db$train = list(X = din[trainIndex,],Y = dout[trainIndex])
db$val = list(X = din[-trainIndex,],Y = dout[-trainIndex])
########################################################################

############# Normaliza de acordo com o conjunto de testes #############
dmean_out = mean(db$train$Y)
drng_out = sd(db$train$Y)

db$train$Y = (db$train$Y-dmean_out)/drng_out
db$test$Y = (db$test$Y-dmean_out)/drng_out

dmean_in = apply(db$train$X,2,mean)
drng_in = apply(db$train$X,2,sd)

for(col in colnames(db$train$X))
{
  db$train$X[,col] = (db$train$X[,col]-dmean_in[col])/drng_in[col]
  db$val$X[,col] = (db$val$X[,col]-dmean_in[col])/drng_in[col]
}
db$train$X[is.nan(db$train$X)] = 1
db$val$X[is.nan(db$val$X)] = 1

db_trimmed = list()
db_trimmed$train = list(X = db$train$X[,colunas],Y = db$train$Y)
db_trimmed$val = list(X = db$val$X[,colunas],Y = db$val$Y)
######################################################################


#### Parâmetros de treinamento ####
Nneurons = 20
b_size = 1
epochs = 50
lr = 1e-4
loss = 'logcosh'
act_fnc = "tanh"
optmizer = optimizer_adam
###################################

############### Modelo com todas as variáveis ################
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = Nneurons, 
              activation = act_fnc, 
              input_shape = ncol(db$train$X)) %>% 
  layer_dense(units = 1, activation = 'linear')
model %>% compile(
  loss = loss,
  optimizer = optmizer(lr = lr),
  metrics = c('mse')
)
history <- model %>% fit(
  db$train$X, db$train$Y, 
  epochs = epochs, batch_size = b_size, 
  validation_data = list(db$val$X,db$val$Y)
)
##############################################################



######## Modelo com apenas as variáveis selecionadas #########
model_trim <- keras_model_sequential() 
model_trim %>% 
  layer_dense(units = Nneurons, 
              activation = act_fnc, 
              input_shape = ncol(db_trimmed$train$X)) %>% 
  layer_dense(units = 1, activation = 'linear')
model_trim %>% compile(
  loss = loss,
  optimizer = optmizer(lr = lr),
  metrics = c('mse')
)
history_trim <- model_trim %>% fit(
  db_trimmed$train$X, db_trimmed$train$Y, 
  epochs = epochs, batch_size = b_size, 
  validation_data = list(db_trimmed$val$X,db_trimmed$val$Y)
)
##############################################################


########## Plota o resultado dos treinamentos ################
error_trim_v = data.frame(CVerror = history_trim$metrics$val_mean_squared_error,it = seq(1:epochs), type = "Selecionado - Validação")
error_trim_t = data.frame(CVerror = history_trim$metrics$mean_squared_error,it = seq(1:epochs), type = "Selecionado - Treinamento")

error_full_v = data.frame(CVerror = history$metrics$val_mean_squared_error,it = seq(1:epochs), type = "Completo - Validação")
error_full_t = data.frame(CVerror = history$metrics$mean_squared_error,it = seq(1:epochs), type = "Completo - Treinamento")

error_data = rbind(error_full_t,error_trim_t, error_full_v,error_trim_v)
error_data$CVerror = sqrt(error_data$CVerror)

g = ggplot(data = error_data, aes(y = CVerror, x = it,color = type))
g = g + geom_line()+geom_point()+
  xlab("Iteração")+
  ylab("Raiz do erro quadrático médio (RMSE)")+
  scale_color_discrete(name = "Conjunto de dados:")+
  labs(title=dtitle, subtitle = paste("MLP de camada única - Treinamento",Nneurons,"neurons."))
g
##############################################################
