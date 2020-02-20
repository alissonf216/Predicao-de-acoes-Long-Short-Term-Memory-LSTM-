# coding: utf-8

# In[5]:


# Descrição: este programa usa uma rede neural recorrente artificial chamada Long Short Term Memory (LSTM)
# para prever o preço de fechamento das ações de uma emprsa ( ou Qualquer ação sa.) usando o preço das ações dos últimos 60 dias.


# In[2]:


#Importando as bibliotecas
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[38]:


# Obter cotação da bolsa
df = web.DataReader('ITSA3.SA', data_source='yahoo', start='2012-01-01', end='2020-02-05')
#motrar daods do data freame
df


# In[39]:


#Função shape apenas para mostrar quantas linhas e quantas colunas
df.shape


# In[40]:


#Plotando um grafico com os dados que eu busquei
#deixando o grafico grande e bonito
plt.figure(figsize=(18,8)) #tamanho
plt.title('Historico do preço final por Dia') #titulo
plt.plot(df['Close'] )# Pelando o valor da coluna Close e plotando
plt.xlabel('Data', fontsize=18)# definido titulo e tamanho do titulo do eixo Y
plt.ylabel('Valor em Real (R$)', fontsize=18) # definido titulo e tamanho do titulo do eixo X
plt.show()


# In[41]:


#Crie um novo quadro de dados apenas com a coluna 'close
data = df.filter(['Close'])
#Converter os dados para o padrao de matriz numpy
dataset = data.values
# Obtenha o número de linhas para treinar o modelo
training_data_len = math.ceil( len(dataset) * .8 )
training_data_len


# In[77]:


#Scale os data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data


# In[68]:


#Criar o conjunto de dados de treinamento
#Criar o conjunto de dados de treinamento em escala
train_data = scaled_data[0:training_data_len , :]
# Divida os dados nos conjuntos de dados x_train e y_train
x_train = []
y_train = []

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])
  if i<= 61:
    print(x_train)
    print(y_train)
    print()


# In[81]:


#Converta o x_train e o y_train em matrizes numpy
x_train, y_train = np.array(x_train), np.array(y_train)


# In[82]:


#Formatar os dados
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape


# In[83]:


#Construindo o modelo LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences= False))
model.add(Dense(25))
model.add(Dense(1))


# In[94]:


#Compile o modelo
model.compile(optimizer='adam', loss='mean_squared_error')


# In[95]:


# Treinando Modelo
model.fit(x_train, y_train, batch_size=1, epochs=1)


# In[96]:


#Criar o conjunto de dados de teste
#Crie uma nova matriz que contenha valores em escala do índice 1543 a 2002
test_data = scaled_data[training_data_len - 60: , :]
#Crie os conjuntos de dados x_test e y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])


# In[97]:



#Converte o array em dados numpy
x_test = np.array(x_test)


# In[98]:


#vamos formatar os dados
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))


# In[99]:


#Obter os valores de preço previstos dos modelos
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# In[106]:


#Obter o erro quadrático médio da raiz (RMSE)
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse


# In[101]:


#Plotando os dados
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#Visualize os dodos
plt.figure(figsize=(16,8))
plt.title('Modelo')
plt.xlabel('Data', fontsize=18)
plt.ylabel('Preço de Fechamando em Reais(R$)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Treino', 'Validação', 'Predição'], loc='lower right')


# In[102]:


#mostrando os dados em relação do tempo dos valores de fechamento e predição 
valid.tail()


# In[104]:


#Pegando cotação 
acoes_quote = web.DataReader('ITSA3.SA', data_source='yahoo', start='2012-01-01', end='2020-02-06')
#Criando novo  dataframe
new_df = acoes_quote.filter(['Close'])
#Obtenha os últimos preços de fechamento de 60 dias e converta o quadro de dados em uma matriz
last_60_days = new_df[-60:].values
#Escale os dados para serem valores entre 0 e 1
last_60_days_scaled = scaler.transform(last_60_days)
#Criando uma lista vazia para receber os valores
X_test = []
#Acolocand os valods dos umtimos 60 dias nessa lista
X_test.append(last_60_days_scaled)
#Converter o X_test data no ofrmando numpy array
X_test = np.array(X_test)
#Remodelar os dados
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#prcidncendo valor 
pred_price = model.predict(X_test)
#defazer escala
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)


# In[105]:


#Obter a cotação

acoes_quote2 = web.DataReader('ITSA3.SA', data_source='yahoo', start='2020-02-05', end='2020-02-07')
print(acoes_quote2['Close'])
