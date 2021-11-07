#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#pip install stocker


# In[25]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import stocker
from stocker.predict import tomorrow
import seaborn as sn


# In[3]:


Microsoft = 'MSFT'


# #### Checando as diferenças usando diferentes períodos de data

# In[6]:


error1 = tomorrow(Microsoft, years = 1)[1]
error2 = tomorrow(Microsoft, years = 2)[1]
error3 = tomorrow(Microsoft, years = 3)[1]
print('Erro usando 1 ano de data:', error1, '%')
print('Erro usando 2 anos de data:', error2, '%')
print('Erro usando 3 anos de data:', error3, '%')


# #### Aparentemente é melhor usar somente dados do último ano. Isso pode ser devido ao entendimento do modelo, se somente o comportamento mais recente esta envolvido no modelo, portanto é feita uma melhor predição.

# #### Agora vamos checar a diferença usando quantidades diferentes de dias para os passos de entradas:

# In[7]:


error1 = tomorrow(Microsoft, steps = 1)[1]
error2 = tomorrow(Microsoft, steps = 10)[1]
error3 = tomorrow(Microsoft, steps = 20)[1]
print('Erro usando 1 dia anterior de dados:', error1, '%')
print('Erro usando 10 dias anterior de dados:', error2, '%')
print('Erro usando 20 dias anterior de dados:', error3, '%')


# #### Novamente, checando que a data mais recente é a melhor opção. Isso não significa que o modelo não está levando em conta todo o comportamento durante todo o período de tempo.

# #### Agora vamos checar a diferença usando diferentes features:

# In[8]:


error1 = tomorrow(Microsoft, features=['Open'])[1]
error2 = tomorrow(Microsoft, features=['Low'])[1]
error3 = tomorrow(Microsoft, features=['High'])[1]
error4 = tomorrow(Microsoft, features=['Volume'])[1]
error5 = tomorrow(Microsoft, features=['Adj Close'])[1]
error6 = tomorrow(Microsoft, features=['Interest'])[1]
error7 = tomorrow(Microsoft, features=['Wiki_views'])[1]
error8 = tomorrow(Microsoft, features=['RSI', '%K', '%R'])[1]
error9 = tomorrow(Microsoft)[1]
error10 = tomorrow(Microsoft, features=['Open','Low','High','Volume',
                                       'Adj Close', 'Interest',
                                       'Wiki_views', 'RSI', '%K', '%R'])[1]
error11 = tomorrow(Microsoft, features=['Open', 'Low', 'High', 'Volume',
                                        'Adj Close'])
print('Error by including Open prices:',error1,'%')
print('Error by including Low prices:',error2,'%')
print('Error by including High prices:',error3,'%')
print('Error by including Volume:',error4,'%')
print('Error by including Adj Close prices:',error5,'%')
print('Error by including Interest:',error6,'%')
print('Error by including Wiki_views:',error7,'%')
print('Error by including indicators:',error8,'%')
print('Error by including only Close prices:',error9,'%')
print('Error by including all the features:',error10,'%')
print('Error by including the features from Yahoo Finance:',error11,'%')


# #### Usando somente o preço de fechamento anterior mostra um erro mais aproximado do que usando todas as features, então é uma boa opção para economizar  tempo enquanto está executando o código. De qualquer forma, é recomendado implementar vários casos com diferentes features e escolher o caso com o menor erro.

# #### Finalmente vamos checar o coeficiente de correlação de Pearson para cada feature contra os preços de fechamento

# In[9]:


from stocker.get_data import correlation


# In[27]:


corr = correlation(Microsoft, interest = True, wiki_views = True, indicators = True)


# In[30]:


corr


# In[11]:


Microsoft_prediction = stocker.predict.tomorrow('MSFT')


# In[13]:


Microsoft_prediction
# [Preço previsto, error(%), data da próx abertura]

