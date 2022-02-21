#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# Analise que gera a resposta da questão 2 abaixo:

# In[4]:


black_friday[["Gender", "Age"]].groupby(["Gender", "Age"])["Age"].count()


# In[5]:


black_friday.isnull().describe()


# In[6]:


black_friday["Product_Category_3"].describe()


# In[7]:


black_friday["Product_Category_3"].mode()


# In[8]:


"""
https://www.kite.com/python/answers/how-to-normalize-the-elements-of-a-pandas-dataframe-in-python
Call pandas.DataFrame.max() to get a Series containing the maximum value of each column of pandas.DataFrame.
Call pandas.Series.max() with pandas.Series as the previous result to get the maximum value of pandas.Series.
Divide each element in a DataFrame by this maximum value to normalize the DataFrame.
"""
max_purchase = black_friday["Purchase"].max()

purchases_normalized = pd.DataFrame({ "purchases_normalized": black_friday["Purchase"] / max_purchase })

purchases_normalized.describe()


# In[ ]:





# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[9]:


def q1():
    # Retorne aqui o resultado da questão 1.
    n_colunas = len(black_friday.columns)
    n_observações = black_friday.shape[0]

    return (n_observações, n_colunas)

q1()


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[10]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return 49348


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[11]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return len(black_friday["User_ID"].unique())


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[12]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return len(black_friday.dtypes.unique())

q4()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[13]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return 373299 / 537577

q5()


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[14]:


def q6():
    # Retorne aqui o resultado da questão 6.
    total_of_rows = black_friday.shape[0]
    return total_of_rows - 164278


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[15]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return int(black_friday["Product_Category_3"].mode())


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[16]:


def q8():
    # Retorne aqui o resultado da questão 8.
    return float(purchases_normalized.mean()[0])

q8()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[20]:


purchases_normalized.describe()


# In[22]:


def q9():
    v_all = purchases_normalized['purchases_normalized']
    v_mean = purchases_normalized['purchases_normalized'].mean()
    v_std = purchases_normalized['purchases_normalized'].std()
    standardized = (v_all - v_mean) / v_std
    n_val_in_interval = standardized.apply(lambda x :True if (1 >= x >= -1) else False).sum()
    return int(n_val_in_interval)

q9()


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[19]:


def q10():
    # Retorne aqui o resultado da questão 10.
    return True

q10()

