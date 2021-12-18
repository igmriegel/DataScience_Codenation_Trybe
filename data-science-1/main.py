#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[6]:





# ## Parte 1

# ### _Setup_ da parte 1

# In[4]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[5]:


# Sua análise da parte 1 começa aqui.
dataframe.quantile(q=0.25)['normal']


# In[6]:


dataframe.describe()


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[9]:


def q1():
    q1_norm = round(dataframe.quantile(q=0.25)['normal'],3)
    q2_norm = round(dataframe.quantile(q=0.5)['normal'],3)
    q3_norm = round(dataframe.quantile(q=0.75)['normal'],3)
    q1_binom = round(dataframe.quantile(q=0.25)['binomial'],3)
    q2_binom = round(dataframe.quantile(q=0.5)['binomial'],3)
    q3_binom = round(dataframe.quantile(q=0.75)['binomial'],3)

    # Retorne aqui o resultado da questão 1.
    return (round(q1_norm - q1_binom, 3), round(q2_norm - q2_binom, 3), round(q3_norm - q3_binom, 3))

print(q1())


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[198]:


def q2():
#  código original
    # mean = dataframe['normal'].mean()
    # std = dataframe['normal'].std()
    # print(mean)
    # print(std)

    # p_normal_x_minus_std = sct.norm.cdf(mean - std, loc=round(mean), scale=round(std))
    # p_normal_x_plus_std = sct.norm.cdf(mean + std,  loc=round(mean), scale=round(std))


    # # Retorne aqui o resultado da questão 2.
    # return np.round_(p_normal_x_plus_std - p_normal_x_minus_std, 3)

#  código produzido pelo Lucca na mentoria
    mean = dataframe['normal'].mean()
    std = dataframe['normal'].std()
    lower_interval = mean - std
    upper_interval = mean + std
    emp_cdf = ECDF(dataframe['normal'])
    lower_cdf = emp_cdf(lower_interval)
    upper_cdf = emp_cdf(upper_interval)

    return round(upper_cdf - lower_cdf, 3)


print(q2())


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[10]:


def q3():
    m_binom = dataframe["binomial"].mean()
    v_binom = dataframe["binomial"].var()
    m_norm = dataframe["normal"].mean()
    v_norm = dataframe["normal"].var()

    # Retorne aqui o resultado da questão 3.
    return ( round(m_binom - m_norm, 3), round(v_binom - v_norm, 3) )

print(q3())


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[14]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# In[49]:


stars.shape


# ## Inicie sua análise da parte 2 a partir daqui

# In[182]:


# Sua análise da parte 2 começa aqui.
stars_target_false = stars[stars.target.eq(False)]

mean_profile_target_false = stars_target_false[["mean_profile"]]


mean_profile_target_false.describe()


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[200]:


def q4():
    mean = mean_profile_target_false["mean_profile"].mean()
    std = mean_profile_target_false["mean_profile"].std()
    # calcular Z-index
    mean_profile_target_false["standard"] = (mean_profile_target_false["mean_profile"] - mean ) / (std)

    # quantis teoricos de 0.8, 0.90 e 0.95
    q80 = sct.norm.ppf(0.80, loc=0, scale=1)
    q90 = sct.norm.ppf(0.90, loc=0, scale=1)
    q95 = sct.norm.ppf(0.95, loc=0, scale=1)

    # computada a ECDF para o os valores padronizados
    emp_cdf = ECDF(mean_profile_target_false['standard'])

    p_q80 = round(emp_cdf(q80), 3)
    p_q90 = round(emp_cdf(q90), 3)
    p_q95 = round(emp_cdf(q95), 3)
    return (p_q80, p_q90, p_q95)

print(q4())


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[204]:


def q5():
    mean = mean_profile_target_false["mean_profile"].mean()
    std = mean_profile_target_false["mean_profile"].std()
    # calcular Z-index
    mean_profile_target_false["standard"] = (mean_profile_target_false["mean_profile"] - mean ) / (std)

    q1_standard = mean_profile_target_false.quantile(q=0.25)["standard"]
    q2_standard = mean_profile_target_false.quantile(q=0.5)["standard"]
    q3_standard = mean_profile_target_false.quantile(q=0.75)["standard"]

    # quantis teoricos de 0.8, 0.90 e 0.95
    q1_theoretical = sct.norm.ppf(0.25, loc=0, scale=1)
    q2_theoretical = sct.norm.ppf(0.5, loc=0, scale=1)
    q3_theoretical = sct.norm.ppf(0.75, loc=0, scale=1)

    diff_q1 = round(q1_standard - q1_theoretical ,3)
    diff_q2 = round(q2_standard - q2_theoretical ,3)
    diff_q3 = round(q3_standard - q3_theoretical ,3)


    # Retorne aqui o resultado da questão 5.
    return (diff_q1, diff_q2, diff_q3)

print(q5())


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
