#!/usr/bin/env python
# coding: utf-8

import configs as config
import pandas as pd
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier


URI = f"{config.database}://{config.user}:{config.password}@{config.host}:{config.port}/{config.database}"
engine = create_engine(URI)


# Uma das formas para acessar o banco é utilizando o DBeaver.
# Ele permite três tipos de modelagem, e você pode escolher
# apenas uma delas, a qual se sinta mais confortável. As três
# possíveis variáveis respostas são:
# 
# - Previsão do preço da estadia (feature ‘price’)
# 
# - Classificação do room type (feature ‘room_type’)
# 
# - Segmentação dos principais assuntos das reviews (feature review_scores_rating’)
# 
# Faça uma análise exploratória para avaliar a consistência dos
# dados e identificar possíveis variáveis que impactam sua variável
# resposta.
# Para a realização deste teste você pode utilizar o software de sua
# preferência (Python ou R), só pedimos que compartilhe conosco o
# código fonte (utilizando um repositório git). Além disso, inclua um
# arquivo README.md onde você deve cobrir as respostas para o
# ponto abaixo:
# O DESAFIO
# 
# a. Como foi a definição da sua estratégia de modelagem?


#listing = pd.read_sql_query("select * from listing limit 10000", engine)
#print(listing.shape)
#listing.to_csv("./listing.csv")
listing = pd.read_csv("./listing.csv")

#reviews = pd.read_sql_query("select * from reviews limit 10000", engine)
#print(reviews.shape)
#reviews.to_csv("./reviews.csv")
reviews = pd.read_csv("./reviews.csv")

#calendar = pd.read_sql_query("select * from calendar limit 10000", engine)
#print(calendar.shape)
#calendar.to_csv("./calendar.csv")
calendar = pd.read_csv("./calendar.csv")


# Faça uma análise exploratória para avaliar a consistência dos
# dados e identificar possíveis variáveis que impactam sua variável
# resposta.
# Para a realização deste teste você pode utilizar o software de sua
# preferência (Python ou R), só pedimos que compartilhe conosco o
# código fonte (utilizando um repositório git). Além disso, inclua um
# arquivo README.md onde você deve cobrir as respostas para o
# ponto abaixo:
# O DESAFIO
# 
# a. Como foi a definição da sua estratégia de modelagem?


#  Clean data
listing["clean_price"] = listing.price.apply(lambda x : float(x.replace("$","").replace(",","")))
listing_var = listing.loc[:,['clean_price','number_of_reviews', 'room_type']].copy()
room_type_dummies = pd.get_dummies(listing.room_type, drop_first=True)
listing_var = listing_var.merge(room_type_dummies, left_index=True, right_index=True, how="inner")
listing_var = listing_var.drop("room_type", axis=1)

# Set explained and explanatory variables
y = listing_var.clean_price
X = listing_var.loc[:,['number_of_reviews', 'Hotel room', 'Private room','Shared room']]

# Compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print(f"{str(f + 1)} feature {X.columns[indices[f]]} {str(importances[indices[f]])}")
# A quantidade de reviews é o atributo com maior capacidade preditiva entre as seleccionadas.

reg = LinearRegression().fit(X, y)
# Price prediction of a listing of a Private room com 1 review
reg.predict([[1, 0, 1, 0]])


# # SQL - 1

# EXERCÍCIO 1
# Na tabela calendar é possível encontrar a disponibilidade
# (available, onde 't' indica que está disponível e 'f' indica
# indisponibilidade) dos anúncios de acordo com o dia, o valor e o
# número de noites. Encontre o número de anúncios únicos, do
# tipo apartamento (Apartment), disponíveis por semana no mês de
# outubro de 2019.

# In[ ]:


query1 = """
select
  date_part('week', c.date::date) AS weekly,
  count( distinct l.id ) as unique_listing_id
from listing l
inner join calendar c 
  on l.id = c.listing_id 
where room_type = 'Entire home/apt'
  and c.available = 't'
  and c.date between '2019-10-01' and '2019-10-30'
group by date_part('week', c.date::date)
limit 10000
"""
s1 = pd.read_sql_query(query1, engine)
print(s1)


# # SQL 2

# EXERCÍCIO 2
# Na tabela reviews é possível encontrar as avaliações deixadas
# pelos usuários de acordo com a data, id do usuário e comentário.
# Para os anúncios disponíveis e não disponíveis por semana no
# mês de outubro de 2019, encontre o percentual deles com mais
# de 10 usuários (reviewers) únicos, menos de 10 usuários
# (reviewers) únicos ou nenhum usuário (reviewer) (para reviews
# feitos até 01/10/2019).

query2 = """
select
  n.weekly,
  n.available,
  n.cohort_review,
  count(*) as count_obs
from (
select
  t.weekly,
  t.available,
  t.listing_id,
  t.unique_reviews,
  case 
    when unique_reviews >= 10 then 'over 10'
    when (unique_reviews between 1 and 9) then 'between 1 and 9'
    else 'is 0'
    end as cohort_review
from (
select
  date_part('week', c.date::date) AS weekly,
  c.available,
  r.listing_id,
  count (distinct r.reviewer_id) as unique_reviews
from  reviews r
left join calendar c 
  on r.listing_id = c.listing_id
where c.date between '2019-10-01' and '2019-10-30'
group by date_part('week', c.date::date), c.available, r.listing_id
) t
) n
group by n.weekly, n.available, n.cohort_review
;
"""

s2 = pd.read_sql_query(query2, engine)
print(s2)