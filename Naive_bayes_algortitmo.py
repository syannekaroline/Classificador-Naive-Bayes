import numpy as np
import pandas as pd
import math
from collections import Counter
import csv

def navie_bayes(X,dados_csv):
    #tratamendo dos dados - organização
    #separar as intâncias que se quer analisar em uma lista

    def instancias_list(instancia):
        instancia = pd.read_csv(f"{instancia}",delimiter=',',header=0)
        instancia=instancia.dropna(axis=1)
        print("\033[1;34m Instâncias Recebidas (X) :\033[m")
        print(instancia.head(),"\n","*"*30)
        instancia=instancia.to_numpy()

        array_instancias=[]
        for i in instancia:
                aux=[]
                for j in i:
                    aux.append(j.lstrip())
                array_instancias.append(aux)
        return array_instancias

    # tratar os dados
    def database(dados):

        Resultados={"probabilidade classe [P(c)]":0,"Base de dados":"","classes":""}
        #separar apenas os dados:
        database = pd.read_csv(f"{dados}",delimiter=',',header=0).to_numpy()
        Resultados["Base de dados"]=database
       
        #separar os atributos
        # array que armazena os tipos de atributos - categorias
        with open(f"{dados}",mode='r') as atributos:
            leitor=csv.reader(atributos)
            atributos=next(leitor)

        #separar as categorias
        categorias=[]
        for i in atributos:
        #considera apenas a coluna com o atributo i
            xi = pd.read_csv(f"{dados}",delimiter=',',usecols=[i],)
            xi=xi.dropna(how='all')#retira as linhas onde não há classificação (NaN)

            #guarda todas as classificações em um array
            classific_xi=xi.to_numpy()
            #guarda em um único array todos os vetores da classe(coluna)
            array_Xi=[]
            for j in classific_xi:
                array_Xi.append(j[0].lstrip())
            categorias.append(dict(Counter(array_Xi)))
        P_cs=dict()
        classes=[]
        for k , v in categorias[len(categorias)-1].items():
            P_cs[k]=round(v/len(array_Xi),2)
            classes.append(k)
            Resultados[k]=round(v/len(array_Xi),2)
            Resultados["classes"]=classes

        return Resultados

    def prob_condicional_X_c(X,classe,dadosX):
        P=0
        database_X=dadosX["Base de dados"]
        add_one_smoothin=len(database(dados_csv)["classes"])
        # print(f"\033[1;35mClasse em análise: {classe}\033[m")
        for Xi in X:#pra cada atributo do dado observado(vetor X)
            Nc=0# conta o número de vezes que a classe aparece 
            Nxi=0#conta o número de vezes que o atributo pertemce a classe
            for array_line in database_X:
                array_line=list(array_line)

                if classe in array_line:#se a classe estiver na lista
                    Nc+=1
                if Xi in array_line and classe in array_line:
                    Nxi+=1
                
                if Nxi==0:
                    Nxi=1
                    Nc+=len(database(dados_csv)["classes"])
            Pxic=round(Nxi/Nc,2)
            P+=round(math.log(Pxic,10),2)
            print(f"\033[1;34m\nAtributo em análise : {Xi.lstrip()}\033[m")
            print(f"P({Xi} / {classe}): {Nxi/Nc:.2f}")
        return P


    ####CÁCULO DO TEOREMA DE NAIVE BAYES###

    #dados necessários :
    instancias_analise= instancias_list(X)
    probabilidade_soma=dict()
    for i in instancias_analise:
        print(f"\033[1;33m\nInstância em Análise (X): {i} \033[m")
        for classe in database(dados_csv)["classes"]:
        
            pxi_c=prob_condicional_X_c(i,classe,database(dados_csv))
            probabilidade_soma[classe]=pxi_c+database(dados_csv)[classe]
            # print(f"P({classe}) = {database(dados_csv)[classe]}") 
            C_map= max(probabilidade_soma, key=lambda key: probabilidade_soma[key])
            # print(C_map)
            print(f"\n P(X/{classe}) : {probabilidade_soma[classe]:.2f}")
        print(f"\n\033[1;34mCLASSE MAIS PROVÁVEL PARA A INSTÂNCIA : {i} : \033[1;33m{C_map} | \033[1;34mP(X/C) = {probabilidade_soma[C_map]:.2f}\033[m")
        # database(dados_csv)["Base de dados"].numpy.incert(database(dados_csv)["Base de dados"],i)
        # print(database(dados_csv)["Base de dados"])
        # print(type(database(dados_csv)["Base de dados"]))

navie_bayes("/home/syanne/Documentos/códigos/mat.concre/naive bayes/teste.csv","/home/syanne/Documentos/códigos/mat.concre/naive bayes/datase01.csv")

