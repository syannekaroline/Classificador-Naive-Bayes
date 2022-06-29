import numpy as np
import pandas as pd
import math
from collections import Counter
import csv


#Função que recebe como parâmetro o caminho de um arquivo csv com as instâncias que se quer classificar (X) e o caminho do arquivo csv com os dados de treinamento (dados_csv)
def navie_bayes(X,dados_csv):
    #tratamendo dos dados - organização
    # função que retorna uma lista que armazena intâncias que se quer analisar 
    def instancias_list(instancia):
        instancia = pd.read_csv(f"{instancia}",delimiter=',',header=0)
        instancia=instancia.dropna(axis=1)
        print("\033[1;34m Instâncias Recebidas (X) :\033[m")
        print(instancia.head(),"\n","*"*30)
        instancia=instancia.to_numpy()

        #armazenamento de cada instância com tratamento dos dados (remove espaços em branco das strings)
        array_instancias=[]
        for i in instancia:
                aux=[]
                for j in i:
                    aux.append(j.lstrip())
                array_instancias.append(aux)
        return array_instancias

    # tratar os dados de treinamento e separar os resultados necessários para o cálculo
    def database(dados):

        Resultados={"Base de dados":"","classes":"","features":""}
        #separar apenas os dados de treinamento:
        database = pd.read_csv(f"{dados}",delimiter=',',header=0).to_numpy()
        Resultados["Base de dados"]=database
       
        #separar os atributos
        # array que armazena os tipos de atributos - categorias
        with open(f"{dados}",mode='r') as atributos:
            leitor=csv.reader(atributos)
            atributos=next(leitor)

        #separar as categorias/features e tratar os dados retirando os que não possuem classe 
        categorias=[]
        for feature in atributos:
        #considera apenas a coluna com o atributo i
            xi = pd.read_csv(f"{dados}",delimiter=',',usecols=[feature],)
            xi=xi.dropna(how='all')#retira as linhas onde não há classificação (NaN)

            #guarda todas as classificações em um array
            classific_xi=xi.to_numpy()
            # print(classific_xi)
            #guarda em um único array todos os vetores da 
            array_Xi=[]
            for j in classific_xi:
                array_Xi.append(j[0].lstrip())
            categorias.append(dict(Counter(array_Xi)))
            # print(i)
        # print(categorias)
        P_cs=dict()
        classes=[]
        for k , v in categorias[len(categorias)-1].items():
            P_cs[k]=round(v/len(array_Xi),2)
            classes.append(k)
            Resultados[k]=round(v/len(array_Xi),2)
            Resultados["classes"]=classes

        # for classe in Resultados["classes"]:
        #     for feature in categorias:
        #         for atributo in feature :
        #             print(classific_xi.Countet(atributo))


        Ntc=dict()

        vetor_aux=[]#vetor que armazena as instancias que pertencem a determinada classe
        for classe in Resultados["classes"]:
            vetor_aux=[]
            for instancia in Resultados["Base de dados"] :
                if classe == instancia[len(instancia)-1]:
                   vetor_aux.append(list(instancia))
            Ntc[classe]=vetor_aux
        # print(Ntc)
        features=[]
        n=0
        for feature in atributos:
            n+=1
            aux=dict()
            aux[feature]=list(categorias[n-1])
            # print(aux[feature])
            features.append(aux)
        # print(features)

        Resultados["features"]=features
        # print(Resultados["features"])
                
        return Resultados

    def prob_condicional_X_c(X,classe,dadosX):
        P=0
        database_X=dadosX["Base de dados"]
        features=dadosX["features"]
        # print(features)
    

        print(f"\033[1;33m\nClasse em análise: {classe}\033[m")
        i=0
        for Xi in X: #pra cada atributo do dado observado(vetor X)
            i+=1
            V=list(features[i-1].values())
            V=len(V[0])#Número de clasificações possíveis pra aquela feature
            Nc=V# conta o número de vezes que a classe aparece -> |v|+ somatório(Nt'c)( já inicia sendo o tamanho de V)
            # print(V)
            Ntc=1#conta o número de vezes que o atributo aparace na classe( inicia com 1 por causa do add on smoothing)
            for array_line in database_X:#percorre cada vetor("linha") dos dados de treinamento
                array_line=list(array_line)
                if classe in array_line:
                    Nc+=1
                if Xi in array_line and classe in array_line:
                    Ntc+=1

            Pxic=round(Ntc/Nc,2)
            P+=round(math.log(Pxic,10),4)
            print(f"\033[1;34m\nAtributo em análise : {Xi.lstrip()}\033[m")
            print(f"P({Xi} / {classe}): {Ntc/Nc:.2f}")
        return P


    ####CÁCULO DO TEOREMA DE NAIVE BAYES###

    #dados necessários :
    instancias_analise= instancias_list(X)
    probabilidade_soma=dict()
    for i in instancias_analise:
        print("\n","*"*100)
        print(f"\033[1;32m\nInstância em Análise (X): {i} \033[m")
        for classe in database(dados_csv)["classes"]:
        
            pxi_c=prob_condicional_X_c(i,classe,database(dados_csv))
            probabilidade_soma[classe]=pxi_c+database(dados_csv)[classe]
            # print(f"P({classe}) = {database(dados_csv)[classe]}") 
            C_map= max(probabilidade_soma, key=lambda key: probabilidade_soma[key])
            # print(C_map)
            print(f"\033[1;33m\n P(X/{classe}) : {10**probabilidade_soma[classe]:.2f}\033[m")
        print(f"\n\033[1;34mCLASSE MAIS PROVÁVEL PARA A INSTÂNCIA : {i} : \033[1;33m{C_map} | \033[1;34mP(X/C) = {10**probabilidade_soma[C_map]:.2f} \033[m")


navie_bayes("/home/syanne/Documentos/códigos/mat.concre/naive bayes/teste.csv","/home/syanne/Documentos/códigos/mat.concre/naive bayes/datase01.csv")

