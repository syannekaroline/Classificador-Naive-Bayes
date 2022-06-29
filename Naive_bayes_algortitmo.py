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

            #guarda em um único array todos os valores da feature
            array_Xi=[]
            for j in classific_xi:
                array_Xi.append(j[0].lstrip())
            categorias.append(dict(Counter(array_Xi)))
        P_cs=dict()#guarda a probabilidade a priori de cada classe

        #separa em uma lista as classes existentes e as guarda na lista de resultados
        classes=[]
        for k , v in categorias[len(categorias)-1].items():
            P_cs[k]=round(v/len(array_Xi),2)
            classes.append(k)
            Resultados[k]=round(v/len(array_Xi),2)
            Resultados["classes"]=classes

        #separa todas as features e suas possívies classificações - cada feature é a chave do dicionário e seu valor é uma lista de suas possíveis classificações
        features=[]
        n=0
        for feature in atributos:
            n+=1
            aux=dict()
            aux[feature]=list(categorias[n-1])
            features.append(aux)
        Resultados["features"]=features
                
        return Resultados

    #cálculo da probabilidade condicional de uma instância X dado uma classe (recebe como parâmetros a lista da instância, a string da classe e a lista com os dados de treinamento)
    def prob_condicional_X_c(X,classe,dadosX):
        P=0#armazena o  somatório dos log's das probabilidades condicionais de cada atributo em X

        #separa os dados de trinamento necessários
        database_X=dadosX["Base de dados"]
        features=dadosX["features"]

        print(f"\033[1;33m\nClasse em análise: {classe}\033[m")
        aux=0#variável auxiliar para a contagem do número de clasificações possíveis pra aquela feature ao qual o atributo pertence
        for atributo in X: #pra cada atributo do dado observado(vetor X)
            aux+=1
            V=list(features[aux-1].values())
            V=len(V[0])#Número de clasificações possíveis pra aquela feature
            Nc=V# conta o número de vezes que a classe aparece -> |v|+ somatório(Nt'c)( já inicia sendo o tamanho de V)
            Ntc=1#conta o número de vezes que o atributo aparace na classe( inicia com 1 por causa do add on smoothing)

            #cálculo das frequências
            #percorre cada vetor("linha") dos dados de treinamento contando quantas a frequência da classe e do atributo
            for array_line in database_X:
                array_line=list(array_line)
                if classe in array_line:
                    Nc+=1
                if atributo in array_line and classe in array_line:
                    Ntc+=1
            
            Pxic=round(Ntc/Nc,2)
            P+=round(math.log(Pxic,10),4)
            print(f"\033[1;34m\nAtributo em análise : {atributo.lstrip()}\033[m")
            print(f"P({atributo} / {classe}): {Ntc/Nc:.2f}")
        return P

    ####CÁCULO DO TEOREMA DE NAIVE BAYES######

    #dados necessários :
    instancias_analise= instancias_list(X)
    probabilidade_soma=dict()

    #análise de cada instância recebida
    for instancia in instancias_analise:
        print("\n","*"*100)
        print(f"\033[1;32m\nInstância em Análise (X): {instancia} \033[m")

        #Cálculo da probabilidade condicional de cada instância para cada classe - posteriori
        for classe in database(dados_csv)["classes"]:
            pxi_c=prob_condicional_X_c(instancia,classe,database(dados_csv))
            probabilidade_soma[classe]=pxi_c+database(dados_csv)[classe]
            print(f"\nProbabilidade à priori: P({classe}) = {database(dados_csv)[classe]}") 
            C_map= max(probabilidade_soma, key=lambda key: probabilidade_soma[key])
            print(f"\033[1;33m\n P(X/{classe}) : {10**probabilidade_soma[classe]:.2f}\033[m")

        #Resultado da classificação
        print(f"\n\033[1;34mCLASSE MAIS PROVÁVEL PARA A INSTÂNCIA : {instancia} : \033[1;33m{C_map} | \033[1;34mP(X/C) = Cmap = {10**probabilidade_soma[C_map]:.2f} \033[m")

navie_bayes("/home/syanne/Documentos/códigos/Naive_Bayes/Instancias_para_classificar.csv","/home/syanne/Documentos/códigos/mat.concre/naive bayes/datase01.csv")