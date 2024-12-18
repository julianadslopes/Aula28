import pandas as pd
import polars as pl
# POLARS> http://pola.rs
# é uma biblioteca de manipulação de dados que
# tem por finalidade trabalhar com dados de volumes maiores
# Utilizando recursos de multithreading, para paralelizar 
# o processamento na memória e na cpu (utilizando os núcleos)
# Quem a faz a gestão da pseudo-distribuição é a biblioteca e não VC!
# os dados são armazenados em Dataframes Polars, que tem uma estrutura
# capaz de otimizar o processamento do dado
# Para instalar: pip install polars 

# Criando uma função customizada
# para criar uma função customizada, utilize
# a palavra reservada def
def obter_dados(endereco_arquivo, nome_arquivo, tipo_arquivo, separador):
    # coletar dados do Excel
    try: # Tente executar as ações (dentro do try)
        if tipo_arquivo == 'excel':
            df = pd.read_excel(f'{endereco_arquivo}{nome_arquivo}')
        elif tipo_arquivo == 'csv':
            #df = pd.read_csv(endereco_arquivo, sep=separador, encoding='iso-8859-1')
            df = pl.read_csv( # polars! 
                endereco_arquivo,
                separator=separador,
                encoding='Latin 1' # lembrar de ajustar conforme necessidade
            ) 
        else:
            print('Tipo de arquivo não suportado!')

        # retornar o resultado, se estiver tudo certo
        return df
    except Exception as e:
        print("Erro ao obter dados - função obter_dados: ", e)
        return None
    
def obter_dados_pd(endereco_arquivo, nome_arquivo, tipo_arquivo, separador):
    # coletar dados do Excel
    try: # Tente executar as ações (dentro do try)
        if tipo_arquivo == 'excel':
            df = pd.read_excel(f'{endereco_arquivo}{nome_arquivo}')
        elif tipo_arquivo == 'csv':
            df = pd.read_csv(endereco_arquivo, sep=separador, encoding='iso-8859-1') 
        else:
            print('Tipo de arquivo não suportado!')

        # retornar o resultado, se estiver tudo certo
        return df
    except Exception as e:
        print("Erro ao obter dados - função obter_dados: ", e)
        return None

# criando uma função para conectar no mysql
# mysql.connector: Biblioteca para conexão com o mysql
# pip install mysql.connector
import mysql.connector as mysql
def obter_dados_mysql(hostname, usuario, senha, banco, query):
    try:
        #conectando no mysql
        # variável de conexão com o banco é chamada de instância
        # de banco de dados
        conexao = mysql.connect(
            host = hostname, # endereço do servidor do banco de dados
            user = usuario, # usuário do banco de dados
            password = senha, # senha do banco de dados
            database = banco # nome do banco de dados
        )

        # obter os dados
        df = pd.read_sql(query, conexao)

        return df
    except Exception as e:
        print("Erro ao obter dados do MySQL: ", e)
        return None