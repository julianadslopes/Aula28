import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime



# Constante do Endereço dos dados
ENDERECO_DADOS = r'./dados/'


# obtendo dados do dados_bf.parquet
try:
    print('Obtendo dados....')

    hora_inicio = datetime.now()

    # Polars - Bolsa família
    df_bolsa_familia = pl.read_parquet(ENDERECO_DADOS + 'bolsa_familia.parquet')
    
    # Votação
    df_votacao = pl.read_csv(ENDERECO_DADOS + 'votacao_2022_BR.csv', separator=',', encoding='utf-8')

    # print(df_votacao)

    hora_fim = datetime.now()
    print('Dados obtidos com sucesso! Tempo de processamento: ', hora_fim - hora_inicio)

except ImportError as e:
    print('Erro ao obter dados: ',e)
    exit()


# Iniciar processamento
try:
    print('Processando dados....')

    hora_inicio = datetime.now()   

    #filtrar NR_TURNO = 2
    # filtrar NR_VOTAVEL in [13,22]
    df_votacao = df_votacao.filter(
        (pl.col('NR_TURNO') == 2) &
        (pl.col('NR_VOTAVEL').is_in([13, 22]))
    )

    #Delimitar colunas df_votacao: SG_UF, NM_VOTAVEL e QT_VOTOS
    # df_votacao = df_votacao.select(['SG_UF', 'NM_VOTAVEL', 'QT_VOTOS'])
    
    # # converter para float o valor parcela
    # df_bolsa_familia = df_bolsa_familia.with_columns(
    #     pl.col('VALOR PARCELA').str.replace(',', '.').cast(pl.Float64)
    # )

    # Memória cache é a memória de acesso RÁPIDO
    # Ativar o método StringCache, do polars
    # Isso é muito útil para filtros, cálculos em dados de larga escala
    with pl.StringCache():

        '''VOTAÇÃO'''
        # lazzy com delimitação das colunas. 
        df_votacao_lazy = df_votacao.lazy().select(['SG_UF', 'NM_VOTAVEL', 'QT_VOTOS'])

        # Os tipos de dados categóricos são mais eficientes que a string
        # O Polar, qdo o tipo de dado é categórico, 
        # cria um dicionário de índices NUMÉRICOS, que otimiza o consumo de memória
        
        # Converter string para categórico
        df_votacao_lazy = df_votacao_lazy.with_columns([
            pl.col('SG_UF').cast(pl.Categorical),
            pl.col('NM_VOTAVEL').cast(pl.Categorical)
        ])

        # Agrupar qtd de votos. Totalizar por UF e candidato
        df_votacao_uf = df_votacao_lazy.group_by(['SG_UF','NM_VOTAVEL']).agg(pl.col('QT_VOTOS').sum())

        # Coleta os dados
        df_votacao_uf = df_votacao_uf.collect()


        '''BOLSA FAMÍLIA'''
        # lazzy com delimitação das colunas. 
        # Dessa forma o df_bolsa_familia continua com todas as colunas da fonte de dados
        df_bolsa_familia_uf_lazy = df_bolsa_familia.lazy().select(['UF', 'VALOR PARCELA'])

        # converter UF para categórico
        df_bolsa_familia_uf_lazy = df_bolsa_familia_uf_lazy.with_columns(pl.col('UF').cast(pl.Categorical))

        # Totalizar o valor das parcelas por estado
        df_bolsa_familia_uf = df_bolsa_familia_uf_lazy.group_by('UF').agg(pl.col('VALOR PARCELA').sum())

        # Coletar dados
        df_bolsa_familia_uf = df_bolsa_familia_uf.collect()

        # Juntar os dois dataframes. Faz o mesmo que o merge() do Pandas
        '''JOIN DOS DATAFRAMES'''
        df_votos_bolsa_familia = df_votacao_uf.join(df_bolsa_familia_uf, left_on='SG_UF', right_on='UF')
    

    # Exibir todas as linhas de um dataframe. Evitar em dados de larga escala
    pl.Config.set_tbl_rows(-1)
    print(df_votacao_uf)

    # formatação numérica para o valor parcela
    pl.Config.set_float_precision(2)
    pl.Config.set_decimal_separator(',')
    pl.Config.set_thousands_separator('.')

    print(df_bolsa_familia_uf)

    print(df_votos_bolsa_familia)

    hora_fim = datetime.now()
    print('Dados procesados com sucesso! Tempo de processamento: ', hora_fim - hora_inicio)

except ImportError as e:
    print('Erro ao processar dados: ', e)
    exit()



# CORRELAÇÃO
try:
    print('Correlacionando dados....')

    hora_inicio = datetime.now() 

    # diconário candidato:correlacao
    dict_correlacoes = {}

    for candidato in df_votos_bolsa_familia['NM_VOTAVEL'].unique():
        #filtrar dados pelo candidato
        df_candidato = df_votos_bolsa_familia.filter(pl.col('NM_VOTAVEL') == candidato)

        #Arrays com as variáveis quantitativas do candidato filtrado
        array_votos = np.array(df_candidato['QT_VOTOS'])
        array_valor_parcela = np.array(df_candidato['VALOR PARCELA'])
        
        # Calcular o coeficiente de correlação (r)
        # o resultado é sempre em um matriz (linha x coluna)
        # todas as variáveis em linhas e todas em colunas
        correlacao = np.corrcoef(array_votos, array_valor_parcela)[0, 1]

        print(f'Correlação para {candidato}: {correlacao}')

        #adicioanr correlação ao dicionário
        dict_correlacoes[candidato] = correlacao

    hora_fim = datetime.now()
    print('Correlação realizada com sucesso! Tempo de processamento: ', hora_fim - hora_inicio)

except ImportError as e:
    print('Erro ao correlacionar dados: ', e)
    exit()


# VISUALIZAÇÃO
try:
    print('Visualizando dados....')
    hora_inicio = datetime.now()

    plt.subplots(2, 2, figsize=(17, 7))
    plt.suptitle('Votação x Bolsa Família', fontsize=16)

    #Posição 1: Ranking Lula
    plt.subplot(2, 2, 1)
    plt.title('Lula')

    df_lula = df_votos_bolsa_familia.filter(pl.col('NM_VOTAVEL') == 'LUIZ INÁCIO LULA DA SILVA')

    df_lula = df_lula.sort('QT_VOTOS', descending=True)

    plt.bar(df_lula['SG_UF'], df_lula['QT_VOTOS'])


    #Posição 2: Ranking Bolsonaro
    plt.subplot(2, 2, 2)
    plt.title('Bolsonaro')

    df_bolsonaro = df_votos_bolsa_familia.filter(pl.col('NM_VOTAVEL') == 'JAIR MESSIAS BOLSONARO')

    df_bolsonaro = df_bolsonaro.sort('QT_VOTOS', descending=True)

    plt.bar(df_bolsonaro['SG_UF'], df_bolsonaro['QT_VOTOS'])


    #Posição 3: Ranking do bolsa família por UF
    plt.subplot(2, 2, 3)
    plt.title('Valor Parcela')

    df_bolsa_familia_uf = df_bolsa_familia_uf.sort('VALOR PARCELA', descending=True)

    plt.bar(df_bolsa_familia_uf['UF'], df_bolsa_familia_uf['VALOR PARCELA'])


    #Posição 4: Correlação
    plt.subplot(2, 2, 4)
    plt.title('Correlações')

    # coordenadas do plt.text
    x = 0.2
    y = 0.6

    for candidato, correlacao in dict_correlacoes.items():
        plt.text(x, y, f'{candidato}: {correlacao}', fontsize=12)

        # reduzir 0.2 do eixo Y
        # y = y - 0.2
        y -= 0.2
    
    plt.axis('off')

    plt.tight_layout()

    hora_fim = datetime.now()
    print('Visualização realizada com sucesso! Tempo de processamento: ', hora_fim - hora_inicio)

    plt.show()
except ImportError as e:
    print('Erro ao visualizar dados: ', e)
    exit()

