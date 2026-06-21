import pandas as pd

df_results     = pd.read_csv("results.csv")
df_shootouts   = pd.read_csv("shootouts.csv")
df_goalscorers = pd.read_csv("goalscorers.csv")
df_former_names = pd.read_csv("former_names.csv")

def inspeccionar(df, nombre):
    print(f"--- {nombre} ---")
    print("Shape:", df.shape)
    print("Columnas:", df.columns.tolist())
    print("Dtypes:")
    print(df.dtypes)
    print("Nulos por columna:")
    print(df.isna().sum())
    print("Primeras filas:")
    print(df.head())
inspeccionar(df_results, "results")
inspeccionar(df_shootouts, "shootouts")
inspeccionar(df_goalscorers, "goalscorers")

mascara = df_results['home_score'].isna()
print(mascara.sum())   

filas_nulas = df_results[mascara]
print(filas_nulas)

print(filas_nulas['tournament'].value_counts())
print(filas_nulas['date'].min(), filas_nulas['date'].max())

df_results['date'] = pd.to_datetime(df_results['date'])
futuros = df_results[df_results['date'] > pd.Timestamp.today()]
print(futuros.shape)
print(futuros['tournament'].value_counts())

df_results['date'] = pd.to_datetime(df_results['date'])
print(df_results['date'].min(), df_results['date'].max())

df_results['decada'] = (df_results['date'].dt.year // 10) * 10
print(df_results['decada'].value_counts().sort_index())

equipos = pd.concat([df_results['home_team'], df_results['away_team']]).unique()
germany_variantes = [e for e in equipos if 'Germany' in e]
sospechosos = [e for e in equipos if any(x in e for x in ['German', 'Soviet', 'USSR', 'Yugoslav', 'Czech'])]
print(sorted(sospechosos))
print(germany_variantes)

inspeccionar(df_former_names, "former_names") 
palabras_clave = ['Germany', 'German DR','West Germany']
mascara_current = df_former_names['current'].str.contains('|'.join(palabras_clave))
mascara_former  = df_former_names['former'].str.contains('|'.join(palabras_clave))
resultado = df_former_names[mascara_current | mascara_former]
print(resultado)

k_por_torneo = {
    'default': 20,
    'FIFA World Cup': 40,
    'Friendly': 15,
}

def calcular_elo_historico(df_results, k_por_torneo, elo_inicial=1500):
    """
    df_results: dataframe ordenado cronológicamente por 'date'
    k_por_torneo: dict, ej. {"Friendly": 20, "FIFA World Cup": 40, ...}
    """
    elo_actual = {}  # equipo -> rating actual
    
    lista_elo_local_antes = []
    lista_elo_visita_antes = []
    
    for indice, fila in df_results.iterrows():
        local = fila['home_team']
        visita = fila['away_team']
        
        # 1. Obtener Elo actual de cada equipo (o elo_inicial si es debut)
        elo_local = elo_actual.get(local, elo_inicial)
        elo_visita = elo_actual.get(visita, elo_inicial)
        
        # 2. GUARDAR el Elo "antes" como feature de este partido
        lista_elo_local_antes.append(elo_local)
        lista_elo_visita_antes.append(elo_visita)
        
        # 3. Calcular E_local (probabilidad esperada de que gane el local)
        E_local = 1 / (1 + 10**((elo_visita - elo_local) / 400))
        E_visita = 1 - E_local
        
        # 4. Determinar S_local según el marcador real de esta fila
        #    S_local = 1 si home_score > away_score
        #    S_local = 0.5 si empate
        #    S_local = 0 si home_score < away_score
        if fila[fila['home_score'].notna() and fila['away_score'].notna()]:
            if fila['home_score'] > fila['away_score']:
                S_local = 1
            elif fila['home_score'] == fila['away_score']:
                S_local = 0.5
            else:
                S_local = 0

        S_visita = 1 - S_local
        
        # 5. Obtener K según el tipo de torneo de esta fila
        K = k_por_torneo.get(fila['tournament'], k_por_torneo['default'])
        
        # 6. Actualizar Elo de ambos equipos
        nuevo_elo_local = elo_local + K * (S_local - E_local)
        nuevo_elo_visita = elo_visita + K * (S_visita - E_visita)
        
        elo_actual[local] = nuevo_elo_local
        elo_actual[visita] = nuevo_elo_visita
    
    # 7. Agregar las listas como nuevas columnas del dataframe
    df_results['elo_local_antes'] = lista_elo_local_antes
    df_results['elo_visita_antes'] = lista_elo_visita_antes
    
    return df_results

print(df_results['tournament'].unique())