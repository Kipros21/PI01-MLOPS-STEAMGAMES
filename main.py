# Funciones del Proyecto Final Individual Nº 1 - MLOps

#Importacion de Librerías
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import pandas as pd # Cargamos la libreria de "pandas" para la manipulación y el análisis de datos
import numpy as np # Cargamos la librería de "numpy" para realizar cálculos lógicos y matemáticos sobre cuadros y matrices en el caso que lo necesitemos
import uvicorn
import operator

# Instanciamos la aplicación
app = FastAPI()  #http://127.0.0.1:8000

#Cargamos DataFrames necesarios para nuestras funciones

#Establecemos las rutas de los archivos
ruta_steam_games_parquet = r'Dataset_Clean/steam_games_clean.parquet'
ruta_user_reviews_parquet = r'Dataset_Clean/australian_user_reviews_sentanaly_clean.parquet'
ruta_user_items_parquet = r'Dataset_Clean/australian_user_items_clean.parquet'
ruta_juegos_df_parquet = r'Dataset_Clean/juego_df_clean.parquet'
ruta_usuario_df_parquet = r'Dataset_Clean/usuario_df_clean.parquet'
ruta_df_recomendacion_parquet = r'Dataset_Clean/df_recomendaciones.parquet'

#Cargamos los archivos limpios luego de hacer el ETL
df_steam_games = pd.read_parquet(ruta_steam_games_parquet) #Cargamos el archivo steam_games_clean
df_user_reviews = pd.read_parquet(ruta_user_reviews_parquet) #Cargamos el archivo australian_user_reviews
df_user_items = pd.read_parquet(ruta_user_items_parquet) #Cargamos el archivo australian_user_items
df_juegos_df = pd.read_parquet(ruta_juegos_df_parquet) #Cargamos el archivo juegos_df
df_usuario_df = pd.read_parquet(ruta_usuario_df_parquet) #Cargamos el archivo usuario_df
df_recomendacion = pd.read_parquet(ruta_df_recomendacion_parquet) #Cargamos el archivo df_recomendacion

### Funciones para Alimentar la API
"""
En este proyecto, se han desarrollado funciones específicas para alimentar la API que impulsa la plataforma de recomendación de juegos en Steam.
Cada función desempeña un papel crucial en la recopilación, transformación y entrega eficiente de datos, contribuyendo así al funcionamiento integral del sistema.

"""

@app.get("/", response_class=HTMLResponse)
async def inicio():
    template = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>API Steam</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    padding: 20px;
                }
                h1 {
                    color: #333;
                    text-align: center;
                }
                p {
                    color: #666;
                    text-align: center;
                    font-size: 18px;
                    margin-top: 20px;
                }
            </style>
        </head>
        <body>
            <h1>API de consultas de la plataforma STEAM</h1>
            <p>Bienvenido a la API de STEAMGAMES, su fuente confiable para consultas especializadas sobre la plataforma de videojuegos.</p>
        </body>
    </html>
    """
    return HTMLResponse(content=template)


@app.get('/developer/{desarrollador}')
def developer(desarrollador: str):
    """
    La función developer analiza el DataFrame df_steam_games para proporcionar información sobre la cantidad
    de juegos y el porcentaje de contenido gratuito por año, según un desarrollador específico.

    Parameters:
    ----------
    desarrollador (str): Nombre del desarrollador.

    Returns:
    ----------
    Un DataFrame con las columnas 'Año', 'Cantidad de Items' y 'Contenido Free'.

    Pasos:
    ----------
    1. Filtrar las fechas con "dato sin especificar".
    2. Seleccionar juegos gratuitos del desarrollador y agruparlos por año.
    3. Obtener el total de juegos del desarrollador y agruparlos por año.
    4. Calcular el porcentaje de juegos gratuitos respecto al total por año y crear un DataFrame resultante.

    Ejemplo:
    
    desarrollador_ejemplo = "NombreDelDesarrollador"
    resultado = developer(desarrollador_ejemplo)
    print(resultado)
    
    """

    # Paso 1
    df_filtered = df_steam_games[df_steam_games['release_date'] != "dato sin especificar"]

    # Paso 2
    free_games = df_filtered[(df_filtered['developer'] == desarrollador) & (df_filtered['price'] == 0)].drop_duplicates(subset=['id'])
    free_games = free_games.groupby('release_date')['app_name'].nunique()

    # Paso 3
    all_games = df_filtered[df_filtered.developer == desarrollador].groupby('release_date')['app_name'].nunique()

    # Combinar las series para asegurarse de que todos los años estén presentes en ambas
    combined = pd.merge(all_games, free_games, how='outer', left_index=True, right_index=True, suffixes=('_total', '_free'))

    # Paso 4
    combined['Contenido Free_%'] = combined['app_name_free'] / combined['app_name_total']

    # Crear el DataFrame resultante
    result = combined.reset_index().rename(columns={'release_date': 'Año', 'app_name_total': 'Cantidad de Items'}).fillna(0)

    return result.to_dict()

@app.get("/userdata/{User_id}")
def userdata(User_id: str):
  """
  La función recibirá un id de usuario y aplicándolo devolverá la cantidad de
  dinero gastado junto al porcentaje de recomendaciones.

  Parameters:
  ----------
  - User_id (str): El identificador del usuario.

  Returns:
  ----------
  Un diccionario con la información del usuario incluyendo el dinero gastado,
  el porcentaje de recomendaciones y la cantidad de items.

  Steps:
  ----------
  1. Con el User_id, se buscan los juegos del usuario en df_user_items y se guardan en 'games'.
  2. Se calcula la cantidad de dinero gastado basado en la columna 'price' de los juegos.
  3. Se cuenta la cantidad total de juegos del usuario.
  4. Se suman las recomendaciones totales del usuario.
  5. Se calcula el porcentaje de recomendaciones.
  6. Se crea el diccionario.

  """
  global df_steam_items  # Accede al DataFrame global

  # Paso 1
  games = df_user_items[df_user_items['user_id'] == User_id]['item_id']
  games = games.tolist()
  spend_money = 0.0

  # Paso 2
  for game in games:
      price_games = df_steam_games.loc[df_steam_games['id'] == game, 'price']
      if not price_games.empty:
          price = price_games.values[0]
          spend_money += float(price)

  # Paso 3
  cant_games = df_user_items[df_user_items['user_id'] == User_id].shape[0]

  # Paso 4
  recomendations = df_user_reviews['recommend'][df_user_reviews['user_id']==User_id].sum()

  # Paso 5
  ratio = recomendations / cant_games

  # Paso 6
  result = {
      "Usuario": User_id,
      "Dinero gastado": f"{spend_money} USD",
      "% de recomendación": f"{round(ratio * 100, 2)}%",
      "Cantidad de items": cant_games
  }

  return result

@app.get('/UserForGenre/{genero}')
def UserForGenre(genero: str):
  """
  La función devuelve el usuario que acumula más horas jugadas para el género dado
  y una lista de la acumulación de horas jugadas por año de lanzamiento.

  Parameters:
  ----------
  - genero (str): El género de interés para realizar el filtro.

  Returns:
  ----------
  Un diccionario con la información del usuario con más horas jugadas para el género y la acumulación
  de horas jugadas por año de lanzamiento.
  Ejemplo de retorno: {"Usuario con más horas jugadas para Género X": "us213ndjss09sdf", "Horas jugadas": [{"Año": 2013, "Horas": 203}, {"Año": 2012, "Horas": 100}, {"Año": 2011, "Horas": 23}]}

  Steps:
  ----------
  1. Combinamos los DataFrames df_user_items y df_steam_games mediante el campo 'item_id' y 'id'.
  2. Filtramos los datos combinados para el género específico.
  3. Eliminamos los datos sin especificar en el campo 'release_year'.
  4. Encuentramos el usuario con más horas jugadas para el género.
  5. Filtramos los datos del usuario con más horas jugadas para el género.
  6. Agrupamos las horas jugadas por año de lanzamiento.
  7. Creamos la lista de acumulación de horas jugadas por año en el formato especificado.

  """

  # Paso 1
  data_by_genre = df_user_items.merge(df_steam_games, how='inner', left_on='item_id', right_on='id')

  # Paso 2
  data_by_genre = data_by_genre[data_by_genre['genres'] == genero]

  if data_by_genre.empty:
      return {"Usuario con más horas jugadas para " + genero: None, "Horas jugadas": []}

  # Paso 3
  data_by_genre = data_by_genre[data_by_genre['release_date'] != 'dato sin especificar']

  # Paso 4
  top_user = data_by_genre.groupby(['user_id'])['playtime_forever'].sum().idxmax()

  # Paso 5
  user_data = data_by_genre[data_by_genre['user_id'] == top_user]

  # Paso 6
  hours_by_year = user_data.groupby(['release_date'])['playtime_forever'].sum().reset_index()

  # Paso 7
  hours_list = [{"Año": int(year), "Horas": int(hours)} for year, hours in zip(hours_by_year['release_date'], hours_by_year['playtime_forever'])]

  return {"Usuario con más horas jugadas para " + genero: top_user, "Horas jugadas": hours_list}

@app.get('/best_developer_year/{anio}')
def best_developer_year(anio: int):
  """
  La función retorna el top 3 de desarrolladores con juegos MÁS recomendados por usuarios
  para el año dado, considerando solo revisiones recomendadas y con comentarios positivos.

  Parameters:
  ----------
  - año (int): Año para el cual se desea obtener el top 3 de desarrolladores.

  Returns:
  ----------
  Una lista de diccionarios en el formato [{"Puesto {}: {}".format(i + 1, desarrollador)}]
  donde i es la posición en el top y desarrollador es el nombre del desarrollador.

  Steps:
  ----------
  1. Filtramos los juegos del año específico digitado, excluyendo "dato sin especificar".
  2. Convertimos la columna 'release_date' a tipo datetime.
  3. Filtramos los juegos por el año específico.
  4. Fusionamos los dataframes para obtener los juegos y sus recomendaciones correspondientes.
  5. Filtramos revisiones recomendadas y comentarios positivos.
  6. Contamos la cantidad de revisiones positivas para cada desarrollador.
  7. Obtenemos los 3 principales desarrolladores o el top 3 de desarrolladores.
  8. Creamos la lista de retorno en el formato especificado.

  """

  # Paso 1
  juegos_del_año = df_steam_games[(df_steam_games['release_date'] != 'dato sin especificar') & (df_steam_games['release_date'].str.isnumeric())]

  # Paso 2
  juegos_del_año['release_date'] = pd.to_numeric(juegos_del_año['release_date'], errors='coerce')

  # Paso 3
  juegos_del_año = juegos_del_año[juegos_del_año['release_date'] == int(anio)]

  # Paso 4
  df = pd.merge(juegos_del_año, df_user_reviews, left_on='id', right_on='item_id')

  # Paso 5
  df_recomendado_positivo = df[(df['recommend'] == 1) & (df['sentiment_analysis'] == 2)]

  # Paso 6
  desarrolladores_con_revisiones = df_recomendado_positivo['developer'].value_counts()

  # Paso 7
  top_3_desarrolladores = desarrolladores_con_revisiones.head(3).index.tolist()

  # Paso 8
  resultado = [{"Puesto {}: {}".format(i + 1, desarrollador)} for i, desarrollador in enumerate(top_3_desarrolladores)]

  return resultado

@app.get('/developer_reviews_analysis/{desarrolladora}')
def developer_reviews_analysis(desarrolladora: str):
  """
  Esta función retorna un diccionario con la cantidad total de registros de reseñas de usuarios
  categorizados con un análisis de sentimiento como positivo o negativo para el desarrollador dado.

  Parameters:
  ----------
  desarrolladora (str): Nombre del desarrollador para el cual se desea realizar el análisis.

  Returns:
  ----------
  Un diccionario en el formato {desarrolladora: {'Positive': cantidad_positivas, 'Negative': cantidad_negativas}}
  donde 'cantidad_positivas' y 'cantidad_negativas' son la cantidad total de reseñas positivas y negativas, respectivamente.

  Steps:
  ----------
  1. Filtramos las reseñas del desarrollador especificado.
  2. Fusionamos los dataframes para obtener las reseñas correspondientes al desarrollador.
  3. Contamos las reseñas positivas y negativas.
  4. Creamos el diccionario de retorno.
  """

  # Paso 1 y 2
  developer_reviews = df_steam_games[df_steam_games['developer'] == desarrolladora].merge(
      df_user_reviews, left_on='id', right_on='item_id', how='inner'
  )

  # Paso 3
  positive_reviews = developer_reviews[developer_reviews['sentiment_analysis'] == 2].shape[0]
  negative_reviews = developer_reviews[developer_reviews['sentiment_analysis'] == 0].shape[0]

  # Paso 4
  result_dict = {desarrolladora: {'Positive': positive_reviews, 'Negative': negative_reviews}}

  return result_dict

@app.get('/recomendacion/{juego}')
def recomendacion_juego(juego):
  '''
  Esta función muestra una lista de juegos similares a un juego dado.

  Parameters:
  ----------
  juego (str): El nombre del juego para el cual se desean encontrar juegos similares.

  Returns:
  ----------
  juegos_similares: Esta función imprime una lista de juegos 5 similares al dado.

  Pasos:
  ----------
  1. Verificamos si el juego está en el DataFrame de similitud
  2. Obtenemos la lista de juegos similares y mostrarlos
  3. Imprimimos la lista de juegos similares

  '''

  # Paso 1
  if juego not in df_juegos_df.index:
      print(f'No se encontraron juegos similares para {juego}.')
      return

  # Paso 2
  similar_juegos = df_juegos_df.sort_values(by=juego, ascending=False).index[1:6]  # Mostrar siempre los primeros 5

  # Paso 3
  juegos_similares = [item for item in similar_juegos]

  return juegos_similares       

@app.get('/recomendacion_usuario/{usuario}')
def recomendacion_usuario(usuario):
  '''
  Esta función genera una lista de los juegos más recomendados para un usuario, basándose en las calificaciones de usuarios similares.

  Parameters:
  ----------
  usuario (str): El nombre o identificador del usuario para el cual se desean generar recomendaciones.

  Returns:
  ----------
  list: Una lista de los juegos más recomendados para el usuario basado en la calificación de usuarios similares.

  Pasos:
  ----------
  1. Verificamos si el usuario está presente en las columnas de piv_norm.
  2. Obtenemos los usuarios más similares al usuario dado.
  3. Para cada usuario similar, encuentra el juego mejor calificado y lo agrega a la lista 'mejores_juegos'.
  4. Contamos cuántas veces se recomienda cada juego.
  5. Ordenamos los juegos por la frecuencia de recomendación en orden descendente.
  6. Devolvemos los 5 juegos más recomendados.


  '''
  #Inicio
  # Creamos una tabla pivote con usuarios en filas, juegos en columnas y valores de calificación
  tabla_pivote = df_recomendacion.pivot_table(index=['user_id'], columns=['item_name'], values='rating')

  # Normalizamos el dataframe 'tabla_pivote' por filas para tener valores entre 0 y 1
  piv_norm = tabla_pivote.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)), axis=1)

  # Rellenamos los valores NaN con 0 después de la normalización y transponer la tabla
  piv_norm.fillna(0, inplace=True)
  piv_norm = piv_norm.T

  # Eliminamos las columnas que contienen solo ceros o no tienen calificación
  piv_norm = piv_norm.loc[:, (piv_norm != 0).any(axis=0)]

  # Paso 1
  if usuario not in piv_norm.columns:
      print(f'No hay datos disponibles para el usuario {usuario}.')
      return

  # Paso 2
  sim_users = df_usuario_df.sort_values(by=usuario, ascending=False).index[1:11]

  mejores_juegos = []  # Lista para almacenar los juegos mejor calificados por usuarios similares
  mas_comunes = {}  # Diccionario para contar cuántas veces se recomienda cada juego

  # Paso 3
  for i in sim_users:
      max_score = piv_norm.loc[:, i].max()
      mejores_juegos.append(piv_norm[piv_norm.loc[:, i] == max_score].index.tolist())

  # Paso 4
  for i in range(len(mejores_juegos)):
      for j in mejores_juegos[i]:
          if j in mas_comunes:
              mas_comunes[j] += 1
          else:
              mas_comunes[j] = 1

  # Paso 5
  sorted_list = sorted(mas_comunes.items(), key=operator.itemgetter(1), reverse=True)

  # Paso 6
  return sorted_list[:5]