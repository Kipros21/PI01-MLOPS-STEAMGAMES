<h1 align=center> PROYECTO INDIVIDUAL N°01 - MLOps STEAM VIDEOGAMES </h1>

# **Machine Learning Operations (MLOps)**
![image](https://github.com/Kipros21/PI01-MLOPS-STEAMGAMES/assets/142346448/b59a4dd8-c588-48eb-8b0d-29f6b4025950)


## **Contexto**
![image](https://github.com/Kipros21/PI01-MLOPS-STEAMGAMES/assets/142346448/f8d06ad2-d816-4c0a-bedd-5a4cf19f3fa9)

Steam, desarrollada por Valve Corporation desde su lanzamiento en 2003, comenzó como una plataforma para actualizar automáticamente juegos de Valve. Con el tiempo, se expandió para incluir juegos de terceros, convirtiéndose en una plataforma líder en distribución digital de videojuegos. Aunque las cifras precisas de SteamSpy están limitadas desde 2018, la plataforma cuenta con más de 325 millones de usuarios y un catálogo que supera los 25,000 juegos. A pesar de las restricciones en la recopilación de estadísticas, Steam sigue siendo una fuerza influyente en la industria del gaming, destacando por su comunidad activa y diversa.

## **Contenido**

- [Descripción del Proyecto](#Descripción-del-Proyecto)
- [Objetivo](#Objetivo)
- [Esquema de Proyecto](#Esquema-de-Proyecto)
- [Analitica de Información de Proyecto](#Analitica-de-Información-de-Proyecto)
- [Problematica de Proyectos](#Problematica-del-Proyectos)
- [Desarrollo de Proyecto](#Desarrollo-de-Proyecto)
- [Video](#Presentación-de-Video)
- [Conclusiones](#Conclusiones)

## **Descripción del Proyecto: Plataforma MLOps para Steam**
Este proyecto simula el rol integral de un MLOps Engineer para la plataforma multinacional de videojuegos Steam. En esta posición, se combinarán las responsabilidades de un Data Engineer y un Data Scientist para diseñar, implementar y desplegar soluciones avanzadas de Machine Learning. Este enfoque integral permitirá a Steam aprovechar la inteligencia artificial para mejorar la experiencia del usuario y ofrecer recomendaciones más precisas y personalizadas, elevando así la calidad de servicio en la plataforma.

## **Objetivo**
Desarrollar un Producto Mínimo Viable (PMV o MVP) que consta de una API deployada en un servicio en la nube. Esta API realizará dos funciones esenciales: un análisis de sentimientos sobre los comentarios de los usuarios de los juegos y la recomendación de juegos basada en el nombre de un juego proporcionado y/o en los gustos de un usuario específico.

## **Esquema de Proyecto**
![image](https://github.com/Kipros21/PI01-MLOPS-STEAMGAMES/assets/142346448/093f94a6-b032-42c6-a847-50c5c9430ce0)

Siguiendo el esquema como referencia se solicito elaborar un sistema de recomendacion de videojuegos para usuarios,realizar las transformaciones correspondientes , feature engineering ,
desarrolloar un API y unas funciones que se consumiran en el API y se realizara el deploy correspondiente .

## **Analítica de Información de Proyecto**
En esta ocasión, se dispone de Datasets importantes para nuestro proyecto. Hemos empleado un conjunto de datos que detalla información sobre juegos en la plataforma Steam, específicamente enfocado en la comunidad australiana. Los tres conjuntos de datos asociados se encuentran almacenados en la carpeta Dataset, la cual se presenta a continuación:

- **output_steam_games.json**: alberga información detallada sobre diversos juegos en la plataforma Steam. La información incluye aspectos clave como el nombre del juego, el desarrollador, el género, 
  etiquetas asociadas y el precio.
- **australian_users_items.json**: Esta sección proporciona detalles sobre la utilización de juegos por parte de los usuarios, incluyendo información sobre la cantidad de tiempo que cada usuario ha dedicado 
  a juegos específicos en la plataforma Steam.
- **australian_users_reviews.json**: Este conjunto de datos recopila los comentarios de los usuarios acerca de los juegos en Steam, abarcando tanto sus recomendaciones como la fecha en que fueron proporcionadas.

## **Problemática del Proyectos**
Steam enfrentaba el desafío de carecer de un sistema eficaz de recomendación de videojuegos para sus usuarios. Los datos iniciales eran complejos y poco estructurados, requiriendo una intervención significativa en el ámbito de la Ingeniería de Datos. En mi rol como MLOps Engineer, asumí la responsabilidad de transformar estos datos crudos, llevando a cabo tareas de Data Engineering y desarrollando un Producto Mínimo Viable (PMV o MVP) para abordar de manera integral este problema. Además de estas funciones esenciales, también desempeñé otras responsabilidades cruciales para optimizar y potenciar la implementación del sistema de recomendación.

## **Herramientas de Proyecto**
![Python](https://img.shields.io/badge/-Python-333333?style=flat&logo=python)
![Pandas](https://img.shields.io/badge/-Pandas-333333?style=flat&logo=pandas)
![Numpy](https://img.shields.io/badge/-Numpy-333333?style=flat&logo=numpy)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-333333?style=flat&logo=matplotlib)
![Seaborn](https://img.shields.io/badge/-Seaborn-333333?style=flat&logo=seaborn)
![Scikitlearn](https://img.shields.io/badge/-Scikitlearn-333333?style=flat&logo=scikitlearn)
![FastAPI](https://img.shields.io/badge/-FastAPI-333333?style=flat&logo=fastapi)
![Render](https://img.shields.io/badge/-Railway-333333?style=flat&logo=railway)

## **Desarrollo de Proyecto**
## **Etapa ETL(Extract - Transform - Load)**
En esta fase, nos adentraremos en el preprocesamiento de los datos obtenidos. El objetivo es realizar transformaciones necesarias antes de cargarlos, preparando así el terreno para el Análisis Exploratorio de Datos (EDA) subsiguiente. Este paso es fundamental para garantizar la calidad y la utilidad de los datos durante el análisis.

- **Extracción (Extract):**
En esta fase, se recopilan los datos desde diversas fuentes. Pueden ser bases de datos, archivos CSV, API, o cualquier otro medio donde la información esté almacenada. El objetivo es obtener la información necesaria para el análisis posterior. La extracción de datos se realizo a traves del repositorio brindado [Datos Brutos](https://drive.google.com/drive/folders/1HqBG2-sUkz_R3h1dZU5F2uAzpRn7BSpj).

- **Transformación (Transform):**
Una vez que los datos se han extraído, la etapa de transformación entra en juego. Aquí, se llevan a cabo diversas operaciones para mejorar la calidad y la utilidad de los datos:
  1. Limpieza de Datos: Se identifican y corrigen posibles errores, valores nulos o duplicados.
  2. Estandarización: Se ajustan los formatos de datos para asegurar consistencia y comparabilidad.
  3. Enriquecimiento de Datos: Se pueden crear nuevas variables o combinarse datos para obtener información adicional.
  4. Filtrado: Se eliminan datos innecesarios o irrelevantes para el análisis.
  5. Normalización: Se ajustan los datos para mantener una escala consistente.

  Nota: Se llevó a cabo un análisis de sentimientos que implicó la clasificación de los comentarios según la polaridad del sentimiento. Este proceso resultó en la creación de una nueva columna denominada     
  'sentiment_analysis', donde se asignaron valores numéricos para representar la polaridad, siendo 2 para sentimientos positivos, 1 para neutrales y 0 para negativos. Este enfoque proporciona una forma estructurada de 
  entender y categorizar la actitud expresada en los comentarios de manera cuantitativa.

  El análisis ETL de los archivos brutos fueron realizados en estos Notebooks:
  - [ETL steam_games](https://github.com/Kipros21/PI01-MLOPS-STEAMGAMES/blob/main/ETL_steam_games.ipynb)
  - [ETL_user_reviews](https://github.com/Kipros21/PI01-MLOPS-STEAMGAMES/blob/main/ETL_australian_user_reviews.ipynb)
  - [ETL user_items](https://github.com/Kipros21/PI01-MLOPS-STEAMGAMES/blob/main/ETL_australian_user_items.ipynb)
  - [ETL_sentiment_analysis](https://github.com/Kipros21/PI01-MLOPS-STEAMGAMES/blob/main/sentiment_analysis.ipynb)

- **Carga (Load):**
Después de transformar los datos, la etapa final es cargarlos en un repositorio destinado al análisis. Este puede ser un almacén de datos, una base de datos relacional, un data warehouse, o cualquier plataforma definida para el manejo de grandes volúmenes de información. La carga se realiza de manera eficiente para garantizar la disponibilidad y accesibilidad de los datos para su uso posterior.

  Los Dataset limpios son los siguientes:
  - [Data Limpia - steam_games](https://github.com/Kipros21/PI01-MLOPS-STEAMGAMES/blob/main/Dataset_Clean/steam_games_clean.parquet)
  - [Data Limpia - user_reviews](https://github.com/Kipros21/PI01-MLOPS-STEAMGAMES/blob/main/Dataset_Clean/australian_user_reviews_sentanaly_clean.parquet)
  - [Data Limpia - user_items](https://github.com/Kipros21/PI01-MLOPS-STEAMGAMES/blob/main/Dataset_Clean/australian_user_items_clean.parquet)
  - [Data Limpia - recomendaciones_sentimiento](https://github.com/Kipros21/PI01-MLOPS-STEAMGAMES/blob/main/Dataset_Clean/df_recomendaciones.parquet)
  - [Data Limpia - recomendacion_juegos](https://github.com/Kipros21/PI01-MLOPS-STEAMGAMES/blob/main/Dataset_Clean/juego_df_clean.parquet)
  - [Data Limpia - recomendacion_usuario](https://github.com/Kipros21/PI01-MLOPS-STEAMGAMES/blob/main/Dataset_Clean/usuario_df_clean.parquet)

## **Etapa EDA (Exploratory Data Analysis)**
En esta fase se realiza el proceso de análisis de datos donde se exploran y examinan los datos de manera inicial y general. El objetivo principal del EDA es comprender la naturaleza de los datos, identificar patrones, detectar posibles anomalías y obtener percepciones preliminares antes de aplicar métodos analíticos más avanzados. Durante el EDA, se realizan diversas operaciones, como la visualización de datos mediante gráficos, la descripción estadística de variables clave, y la identificación de relaciones entre diferentes características del conjunto de datos. Este proceso es esencial para orientar el análisis de datos subsiguiente y para tomar decisiones informadas sobre el enfoque y las técnicas que se aplicarán en el análisis más detallado.

El Notebook donde se realizo el proceso EDA es el siguiente: [Analisis Exploratorio de Datos](https://github.com/Kipros21/PI01-MLOPS-STEAMGAMES/blob/main/EDA_analisis_exploratorio_datos.ipynb)

## **Etapa Modelado de Aprendizaje Automático ML (Machine Learning)**
En esta etapa se procede con la construcción de nuestro modelo de recomendación, hemos optado por utilizar el enfoque de similitud de coseno. Este modelo se centra en identificar juegos que comparten similitudes con aquellos que ya nos gustan. La lógica subyacente radica en que encontrar juegos con una similitud coseno más alta en comparación con nuestros favoritos puede resultar más valioso, ya que sugiere una afinidad más cercana en términos de características y preferencias. Este método se ajusta a la idea de descubrir juegos que se asemejan más a nuestros gustos actuales, mejorando así la calidad de las recomendaciones.

El notebook donde se encontrará dicho modelado es el siguiente: [Modelo de Aprendizaje](https://github.com/Kipros21/PI01-MLOPS-STEAMGAMES/blob/main/fea_eng_and_ML.ipynb)

## **Etapa Desarrollo de Funciones para la API (Application Programming Interface)**
En esta etapa del proyecto, tras depurar la información, se realizaron elecciones cuidadosas de conjuntos de datos específicos para abordar cada función designada. Este enfoque estratégico se implementó con el propósito de optimizar de manera significativa las operaciones y mejorar los tiempos computacionales asociados a cada tarea. La selección precisa de datos contribuyó a una ejecución más eficiente y a resultados más efectivos en cada una de las funciones del proyecto.
- **def developer( desarrollador : str )**: Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora.
- **def userdata( User_id : str )**: Debe devolver cantidad de dinero gastado por el usuario, el porcentaje de recomendación en base a reviews.recommend y cantidad de items.
- **def UserForGenre( genero : str )**: Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.
- **def best_developer_year( año : int )**: Devuelve el top 3 de desarrolladores con juegos más recomendados por usuarios para el año dado.
- **def developer_reviews_analysis( desarrolladora : str )**: Según el desarrollador, se devuelve un diccionario con el nombre del desarrollador como llave y una lista con la cantidad total de registros de reseñas de     
  usuarios   que se encuentren categorizados con un análisis de sentimiento como valor positivo o negativo.
- **def recomendacion_juego( id de producto )**: Ingresando el id de producto, deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.
- **def recomendacion_usuario( id de usuario )**: Ingresando el id de un usuario, deberíamos recibir una lista con 5 juegos recomendados para dicho usuario.

El Notebook donde se encuentran dichas funciones es el siguiente: [Funciones para API](https://github.com/Kipros21/PI01-MLOPS-STEAMGAMES/blob/main/fea_eng_and_ML.ipynb)

## **Etapa despliegue del API**

## **FastAPI**
El código y las funciones para generar además de garantizar el correcto funcionamiento de la API se encuentra en el archivo [main.py](https://github.com/Kipros21/PI01-MLOPS-STEAMGAMES/blob/main/main.py)  Para ejecutar la API desde localHost se deben seguir los siguientes pasos:

- Clonar el proyecto haciendo `git clone https://github.com/Kipros21-PI1--MLOps_Steam_VideoGames.git`.
- Preparación del entorno de trabajo en Visual Studio Code:
    * Crear entorno `Python -m venv env`
    * Ingresar al entorno haciendo `fastapi-env\Scripts\activate`
    * Instalar dependencias con `pip install -r requirements.txt`
- Ejecutar el archivo main.py desde consola activando uvicorn. Para ello, hacer `uvicorn main:app --reload`
- Hacer Ctrl + clic sobre la dirección `http://XXX.X.X.X:XXXX` (se muestra en la consola).
- Una vez en el navegador, agregar `/docs` para acceder a ReDoc.

## **Railway**
La elección de la plataforma Railway para implementar la API se basó en su enfoque integral y su capacidad para simplificar la creación y ejecución de aplicaciones y sitios web en la nube. La funcionalidad de despliegue automático directamente desde GitHub añade eficiencia al proceso de implementación, permitiendo una integración fluida del código.

En el marco de esta implementación:

- Se creó exitosamente un nuevo servicio en railway.app, estableciendo una conexión efectiva con este repositorio.
- Para acceder a la API en funcionamiento, se proporciona el siguiente enlace: [Enlace RAILWAY](https://pi01-mlops-steamgames-production.up.railway.app/docs)
- Este despliegue en Railway garantiza la accesibilidad y disponibilidad de la API, permitiendo una experiencia óptima para los usuarios finales.

## **Presentación de Video**
Les comparto el video que resume y presenta de manera visual mi proyecto. En este material audiovisual, podrán obtener una visión completa de los aspectos clave, desde la conceptualización hasta la implementación, destacando las funcionalidades y logros alcanzados. ¡Espero que disfruten la presentación y encuentren valiosa la información compartida!

Enlace de video: [Video](https://www.youtube.com/watch?v=YP-Pz6kkw6E&ab_channel=CristhianHuanqui)

## **Conclusiones**
La ejecución de este proyecto se basó en la aplicación de los conocimientos adquiridos durante el programa de Data Science en HENRY. Las tareas abordadas durante este proceso abarcan las responsabilidades típicas tanto de un Data Engineer como de un Data Scientist.
El logro principal radica en el desarrollo exitoso de un Producto Mínimo Viable (PMV o MVP), que incluye la creación de una API y su implementación en un servicio web. Aunque hemos alcanzado con éxito el objetivo establecido, es crucial señalar que, debido a limitaciones de almacenamiento, en algunos casos se han llevado a cabo procesos básicos. Por consiguiente, existe un margen significativo para optimizar aún más las funciones empleadas, lo que podría conducir a resultados más eficientes y mejorados.
Este proyecto ha proporcionado una sólida base para futuras iteraciones y mejoras, respaldando la continuidad del desarrollo y refinamiento de las capacidades de Data Science en entornos similares.

## Autor
* **HUANQUI Tapia, Cristhian**  
    
