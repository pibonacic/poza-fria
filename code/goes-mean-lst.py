# INFORMACION

# Calculadora de LST medio mensual con GOES-19
# Autor: Pedro Bonacic Vera
# Contacto: pibonacic@uc.cl
# Ultima actualizacion: 2025-12-15


# IMPORTACIONES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Manejo de datos geoespaciales
import xarray as xr
import netCDF4
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import geopandas as gpd

# Manejo de solicitudes a la nube
import requests
import boto3
from botocore import UNSIGNED
from botocore.config import Config


# DEFINICION DE VARIABLES AUXILIARES

# Identificadores del producto GOES
bucket_name = 'noaa-goes19'
product_name = 'ABI-L2-LST2KMF'

# Fecha y hora
year = 2025
start_doy = 182     # 1 julio
end_doy = 212       # 31 julio
hour = 9            # UTC, 5 AM hora local

# Lista de coordenadas del area de estudio
extent = [-69.2, -68.5, -20.6, -19.75]    # [W, E, S, N]

# Lista para almacenar matrices diarias
daily_grids = []

# Objetos para almacenar indices de recorte
x_slice = None
y_slice = None

# Objeto para almacenar informacion de la proyeccion de GOES
projection_info = None


# DEFINICION DE FUNCIONES

# Funcion para generar la ruta de objetos almacenados en el servidor de Amazon S3
# Adaptada de https://github.com/HamedAlemo/visualize-goes16/blob/main/visualize_GOES16_from_AWS.ipynb
def get_s3_keys(bucket, s3_client, prefix=''):
    """
    Obtiene la ruta a objetos almacenados en un bucket del servidor de Amazon S3
    a partir de un prefijo especificado.
    """
    # Diccionario con los parámetros de busqueda de archivos
    kwargs = {'Bucket': bucket, 'Prefix': prefix}

    # Inicio de un bucle infinito
    while True:
        # Solicita al servidor los objetos coincidentes con bucket y prefix y los almacena
        response = s3_client.list_objects_v2(**kwargs)

        # Si no existe Contents, significa que la busqueda no arrojo resultados 
        if 'Contents' in response:
            # Iteracion sobre los diccionarios de metadatos guardados en Contents
            for obj in response['Contents']:
                # Extraccion de la ruta de archivo
                yield obj['Key']

        # Si la solicitud arroja mas de 1000 resultados, esta linea extiende la busqueda
        try:
            kwargs['ContinuationToken'] = response['NextContinuationToken']

        # De lo contrario, se detiene el bucle 
        except KeyError:
            break

# Funcion para descargar y abrir archivos netCDF
def download_and_open(bucket_name, key, filename):
    """
    Descarga un objeto desde el servidor S3 y lo abre en la memoria RAM como un dataset de xarray
    """
    # Realiza una solicitud al servidor de Amazon S3 usando la key obtenida antes
    response = requests.get(f'https://{bucket_name}.s3.amazonaws.com/{key}')
    response.raise_for_status()     # Si la descarga falla, levanta el error

    # Abre el dataset desde la memoria RAM usando netCDF
    nc4_ds = netCDF4.Dataset(filename, memory=response.content)   # La respuesta del servidor se pasa al argumento memory

    # Crea un adaptador para que xarray interprete el objeto netCDF
    store = xr.backends.NetCDF4DataStore(nc4_ds)

    # Crea un dataset en formato xarray
    ds = xr.open_dataset(store)

    return ds

# Funcion para convertir los limites del area de estudio al crs de GOES
def get_slice_bounds(ds, extent):
    """
    Calcula los indices de recorte (slices) en coordenadas en radianes a partir
    de un area de estudio definida en coordenadas geograficas (lat/lon)
    """
    # Extrae los atributos de altura y longitud central del sensor
    sat_height = ds['goes_imager_projection'].attrs['perspective_point_height']
    central_lon = ds['goes_imager_projection'].attrs['longitude_of_projection_origin']

    # Establece el CRS de origen de los datos
    goes_crs = ccrs.Geostationary(
        central_longitude=central_lon,
        satellite_height=sat_height,
        sweep_axis='x'                  # X es el eje de barrido estandar para GOES
    )

    # Establece un CRS en coordenadas geograficas
    plate_carree = ccrs.PlateCarree()

    # Desempaqueta las coordenadas del area de estudio y las almacena en arrays
    W, E, S, N = extent
    lons = np.array([W, E, W, E])
    lats = np.array([S, S, N, N])

    # Transforma las coordenadas geograficas a planimetricas en el CRS geoestacionario
    coords_m = goes_crs.transform_points(plate_carree, lons, lats)

    # Convierte las coordenadas de metros a radianes (angulos de barrido)
    x_rads = coords_m[:, 0] / sat_height
    y_rads = coords_m[:, 1] / sat_height

    # Determina la direccion de lectura de los ejes X e Y
    # Eje X
    if ds.x[0] < ds.x[-1]:                              # Si el primer valor es menor al ultimo, el orden
        x_slice = slice(min(x_rads), max(x_rads))       # de X es creciente: el slice debe ser de min->max
    else:
        x_slice = slice(max(x_rads), min(x_rads))
    # Eje Y
    if ds.y[0] > ds.y[-1]:
        y_slice = slice(max(y_rads), min(y_rads))
    else:
        y_slice = slice(min(y_rads), max(y_rads))
        
    return x_slice, y_slice

# Funcion para procesar los archivos diarios
def process_daily_data(ds, x_slice, y_slice, year, doy):
    """
    Recorta, filtra por calidad (DQF) e incorpora la dimension temporal a un ds xarray.
    Retorna un DataArray cargado en memoria listo para concatenar.
    """

    # Recorta la imagen del dia actual usando los slices definidos
    lst_clipped = ds['LST'].sel(x=x_slice, y=y_slice)       # banda con datos de temperatura superficial
    dqf_clipped = ds['DQF'].sel(x=x_slice, y=y_slice)       # banda con info de calidad de datos

    # Filtra usando las quality flags REVISAR
    ds_filtered = lst_clipped.where(dqf_clipped == 0)       # DQF == 0 indica la mejor calidad REVISAR

    # Asigna una dimension temporal a la matriz
    current_time = pd.to_datetime(f'{year}-{doy}', format='%Y-%j')
    ds_temporal = ds_filtered.expand_dims(time=[current_time])

    # Fuerza la carga de los datos recortados a la memoria, rompiendo la dependencia con el dataset original
    ds_temporal.load()

    return ds_temporal


# PROCESAMIENTO

# Accede al servicio de Amazon S3 sin credenciales (UNSIGNED)
s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Bucle para la descarga, preprocesamiento y almacenamiento de las matrices individuales
print(f'Iniciando analisis para dias {start_doy} a {end_doy} de {year}')

# Itera para cada dia del año dentro del rango especificado
for doy in range(start_doy, end_doy + 1):

    # Construye un prefijo de busqueda de archivo. Ej: 'goes-19/2025/185/09/'
    prefix = f'{product_name}/{year}/{doy:03d}/{hour:02d}/'

    try:
        # Accede al servidor, busca la carpeta que coincide con el prefijo y almacena las rutas de todos los 
        # archivos contenidos en ella. Para el producto LST, solo existe una matriz por carpeta
        keys = list(get_s3_keys(bucket_name, s3_client, prefix=prefix))

        # Si no hay rutas en el dia actual, continua al siguiente
        if not keys:
            print(f'No se encontraron archivos el dia {doy}')
            continue

        # Accede a la primera ruta de la lista (en este caso la unica)
        key = keys[0]
        # Extrae el nombre del archivo (ultima cadena despues de /)
        filename = key.split('/')[-1]
        print(f'Procesando dia {doy}: {filename}...', end='\r')

        # Abre la matriz del dia actual como un dataset de xarray
        ds = download_and_open(bucket_name, key, filename)

        # Obtiene los limites del area de estudio (solo en la primera iteracion)
        if x_slice is None:
            # Obtiene los limites del area de estudio en radianes
            x_slice, y_slice = get_slice_bounds(ds, extent)
            # Extrae la informacion de la proyeccion de GOES
            projection_info = ds['goes_imager_projection'].load()

        # Recorta la matriz actual, la filtra por calidad y le asigna dimension temporal
        ds_processed = process_daily_data(ds, x_slice, y_slice, year, doy)

        # Guarda la matriz en la lista
        daily_grids.append(ds_processed)

        # Libera la memoria usada por ds en esta iteracion
        ds.close()

    except Exception as e:
        print(f'\nError en dia {doy}: {e}')

print("\n--- Análisis finalizado ---")

# Verifica que se hayan almacenado las matrices en la lista
if daily_grids:

    # Concatena la lista de matrices usando la dimension temporal. Crea un cubo de datos
    ds_cube = xr.concat(daily_grids, dim='time', coords='minimal', compat='override')

    # Calcula el promedio de cada pixel en el tiempo
    mean_map_K = ds_cube.mean(dim='time', skipna=True, keep_attrs=True)

    # Convierte los valores a Celsius
    mean_map_C = mean_map_K - 273.15

else:
    print('No hay datos disponibles para crear el cubo')


# VISUALIZACION

# Extrae los atributos de altura y longitud central del sensor
sat_height = projection_info.attrs['perspective_point_height']
central_lon = projection_info.attrs['longitude_of_projection_origin']

# Establece el CRS de origen de los datos
goes_crs = ccrs.Geostationary(central_longitude=central_lon, satellite_height=sat_height)

# Convierte las coordenadas de radianes a metros
x_mesh = mean_map_C['x'] * sat_height
y_mesh = mean_map_C['y'] * sat_height

# Crea el lienzo
fig = plt.figure(figsize=(10, 10))

# Crea el mapa en proyeccion geoestacionaria
ax = fig.add_subplot(1, 1, 1, projection=goes_crs)

# Pinta el mapa 
im = ax.pcolormesh(
        x_mesh, y_mesh, mean_map_C,
        transform=goes_crs,
        cmap='plasma', 
        vmin=-25, vmax=10
    )

# Personaliza la barra de leyenda
plt.colorbar(im, label='Land surface temperature (°C)', shrink=0.7)
    
# Personaliza la grilla
gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False,
                  linestyle='', alpha=0)
gl.top_labels = False
gl.right_labels = False
gl.xformatter = cticker.LongitudeFormatter()
gl.yformatter = cticker.LatitudeFormatter()

# Dibuja el titulo
plt.title(f'GOES-19 Mean LST (°C) at {hour}:00 UTC for July 2025')

# Dibuja la cuenca
basin_path = '../data/cuenca_salar_huasco_dga_2009'
basin = gpd.read_file(basin_path)
basin = basin.to_crs(epsg=4326)
basin.plot(
    ax=ax,
    facecolor='none',
    edgecolor='black',
    linewidth=0.8,
    transform=ccrs.PlateCarree()
)

# Dibuja la figura
plt.show()

# Exporta la figura
output_filename = f'GOES19_mean_LST_{year}_doy_{start_doy}_to_{end_doy}.png'
plt.savefig(
    output_filename,
    dpi=300,
    bbox_inches='tight'
)