# IMPORTACIONES

import xarray as xr
import requests
import netCDF4
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import geopandas as gpd


# DEFINICION DE VARIABLES AUXILIARES

# Identificadores del producto GOES
bucket_name = 'noaa-goes19'
product_name = 'ABI-L2-LST2KMF'

# Identificadores de fecha y hora
year = 2025
start_doy = 182     # Incluido
end_doy = 185       # Incluido 212
hour = 9

# Extension del area de estudio
target_extent = [-69.1, -68.6, -20.5, -19.85] # [Lon W, Lon E, Lat S, Lat N]



# Inicializa el cliente del servicio de almacenamiento en la nube (Amazon S3)
# UNSIGNED permite acceder sin ingresar credenciales
s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))


# DEFINICION DE FUNCIONES

# Funcion para generar las rutas de los objetos en un bucket S3
# Adaptada de https://github.com/HamedAlemo/visualize-goes16/blob/main/visualize_GOES16_from_AWS.ipynb
def get_s3_keys(bucket, s3_client, prefix=''):
    """
    Genera rutas de objetos almacenados en una carpeta (bucket) de S3 a partir
    de un prefijo especificado.
    """
    # Definicion de un diccionario para almacenar los argumentos de la solicitud a S3.
    # Bucket y Prefix son los parámetros para buscar archivos.
    kwargs = {'Bucket': bucket, 'Prefix': prefix}

    # Inicio de un bucle infinito (para manejar respuestas superiores a 1000 objetos)
    while True:
        # Ejecucion de la solicitud al servidor usando los argumentos guardados
        # en el diccionario kwargs y almacena su respuesta (response)
        response = s3_client.list_objects_v2(**kwargs)

        # Verifica si response contiene la clave 'Contents'. Si no existe,
        # significa que la búsqueda no arrojó resultados para el prefijo
        if 'Contents' in response:
            # Iteracion sobre los diccionario de metadatos guardado en Contents
            for obj in response['Contents']:
                # Extraccion de la ruta de archivo del diccionario correspondiente
                yield obj['Key']
        
        # Manejo de la paginacion de S3 en caso de solicitudes superiores a 1000 objetos
        try:
            kwargs['ContinuationToken'] = response['NextContinuationToken']
        except KeyError:

            # Detencion del bucle while
            break

# Funcion para convertir los limites del area de estudio al sistema de referencia de GOES
def get_goes_slice_bounds(ds, extent):
    """
    Convierte los limites del area de estudio (lat/lon) en rangos de coordenadas
    x, y (radiantes) para recortar los datos GOES previo al procesamiento.
    """
    # Obtiene la altura y longitud central del sensor
    sat_height = ds['goes_imager_projection'].attrs['perspective_point_height']
    central_lon = ds['goes_imager_projection'].attrs['longitude_of_projection_origin']
    
    # Definicion del CRS geoestacionario: proyeccion de los datos GOES
    goes_crs = ccrs.Geostationary(
        central_longitude=central_lon,
        satellite_height=sat_height,
        sweep_axis='x')                     # X es el eje de barrido estandar para GOES
    
    # Definicion de un CRS en lat/lon
    plate_carree = ccrs.PlateCarree()
    
    # Desempaqueta las coordenadas del area de estudio. Extent: [LonW, LonE, LatS, LatN]
    W, E, S, N = extent

    # Define arrays que contienen las 4 esquinas del area de estudio
    lons = np.array([W, E, W, E])
    lats = np.array([S, S, N, N])
    
    # Transforma los 4 vertices lat/lon a metros proyectados en el CRS geoestacionario
    vertices = goes_crs.transform_points(plate_carree, lons, lats)

    # Convierte las coordenadas de metros a radianes (angulos de barrido)
    x_rads = vertices[:, 0] / sat_height
    y_rads = vertices[:, 1] / sat_height
    
    # Compara el inicio con el final para determinar la dirección de ordenamiento de los ejes X e Y
    # Eje X
    if ds.x[0] < ds.x[-1]:          
        x_slice = slice(min(x_rads), max(x_rads))       # X es creciente: el slice debe ser de min->max
    else:
        x_slice = slice(max(x_rads), min(x_rads))       # X es decreciente el slice debe ser de max->min
    # Eje Y
    if ds.y[0] > ds.y[-1]:
        y_slice = slice(max(y_rads), min(y_rads))       # Y es decreciente el slice debe ser de max->min
    else:
        y_slice = slice(min(y_rads), max(y_rads))       # Y es creciente: el slice debe ser de min->max
        
    return x_slice, y_slice


# --- 3. PROCESAMIENTO PRINCIPAL ---

daily_grids = []

print(f"Iniciando análisis para días {start_doy} a {end_doy} del año {year}...")

for doy in range(start_doy, end_doy + 1):
    # Construir prefijo de búsqueda
    # Estructura: Product / Year / Day / Hour / File
    prefix = f'{product_name}/{year}/{doy:03d}/{hour:02d}/'
    
    # Obtener la primera clave que coincida (asumimos una imagen por hora para el ejemplo)
    keys = list(get_s3_keys(bucket_name, s3_client, prefix=prefix))
    
    if not keys:
        print(f"Día {doy}: No se encontró archivo.")
        continue
    
    # Tomamos el primer archivo de esa hora (generalmente minutos 00 a 10)
    key = keys[0] 
    file_name = key.split('/')[-1]
    
    try:
        # A. DESCARGA A MEMORIA
        print(f"Procesando día {doy}: {file_name} ...", end="\r")
        resp = requests.get(f'https://{bucket_name}.s3.amazonaws.com/{key}')
        
        # B. APERTURA CON NETCDF4 + XARRAY
        nc4_ds = netCDF4.Dataset(file_name, memory=resp.content)
        store = xr.backends.NetCDF4DataStore(nc4_ds)
        ds = xr.open_dataset(store)
        
        # C. CÁLCULO DE LIMITES (Solo en la primera iteración para eficiencia)
        # Asumimos que la grilla no cambia drásticamente entre días
        if 'x_slice' not in locals():
            x_slice, y_slice = get_goes_slice_bounds(ds, target_extent)
            
        # D. RECORTE (SUBSETTING)
        # Seleccionamos solo el área de interés. 
        # Esto reduce los datos de 5000x5000 pixeles a unos pocos cientos.
        ds_subset = ds['LST'].sel(x=x_slice, y=y_slice)

        # Validacion de seguridad
        if doy == start_doy:
             print(f"   -> Dataset Recortado Shape: {ds_subset.shape}")
        
        if ds_subset.size == 0:
            print(f"⚠️ ADVERTENCIA: Recorte vacío para día {doy}. Revisa las coordenadas impresas arriba.")
            ds.close()
            continue
        
        # E. CÁLCULO ESTADÍSTICO
       # Convertir a Celsius
        data_celsius = ds_subset - 273.15
        
        # Asignar una coordenada de tiempo al array para poder concatenar después
        # Usamos la fecha del día actual
        current_time = pd.to_datetime(f'{year}-{doy}', format='%Y-%j')
        data_celsius = data_celsius.expand_dims(time=[current_time])
        
        # .load() fuerza a leer los datos reales ahora mismo y romper el vínculo 
        # con el archivo netcdf original.
        data_celsius.load()

        # Guardar la matriz 2D completa en la lista
        daily_grids.append(data_celsius)
        
        # Limpieza
        ds.close()
        
    except Exception as e:
        print(f"\nError en día {doy}: {e}")

print("\n--- Análisis Finalizado ---")

# --- 4. VISUALIZACIÓN DE RESULTADOS ---

if daily_grids:
    print("\nCalculando el mapa promedio mensual...")
    
    # 1. Concatenar todas las matrices diarias a lo largo del eje 'time'
    # Esto crea un cubo de datos: (Tiempo: ~30, y: ~N, x: ~M)
    ds_combined = xr.concat(daily_grids, dim='time')
    
    # 2. Calcular la media temporal píxel por píxel
    # skipna=True es crucial: si un día hubo nubes (NaN), se ignora y promedia el resto
    mean_pixel_map = ds_combined.mean(dim='time', skipna=True)
    
    # --- VISUALIZACIÓN ---
    print("Generando mapa...")
    
    # Recuperamos la altura 'h' del último dataset abierto o definimos la estándar
    # (GOES-16/18/19 usualmente usan la misma altura nominal)
    h = ds['goes_imager_projection'].attrs['perspective_point_height']
    
    # Definir proyección para Cartopy
    geo_crs = ccrs.Geostationary(central_longitude=-75.0, satellite_height=h)
    
    # Calcular coordenadas en metros para el gráfico
    # mean_pixel_map mantiene las coordenadas x, y originales en radianes
    x_mesh = mean_pixel_map['x'] * h
    y_mesh = mean_pixel_map['y'] * h
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=geo_crs)
    
    # Establecer la extensión del mapa (el mismo target_extent definido al inicio)
    ax.set_extent(target_extent, crs=ccrs.PlateCarree())
    
    # Dibujar bordes
    # ax.add_feature(cfeature.BORDERS, linewidth=1, edgecolor='black')
    # ax.coastlines(resolution='10m', color='black', linewidth=0.5)
    
    # Graficar el mapa promedio
    im = ax.pcolormesh(
        x_mesh, y_mesh, mean_pixel_map,
        transform=geo_crs,
        cmap='plasma', 
        vmin=-25, vmax=10
    )
    
    basin_path = '/Users/pibonacic/Documents/GitHub/poza-fria/data/cuenca_salar_huasco_dga_2009'
    gdf = gpd.read_file(basin_path)
    gdf = gdf.to_crs(epsg=4326)
    gdf.plot(
        ax=ax,                     # Dibuja sobre el eje de Cartopy ('ax')
        facecolor='none',          # Sin relleno (para ver la temperatura debajo)
        edgecolor='black',         # Color de la línea de la cuenca
        linewidth=0.8,             # Grosor de la línea
        transform=ccrs.PlateCarree() # La referencia del Shapefile es Lat/Lon
    )


    # Decoración
    plt.colorbar(im, label='LST Promedio Mensual (°C)', shrink=0.7)
    
    # Grilla
    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False,
                      linestyle='', alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()
    
    plt.title(f'Mapa de Temperatura Media Mensual (Días {start_doy}-{end_doy})\nHora fija: {hour}:00 UTC')
    plt.show()

else:
    print("No se pudieron procesar datos para generar el mapa.")