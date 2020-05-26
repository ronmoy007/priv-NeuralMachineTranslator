from src.db_utils import get_data
import pandas as pd
import folium

class ShortestPath:
    """
    Clase con métodos llamados desde la clase "app.py"
    Tiene los métodos para extraer la tabla de sql y crea mapas.
    """
    def get_coordinates_list(self, ruta):
        coordenadas = []
        for nodo in ruta:
            query = "SELECT * FROM trabajo.nodos \
            WHERE nodo = %s ;" % (str(nodo))
            nodo_info = get_data(query)
            coordenadas.append([nodo_info['lat'][0], \
             nodo_info['lon'][0]])
        return coordenadas

    def create_map(self, coordenadas, map_name):
        centros = pd.DataFrame(coordenadas)
        centros.columns = ['lat', 'lon']
        #Definición de punto: Oficina de servicio
        os = centros.loc[[0], ['lat', 'lon']].values.tolist()

        cen_lon = centros.lon.mean()
        cen_lat = centros.lat.mean()
        mapi = folium.Map(location=[cen_lat,cen_lon],
                          zoom_start = 12,
                          tiles= "OpenStreetMap")
        folium.PolyLine(coordenadas + [coordenadas[0]],
                        color='red',
                        weight=2,
                        opacity=0.8).add_to(mapi)
        folium.Marker(os[0], icon=folium.Icon(color='blue', icon='cloud') ).add_to(mapi)

        mapi.save(map_name)

    def create_double_map(self, coordenadas1, coordenadas2, map_name):
        centros = pd.DataFrame(coordenadas1)
        centros.columns = ['lat', 'lon']
        #Definición de punto: Oficina de servicio
        os = centros.loc[[0], ['lat', 'lon']].values.tolist()

        cen_lon = centros.lon.mean()
        cen_lat = centros.lat.mean()
        mapi = folium.Map(location=[cen_lat,cen_lon],
                          zoom_start = 12,
                          tiles= "OpenStreetMap")


        folium.PolyLine(coordenadas1 + [coordenadas1[0]],
                        color='red',
                        weight=2,
                        opacity=0.8).add_to(mapi)
        folium.Marker(os[0], icon=folium.Icon(color='blue', icon='cloud') ).add_to(mapi)

        folium.PolyLine(coordenadas2 + [coordenadas2[0]],
                color='blue',
                weight=1.5,
                opacity=1).add_to(mapi)

        mapi.save(map_name)



    def get_results(self, algorithm, fza_ventas):

        if algorithm  ==  int(1):
            algorithm_name = 'SA'
        elif algorithm  ==  int(2):
            algorithm_name = 'PS'
        try:
            query = "SELECT * FROM trabajo.resultados WHERE algoritmo = '%s'\
             AND id_fza_ventas = %s ;" % (str(algorithm_name), str(fza_ventas))

            df = get_data(query)
            return df
        except (Exception) as error :
            print ("Error while getting path", error)




    def get_path(self, algorithm, fza_ventas, map_name):
        try:
            if algorithm  ==  int(3):
                df1 =  self.get_results(1, fza_ventas)
                df2 =  self.get_results(2, fza_ventas)

                ruta1 = df1['ruta_optima'][0].split(',')
                ruta2 = df2['ruta_optima'][0].split(',')

                coordenadas1 = self.get_coordinates_list(ruta1)
                coordenadas2 = self.get_coordinates_list(ruta1)

                self.create_double_map(coordenadas1, coordenadas2, map_name)

                distancia1 = df1['distancia'][0]
                distancia2 = df2['distancia'][0]

                if distancia1 < distancia2:
                    distancia = distancia1
                    ruta = ruta1
                else:
                    distancia = distancia2
                    ruta = ruta2

            else:
                df =  self.get_results(algorithm, fza_ventas)

                ruta = df['ruta_optima'][0].split(',')
                coordenadas = self.get_coordinates_list(ruta)

                self.create_map(coordenadas, map_name)
                distancia = df['distancia'][0]

            return distancia, ruta
        except (Exception) as error :
            print ("Error while getting path", error)

    def get_distance(self, algorithm, fza_ventas):
        try:
            if algorithm  ==  int(3):
                query = "SELECT * FROM trabajo.resultados_vw WHERE \
                 id_fza_ventas = %s ORDER BY distancia limit 1;" % (str(fza_ventas))
            else:
                if algorithm  ==  int(1):
                    algorithm_name = 'SA'
                else:
                    algorithm_name = 'PS'
                query = "SELECT * FROM trabajo.resultados_vw WHERE algoritmo = '%s'\
                 AND id_fza_ventas = %s ;" % (str(algorithm_name), str(fza_ventas))

            print(query)
            df = get_data(query)
            print(df)
            distancia = df['distancia'][0]
            ruta = df['ruta_optima'][0].split(',')
            print(ruta)

            return distancia, ruta
        except (Exception) as error :
            print ("Error while getting path", error)

#agent = ShortestPath()
#agent.get_path(1, 124)
