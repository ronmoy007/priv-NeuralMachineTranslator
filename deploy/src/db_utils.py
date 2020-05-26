from io import StringIO
import psycopg2
import pandas as pd
import os

MY_USER = os.environ['MY_USER']
MY_DB = os.environ['MY_DB']
MY_HOST = os.environ['MY_HOST']
MY_PASS = os.environ['MY_PASS']
MY_PORT = os.environ['MY_PORT']


def get_data(query):
    try:
        connection = psycopg2.connect(user=MY_USER , # Usuario RDS
                                     password=MY_PASS, # password de usuario de RDS
                                     host=MY_HOST ,#"127.0.0.1", # cambiar por el endpoint adecuado
                                     port=MY_PORT, # cambiar por el puerto
                                     database=MY_DB ) # Nombre de la base de datos
        cursor = connection.cursor()

        cursor.execute(query)

        #print("Selecting rows from table using cursor.fetchall")
        records = cursor.fetchall()
        df = pd.DataFrame(records)
        col_names =  [i[0] for i in cursor.description]
        df.columns = col_names
        cursor.close()
        connection.close()
        #print("PostgreSQL connection is closed")
        return df

    except (Exception, psycopg2.Error) as error :
        print ("Error while fetching data from PostgreSQL", error)
