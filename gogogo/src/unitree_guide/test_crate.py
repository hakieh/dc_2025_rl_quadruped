from crate import client
import torch
print(torch.__version__)
connection = client.connect("http://localhost:4200",username="crate")
cursor = connection.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS test_data(
               id INTEGER PRIMARY KEY,
               name STRING,
               value FLOAT
               )
""")



cursor.execute("INSERT INTO test_data (id,name,value) VALUES(?,?,?)",(2, 'sensor1',23.5))

cursor.execute("SELECT * FROM test_data")
print(cursor.fetchall())

cursor.close()
connection.close()

# Connect using DB API.
# from crate import client
# #from pprint import PrettyPrinter as pp

# query = "SELECT country, mountain, coordinates, height FROM sys.summits ORDER BY country;"

# with client.connect("localhost:4200", username="crate") as connection:
#     cursor = connection.cursor()
#     cursor.execute(query)
#     print(cursor.fetchall())
#     cursor.close()

