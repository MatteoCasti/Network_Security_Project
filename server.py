from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware

from database import DatabaseSingleton, DbInterface
import uvicorn

app_server = FastAPI()
security = HTTPBasic()

app_server.add_middleware(HTTPSRedirectMiddleware)


def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    # these ones should be encrypted
    correct_username = "username"
    correct_password = "password"
    if credentials.username != correct_username or credentials.password != correct_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials


@app_server.put('/api/resources')
async def receive_data_unity(data: dict, _=Depends(authenticate)):
    try:
        # manage data
        gap_name = data.get('gap_name')
        coordinates_lat_lon = data.get('coordinates_lat_lon')
        coordinates_utm = data.get('coordinates_utm')
        people_in = data.get('people_in')
        people_out = data.get('people_out')
        people_unique = data.get('people_unique')
        date = data.get('date')
        time_slot = data.get('time_slot')
        website_data = [gap_name, coordinates_lat_lon, people_in, people_out, people_unique, date, time_slot]
        unity_data = [gap_name, coordinates_utm, people_in, people_out, people_unique, date, time_slot]

        # print(data)

        # web_db.write_db(website_data)
        # unity_db.write_db(unity_data)

        return {
            "status": "success",
            "message": "Data received"
        }  # send a response to the client

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app_server.get('/api/unity_resources')
async def get_data(_=Depends(authenticate)):
    """
    This function allows to read the database through the server.
    So, we can use this to read the database from other external applications
    """
    try:
        pass
        # data = unity_db.search_db()
        # return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app_server.get('/api/website_resources')
async def get_data(_=Depends(authenticate)):
    """
    This function allows to read the database through the server.
    So, we can use this to read the database from other external applications
    """
    try:
        pass
        # data = web_db.search_db()
        # return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app_server, host="0.0.0.0", port=8000, ssl_keyfile="../key.pem",
                ssl_certfile="../cert.pem")

"""
IMPORTANTE!
I parametri ssl_keyfile e ssl_certificate permettono di utilizzare un certificato ed una chiave privata,
per ottenere una codifica dei dati durante lo scambio dei pacchetti tra client e server.
In fase di sviluppo, è possibile crearsi il proprio certificato autofirmato, ad esempio con il codice:
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365
Il quale permette di creare un certificato autofirmato con una validità di 365 giorni.
In fase di distribuzione, sarà molto importante utilizzare un certificato rilasciato da un organizzazione autorizzata.
"""
