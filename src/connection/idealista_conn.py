import os
import base64
import requests
from datetime import datetime, timedelta
import json
from utils.loguru_conf import logger

class IdealistaConnection:
    def __init__(self):
        self.key = os.getenv('IDEALISTA_KEY')
        self.secret = os.getenv('IDEALISTA_SECRET')
        self.url = os.getenv('IDEALISTA_URL', 'https://api.idealista.com/3.5/es/search')
        
        self.token_file = os.getenv('IDEALISTA_TOKEN_FILE',"/tokens/token_store.json")
        self.load_token_file()

    def load_token_file(self):
        """
        Lee el archivo de tokens y carga el token actual y su fecha de expiración.
        
        Si el archivo no existe, se solicita un nuevo token y se guarda en el archivo.
        Si el token almacenado ha caducado, se solicita un nuevo token y se guarda en el archivo.
        """
        if not os.path.exists(self.token_file):
            # Si el archivo no existe, solicitar un nuevo token y guardarlo en el archivo
            self._request_new_token()
            self.save_token()
        else:
            with open(self.token_file, "r") as f:
                data = json.load(f)
            self.token = data["bearer_token"]
            self.token_expiry = datetime.fromisoformat(data["expiry"])

            if self.token is None or datetime.now() >= self.token_expiry:
                # Si el token almacenado ha caducado, solicitar un nuevo token y guardarlo en el archivo
                self._request_new_token()
                self.save_token()
            else:
                logger.info(f"Se ha cargado el token desde el archivo.")
        
    def save_token(self):
        """
        Guarda el token actual y su fecha de expiración en un archivo JSON.

        El archivo se guarda en la ruta especificada en self.token_file. Si el
        directorio no existe, se crea automáticamente.
        """
        # Crear el directorio si no existe
        os.makedirs(os.path.dirname(self.token_file), exist_ok=True)
        
        # Guardar el token y su fecha de expiración en un archivo JSON
        with open(self.token_file, "w") as f:
            json.dump({
                "bearer_token": self.token,
                "expiry": self.token_expiry.isoformat()
            }, f)

    def _request_new_token(self):
        """
        Solicita un nuevo bearer token desde la API de Idealista y actualiza self.token y self.token_expiry.

        Idealista proporciona un token de autenticación con una duración de 60 minutos. En este método,
        se solicita un nuevo token y se actualizan las propiedades self.token y self.token_expiry para
        recordar el token y su fecha de expiración.

        Raises:
            Exception: Si la solicitud de token no es exitosa.
        """
        # Codificar credenciales en base64 para la autenticación Basic
        # https://docs.idealista.com/idealista-api-3.5/apidocs/idealista-api-3.5.html#section/Authentication
        credentials = base64.b64encode(f"{self.key}:{self.secret}".encode("utf-8")).decode("utf-8")
        
        # Realizar una solicitud POST para obtener un nuevo token
        # https://docs.idealista.com/idealista-api-3.5/apidocs/idealista-api-3.5.html#operation/getToken
        response = requests.post(
            "https://api.idealista.com/oauth/token",
            data="grant_type=client_credentials&scope=read",
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        
        # Verificar si la respuesta es exitosa
        if response.status_code == 200:
            logger.info(f"Se ha obtenido el token correctamente.")
        elif response.status_code == 429:
            logger.error(f"Se ha alcanzado el límite de 100 req/month.")
        else:
            raise Exception(f"Error obteniendo token: {response.status_code} {response.text}")
        
        data = response.json()
        self.token = data["access_token"]
        # Restar 30 segundos a la expiración para tener un margen de seguridad y renovar el token antes de que expire
        self.token_expiry = datetime.now() + timedelta(seconds=data["expires_in"] - 30)
        logger.info(f"Se ha renovado el token.")

    def api_post(self, params: dict):
        """
        Ejemplo de llamada POST a Idealista.

        Esta función obtiene un token de autenticación automático y realiza una solicitud POST a
        la API de Idealista. Si el token no es válido o ha expirado, se solicita uno nuevo.

        Args:
            params (dict): Los parámetros de la solicitud.

        Returns:
            tuple: Un tuple con el estado de la respuesta y el contenido en formato JSON.
        """
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        response = requests.post(self.url, headers=headers, data=params)

        if response.status_code == 200:
            logger.info(f"Se ha realizado la consulta correctamente.")
        elif response.status_code == 429:
            logger.error(f"Se ha alcanzado el límite de 100 req/month.")
        else:
            logger.error(f"Error en la solicitud: {response.status_code} {response.text}")
            raise Exception(f"Error obteniendo token: {response.status_code} {response.text}")
        
        return response.status_code, response.json()
