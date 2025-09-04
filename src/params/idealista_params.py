def get_params(shape: str, location: str) -> dict:
    """
    Obtiene los parámetros que se usarán en la API de Idealista para
    obtener los datos relevantes según la forma y ubicación indicadas.

    Parámetros
    ----------
    shape : str
        La forma del hogar. Puede ser "Big Home" o "Small Home".
    location : str
        La ubicación del hogar. Puede ser cualquiera de las siguientes:
        "La Latina", "Carabanchel", "Alcorcón", "Leganés", "Aluche".

    Retorna
    -------
    dict
        Los parámetros que se usarán en la API de Idealista.
    """

    # Parámetros para hogares grandes
    params_big_home = {
        # Filtros principales
        "country": "es",
        "operation": "sale",
        "propertyType": "homes",
        "maxPrice": "310000",
        "maxItems": "50",
        "sort": "asc",
        "locale": "es",
        "order": "price",
        # Filtros específicos del hogar
        "minSize": 90,
        "maxSize": 150,
        "bedrooms": "3",
        "preservation": "good"
    }

    # Parámetros para hogares pequeños
    params_small_home = {
        # Filtros principales
        "country": "es",
        "operation": "sale",
        "propertyType": "homes",
        "maxPrice": "235000",
        "maxItems": "50",
        "sort": "asc",
        "locale": "es",
        "order": "price",
        # Filtros específicos del hogar
        "bedrooms": "2,3",
        "minSize": 60,
        "maxSize": 75,
        "preservation": "good"
    }

    # Parámetros para hogares pequeños
    params_mid_home = {
        # Filtros principales
        "country": "es",
        "operation": "sale",
        "propertyType": "homes",
        "maxPrice": "290000",
        "maxItems": "50",
        "sort": "asc",
        "locale": "es",
        "order": "price",
        # Filtros específicos del hogar
        "bedrooms": "2,3",
        "minSize": 70,
        "maxSize": 100,
        "preservation": "good"
    }

    # Ubicaciones y sus IDs
    LocationsID = {
        "La Latina": "0-EU-ES-28-07-001-079-10",
        "Carabanchel": "0-EU-ES-28-07-001-079-11",
        "Alcorcón": "0-EU-ES-28-04-002-007",
        "Leganés": "0-EU-ES-28-04-003-074",
        "Aluche": "0-EU-ES-28-07-001-079-10-004",
    }

    # Mapeo de forma a parámetros
    type_home = {
        "Big Home": params_big_home,
        "Small Home": params_small_home,
        "Mid Home": params_mid_home
    }

    # Obtener los parámetros según la forma y ubicación dadas
    params = type_home[shape]
    params["locationId"] = LocationsID[location]

    return params
