def get_params(shape: str, location: str, page: int = 1) -> dict:
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
    page : int, optional
        El número de página de resultados que se desea obtener. Por defecto es 1.

    Retorna
    -------
    dict
        Los parámetros que se usarán en la API de Idealista.
    """

    # Parámetros para hogares grandes
    params_big_home = {
        # Filtros principales
        "country": "es",  # País: España
        "operation": "sale",  # Tipo de operación: Venta
        "propertyType": "homes",  # Tipo de propiedad: Vivienda
        "maxPrice": "310000",  # Precio máximo: 310.000€
        "maxItems": "50",  # Número de resultados por página: 50
        "sort": "asc",  # Orden: Ascendente
        "locale": "es",  # Idioma: Español
        "order": "price",  # Orden: Precio
        # Filtros específicos del hogar
        "minSize": 90,  # Tamaño mínimo: 90 m2
        "maxSize": 150,  # Tamaño máximo: 150 m2
        "bedrooms": "3",  # Número de habitaciones: 3
        "preservation": "good"  # Estado de conservación: Bueno
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
        "bedrooms": "2,3",  # Número de habitaciones: 2 o 3
        "minSize": 60,  # Tamaño mínimo: 60 m2
        "maxSize": 75,  # Tamaño máximo: 75 m2
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
        "bedrooms": "2,3",  # Número de habitaciones: 2 o 3
        "minSize": 70,  # Tamaño mínimo: 70 m2
        "maxSize": 100,  # Tamaño máximo: 100 m2
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
    params["numPage"] = page

    return params
