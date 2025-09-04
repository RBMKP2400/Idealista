EMAIL_TEMPLATE = """
<html>
<body>
<p>¡Hola, buscador/a de gangas inmobiliarias! 🏠</p>

<p>He rastreado idealista como un sabueso y he encontrado <b>{num_registros}</b> nuevas joyitas para ti.</p>

<h3>📋 Registros recién salidos del horno:</h3>
{new_records}

<h3>📊 Métricas globales de la base de datos:</h3>
{metrics}

<p>Te adjunto la base de datos actualizada en formato Excel.</p>

<p>¡Hasta la próxima búsqueda, majo/a!<br>
Tu bot automático 🤖</p>
</body>
</html>
"""
