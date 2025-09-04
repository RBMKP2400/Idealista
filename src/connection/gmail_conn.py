import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from params.gmail_params import EMAIL_TEMPLATE
from utils.loguru_conf import logger

class GmailSender:
    def __init__(self, config):
        self.email = os.getenv('GMAIL_USER')
        self.password = os.getenv('GMAIL_PASSWORD')
        self.smtp_server = 'smtp.gmail.com'
        self.smtp_port = 587
        self.config = config

    def build_message(self, df_new_records=None, df_metrics=None, archivo_adjunto=None):
        """
        Construye y devuelve un objeto MIMEMultipart con el cuerpo del mensaje
        usando la plantilla EMAIL_TEMPLATE.

        Args:
            df_new_records (pd.DataFrame): DataFrame con los nuevos registros.
            df_metrics (pd.DataFrame): DataFrame con las métricas globales.
            archivo_adjunto (str): Ruta del archivo Excel adjunto.

        Returns:
            MIMEMultipart: Mensaje con el cuerpo del correo y los adjuntos.
        """
        mensaje = MIMEMultipart()
        mensaje['From'] = self.email
        mensaje['To'] = ", ".join(self.config["GMAIL_DESTINATION"])
        mensaje['Subject'] = self.config["GMAIL_SUBJECT"]

        # Construir el cuerpo usando la plantilla
        cuerpo_html = EMAIL_TEMPLATE.format(
            num_registros=len(df_new_records) if df_new_records is not None else 0,
            new_records=df_new_records.to_html(index=False, border=1) if df_new_records is not None else "<p>Sin datos</p>",
            metrics=df_metrics.to_html(index=False, border=1) if df_metrics is not None else "<p>Sin datos</p>"
        )

        # Adjuntar HTML directamente
        mensaje.attach(MIMEText(cuerpo_html, 'html'))

        # Adjuntar archivo si existe
        if archivo_adjunto and os.path.isfile(archivo_adjunto):
            with open(archivo_adjunto, 'rb') as adjunto:
                parte = MIMEBase('application', 'vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                parte.set_payload(adjunto.read())
            encoders.encode_base64(parte)
            parte.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(archivo_adjunto)}"')
            mensaje.attach(parte)

        return mensaje

    def send_message(self, df_new_records=None, df_metrics=None, archivo_adjunto=None):
        """
        Envía el correo construido con build_message. Esta función utiliza la
        conexión SMTP de Gmail para enviar el correo.

        Args:
            df_new_records (pd.DataFrame): DataFrame con los nuevos registros.
            df_metrics (pd.DataFrame): DataFrame con las métricas globales.
            archivo_adjunto (str): Ruta del archivo Excel adjunto.
        """
        mensaje = self.build_message(df_new_records, df_metrics, archivo_adjunto)

        try:
            # Conectar al servidor SMTP de Gmail
            servidor = smtplib.SMTP(self.smtp_server, self.smtp_port)
            servidor.starttls()

            # Iniciar sesión en el servidor
            servidor.login(self.email, self.password)

            # Enviar el correo
            servidor.sendmail(self.email, self.config["GMAIL_DESTINATION"], mensaje.as_string())

            # Cerrar la conexión con el servidor
            servidor.quit()

            logger.info("Correo enviado correctamente.")
        except Exception as e:
            logger.error(f"Error al enviar correo: {e}")

