# logging_config.py

import logging
import os
import os as _os
from logging.handlers import RotatingFileHandler

def setup_logging(log_dir: str = "logs"):
    """Configura logging con salida a consola y archivo rotativo.

    Args:
        log_dir: Directorio donde almacenar logs. Se crea si no existe.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "app.log")
    handlers = [logging.StreamHandler()]
    try:
        file_handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3, encoding='utf-8')
        handlers.append(file_handler)
    except Exception as e:
        # Fallback: solo consola si no se puede crear archivo
        logging.error(f"No se pudo inicializar RotatingFileHandler: {e}")
    # Creamos el logger root manualmente para poder forzar el formateador de consola.
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Elimina handlers previos (si setup_logging se llama más de una vez durante tests o recargas)
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)
    for h in handlers:
        root.addHandler(h)

    # Formato base (sin colores)
    base_format = '%(asctime)s %(name)s %(levelname)s %(message)s'
    plain_formatter = logging.Formatter(base_format)
    for h in handlers:
        # FileHandler siempre neutro
        if isinstance(h, RotatingFileHandler):
            h.setFormatter(plain_formatter)

    # -------- Colores (solo levelname en consola) --------
    use_color = _os.getenv('USE_COLOR_LOGS', '1') not in ('0', 'false', 'False')

    console_handlers = [h for h in handlers if isinstance(h, logging.StreamHandler)]
    if use_color and console_handlers:
        try:
            from colorama import init as colorama_init  # type: ignore
            colorama_init()
        except Exception:
            pass

        level_colors = {
            'DEBUG': '\x1b[34m',   # azul
            'INFO': '\x1b[32m',    # verde
            'WARNING': '\x1b[33m', # amarillo
            'ERROR': '\x1b[31m',   # rojo
            'CRITICAL': '\x1b[41;37m',  # blanco sobre rojo
        }
        reset = '\x1b[0m'

        class OnlyLevelFormatter(logging.Formatter):
            BLUE = '\x1b[34m'
            MAGENTA = '\x1b[35m'
            RESET = reset

            def formatTime(self, record, datefmt=None):  # colorea asctime
                ct = super().formatTime(record, datefmt)
                return f"{self.BLUE}{ct}{self.RESET}"

            def format(self, record: logging.LogRecord) -> str:
                txt = super().format(record)
                # Colorear levelname (una sola vez)
                lvl_color = level_colors.get(record.levelname)
                if lvl_color:
                    txt = txt.replace(record.levelname, f"{lvl_color}{record.levelname}{reset}", 1)
                # Colorear nombre del logger (primera aparición exacta)
                name_token = record.name
                if name_token:
                    txt = txt.replace(name_token, f"{self.MAGENTA}{name_token}{self.RESET}", 1)
                return txt

        color_formatter = OnlyLevelFormatter(base_format)
        for ch in console_handlers:
            ch.setFormatter(color_formatter)
        logging.getLogger(__name__).info('Colored logs activos (solo levelname). Desactivar: USE_COLOR_LOGS=0')
    else:
        for ch in console_handlers:
            ch.setFormatter(plain_formatter)
        if not use_color:
            logging.getLogger(__name__).info('Colored logs desactivados (USE_COLOR_LOGS=0)')

if __name__ == '__main__':
    setup_logging()
    logging.info("Configuración de logging cargada con éxito.")
