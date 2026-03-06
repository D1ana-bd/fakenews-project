import logging
import json
from pathlib import Path
from datetime import datetime

LOG_DIR = Path(__file__).resolve().parents[3] / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "project.log"


class JSONFormatter(logging.Formatter):
    """
    Formatter personalizado para converter registos de log em formato JSON.

    Este formatter gera uma string JSON com informações detalhadas sobre cada evento
    de log, incluindo o timestamp, nível, módulo, função e número da linha.
    Caso exista uma exceção associada ao evento, o stack trace é também incluído.

    Métodos
    --------
    format(record)
        Converte o registo de log (record) num objeto JSON formatado.
    """

    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Adiciona informação de exceção se existir
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(
            log_data, ensure_ascii=False
        )  # mostra acentos e caracteres especiais corretamente no JSON


# Configurar handler para ficheiro com formato JSON
file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_handler.setFormatter(JSONFormatter())

# Configurar handler para consola com formato texto (mais legível)
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
)

# Configurar logging básico
logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])


def get_logger(name: str):
    """
    Retorna um logger configurado com os handlers e formatters definidos.

    Parameters
    ----------
    name : str
        Nome do módulo ou componente que solicita o logger. Este nome é utilizado
        para identificar a origem das mensagens de log.

    Returns
    -------
    logging.Logger
        Instância de logger pronta a utilizar com o formato e nível definidos.
    """
    return logging.getLogger(name)
