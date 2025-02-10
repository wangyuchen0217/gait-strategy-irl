
# Redirect stdout to the log file
class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message.strip():  # Only log non-empty messages
            self.level(message)

    def flush(self):
        pass  # Required for compatibility