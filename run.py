import subprocess
import sys
import time

frontend = subprocess.Popen([
    sys.executable, "-m", "http.server", "5500"
])

backend = subprocess.Popen([
    sys.executable, "-m", "uvicorn",
    "app:app", "--reload"
])

print("Frontend: http://localhost:5500")
print("Backend:  http://localhost:8000")
print("Press Ctrl+C to stop both servers.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    frontend.terminate()
    backend.terminate()