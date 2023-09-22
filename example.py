import multiprocessing
import os
import requests
import time

# WARNING: if running in OSX you need to execute the following instruction in each session
# export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

if __name__ == '__main__':
    # Avvia il server Flask in un processo separato
    server_process = multiprocessing.Process(target=os.system, args=('python3 textpreprocessed2topic.py',))
    server_process.start()

    # Aspetta che il server Flask sia avviato
    time.sleep(5)  # Attendi alcuni secondi per avviare il server

    url = "http://127.0.0.1:5000"

    data = {
        'text': {'moral': ': noi, che accogliamo i profughi in famiglia in casa nostra', 'topic': 'accogliere profugo famiglia'},
        'theta': 0.03
        }

    def request_and_print():
        response = requests.get(f"{url}/topic_prediction", json=data)
        print(response.status_code)
        print(response.json())

    request_and_print()

