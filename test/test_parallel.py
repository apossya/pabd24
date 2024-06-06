import time
from multiprocessing import Pool
import requests
from dotenv import dotenv_values
import sys

# Путь к файлу для сохранения вывода
log_file_path = '/home/user1/pabd24/log/test_np_unicorn_3.txt'

# Перенаправляем стандартный вывод в файл
sys.stdout = open(log_file_path, 'w')

config = dotenv_values(".env")
endpoint = 'http://176.109.104.141:8000/predict'
HEADERS = {"Authorization": f"Bearer {config['APP_TOKEN']}"}


def do_request(area: int) -> str:
    data = {'area': area}
    t0 = time.time()
    resp = requests.post(
        endpoint,
        json=data,
        headers=HEADERS
    ).text
    t = time.time() - t0
    return f'Waited {t:0.2f} sec ' + resp 


def test_10():
    with Pool(10) as p:
        print(*p.map(do_request, range(10, 110, 10)))
    

if __name__ == '__main__':
    test_10()

# Закрываем файл после завершения записи
sys.stdout.close()