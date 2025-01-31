#!/opt/anaconda3/envs/itmo-bot/bin/python
import requests
import json
import time
import threading

# URL API бота
BOT_URL = "http://localhost:8081/api/request"
# BOT_URL = "http://77.73.71.31:8081/api/request"


# Количество параллельных запросов
NUM_PARALLEL_REQUESTS = 20  

# Тестовые данные
test_cases = [
    {
        "id": 1,
        "query": (
            "В каком году был основан Университет ИТМО?\n"
            "1. 1900\n"
            "2. 1905\n"
            "3. 1918\n"
            "4. 1930"
        ),
        "expected_answer": 1  # 1900
    },
    {
        "id": 2,
        "query": (
            "Какое название носил Университет ИТМО в советское время?\n"
            "1. Ленинградский институт точной механики и оптики (ЛИТМО)\n"
            "2. Институт высокоточных систем (ИВС)\n"
            "3. Ленинградский институт вычислительной техники (ЛИВТ)\n"
            "4. Петроградский институт механики и оптики (ПИМО)"
        ),
        "expected_answer": 1  # ЛИТМО
    },
    {
        "id": 3,
        "query": (
            "В какой предметной области (согласно рейтингу QS by Subject 2022) "
            "Университет ИТМО входит в топ-100?\n"
            "1. Computer Science & Information Systems\n"
            "2. Physics & Astronomy\n"
            "3. Mathematics\n"
            "4. Engineering - Electrical & Electronic"
        ),
        "expected_answer": 1  # Computer Science & Information Systems
    },
    {
        "id": 4,
        "query": (
            "Сколько мегафакультетов действует в Университете ИТМО?\n"
            "1. 3\n"
            "2. 4\n"
            "3. 5\n"
            "4. 6"
        ),
        "expected_answer": 3  # 5 мегафакультетов
    },
    {
        "id": 5,
        "query": (
            "В каком районе Санкт-Петербурга расположен главный корпус Университета ИТМО "
            "на Кронверкском проспекте, 49?\n"
            "1. Адмиралтейский район\n"
            "2. Петроградский район\n"
            "3. Центральный район\n"
            "4. Василеостровский район"
        ),
        "expected_answer": 2  # Петроградский район
    },
    {
        "id": 6,
        "query": (
            "Какая из перечисленных образовательных программ в ИТМО специализируется "
            "на дизайне и мультимедиа?\n"
            "1. Информационная безопасность\n"
            "2. Химическая биология\n"
            "3. Technological Art & Design\n"
            "4. Киберфизические системы"
        ),
        "expected_answer": 3  # Technological Art & Design
    },
    {
        "id": 7,
        "query": (
            "Как называется бизнес-акселератор, созданный при Университете ИТМО?\n"
            "1. Future Technologies\n"
            "2. IIDF (ФРИИ)\n"
            "3. ITMO Accelerator\n"
            "4. Skolkovo"
        ),
        "expected_answer": 3  # ITMO Accelerator
    },
    {
        "id": 8,
        "query": (
            "Какой статус Университет ИТМО имел в рамках государственной инициативы «Проект 5-100»?\n"
            "1. Лидер\n"
            "2. Участник\n"
            "3. Эксперт\n"
            "4. Не участвовал"
        ),
        "expected_answer": 2  # Участник
    },
    {
        "id": 9,
        "query": (
            "К какому министерству относится Университет ИТМО?\n"
            "1. Министерство культуры РФ\n"
            "2. Министерство науки и высшего образования РФ\n"
            "3. Министерство финансов РФ\n"
            "4. Министерство цифрового развития РФ"
        ),
        "expected_answer": 2  # Минобрнауки РФ
    },
    {
        "id": 10,
        "query": (
            "Как называется университетская медиа-платформа (онлайн-издание) ИТМО?\n"
            "1. ITMO.News\n"
            "2. GO ITMO\n"
            "3. ITMO Media\n"
            "4. ITMO Journal"
        ),
        "expected_answer": 1  # ITMO.News
    },
    {
        "id": 11,
        "query": (
            "В каком направлении научных исследований Университет ИТМО особенно силён, "
            "согласно последним рейтингам?\n"
            "1. Нейробиология\n"
            "2. Фотоника\n"
            "3. Геология\n"
            "4. Социология"
        ),
        "expected_answer": 2  # Фотоника
    },
    {
        "id": 12,
        "query": (
            "Как называется программа, в которой студенты Университета ИТМО могут "
            "оформить собственный стартап в качестве выпускной квалификационной работы?\n"
            "1. Startup as Art\n"
            "2. IGoStartup\n"
            "3. Startup as Thesis\n"
            "4. ITMO Business Incubator"
        ),
        "expected_answer": 3  # Startup as Thesis
    },
    {
        "id": 13,
        "query": (
            "В какой области ITMO активно сотрудничал с Эрмитажем, разрабатывая AR/VR-технологии?\n"
            "1. Проект «Цифровая Эрмитаж-библиотека»\n"
            "2. Art&Science Project\n"
            "3. Hermitage.Connected\n"
            "4. Museum Vision Initiative"
        ),
        "expected_answer": 3  # Hermitage.Connected
    },
    {
        "id": 14,
        "query": (
            "Какую лабораторию (лабораторию электронного здравоохранения) можно найти в Университете ИТМО?\n"
            "1. Лаборатория цифровой медицины\n"
            "2. eHealth Lab\n"
            "3. Smart Health IT\n"
            "4. ITMO Health & AI"
        ),
        "expected_answer": 2  # eHealth Lab
    },
    {
        "id": 15,
        "query": (
            "Сколько мегагрантов (по данным на 2022 год) реализовано с участием учёных "
            "Университета ИТМО (в рамках программы Правительства РФ)?\n"
            "1. Менее 5\n"
            "2. От 5 до 10\n"
            "3. От 10 до 15\n"
            "4. Более 15"
        ),
        "expected_answer": 2  # По данным ИТМО ~ 8 мегагрантов (в разное время) 
    },
    {
        "id": 16,
        "query": "В каком году был основан Университет ИТМО?",
        "expected_answer": None  # Нет вариантов -> answer = None
    },
    {
        "id": 17,
        "query": "Какова основная цель программы 'Научная весна' в ИТМО?",
        "expected_answer": None  # Нет вариантов -> answer = None
    }
]

# Статистика тестирования
correct = 0
incorrect = 0
errors = 0
results = []

# Таймер
start_time = time.time()

def send_request(test_case):
    global correct, incorrect, errors
    try:
        response = requests.post(BOT_URL, json={"query": test_case["query"], "id": test_case["id"]}, timeout=10)
        if response.status_code == 200:
            data = response.json()
            bot_answer = data.get("answer")

            # Проверяем правильность ответа
            is_correct = bot_answer == test_case["expected_answer"]
            if is_correct:
                correct += 1
            else:
                incorrect += 1

            results.append({
                "id": test_case["id"],
                "expected": test_case["expected_answer"],
                "received": bot_answer,
                "status": "✅ Correct" if is_correct else "❌ Incorrect"
            })

            print(f"✅ Test {test_case['id']} OK: {json.dumps(data, ensure_ascii=False)}")
        else:
            errors += 1
            print(f"❌ Test {test_case['id']} Error: {response.status_code}")
    except Exception as e:
        errors += 1
        print(f"❌ Test {test_case['id']} Failed: {str(e)}")

# Запуск тестов в параллельных потоках
threads = []
for test in test_cases:
    thread = threading.Thread(target=send_request, args=(test,))
    thread.start()
    threads.append(thread)

# Ожидание завершения всех потоков
for thread in threads:
    thread.join()

end_time = time.time()

# Итоговый отчёт
print("\n=== Test Summary ===")
print(f"✅ Correct: {correct}")
print(f"❌ Incorrect: {incorrect}")
print(f"⚠️ Errors: {errors}")
print(f"⏳ Total execution time: {end_time - start_time:.2f} seconds")