#!/opt/anaconda3/envs/itmo-bot/bin/python
import requests
import json
import time

# URL API вашего бота
BOT_URL = "http://localhost:8081/api/request"

# Количество тестов для исполнения (по умолчанию = все)
NUM_TESTS = 17  # Измените число, если хотите выполнить меньше тестов

# Тестовые запросы с ожидаемыми ответами
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

# Обрезаем тесты до количества NUM_TESTS
test_cases = test_cases[:NUM_TESTS]

# Запускаем таймер
start_time = time.time()

# Статистика тестирования
correct = 0
incorrect = 0
detailed_results = []

# Запуск тестов
for test in test_cases:
    print("\n=== Running Test ===")
    print(f"📝 Test ID: {test['id']}")
    print("🔍 Question:", test['query'].split("\n")[0], "\n")

    response = requests.post(BOT_URL, json={"query": test["query"], "id": test["id"]})
    
    if response.status_code == 200:
        data = response.json()
        bot_answer = data.get("answer")
        reasoning = data.get("reasoning", "No reasoning provided")
        sources = data.get("sources", [])

        # Проверяем правильность ответа
        is_correct = bot_answer == test["expected_answer"]
        if is_correct:
            correct += 1
        else:
            incorrect += 1

        # Лог результатов
        detailed_results.append({
            "id": test["id"],
            "question": test["query"].split("\n")[0],  # Берём только текст вопроса
            "expected": test["expected_answer"],
            "received": bot_answer,
            "status": "✅ Correct" if is_correct else "❌ Incorrect"
        })

        # Вывод детального лога
        print(f"✅ API Response: {json.dumps(data, indent=2, ensure_ascii=False)}")

    else:
        incorrect += 1
        detailed_results.append({
            "id": test["id"],
            "question": test["query"].split("\n")[0],
            "expected": test["expected_answer"],
            "received": f"Error {response.status_code}",
            "status": "❌ API Error"
        })

        # Вывод ошибки
        print(f"❌ API Error: {response.status_code}")

# Останавливаем таймер
end_time = time.time()
total_time = end_time - start_time

# Итоговый отчёт
print("\n=== Test Results ===")
print(f"✅ Correct: {correct}")
print(f"❌ Incorrect: {incorrect}")
print(f"⏳ Total execution time: {total_time:.2f} seconds\n")

# Выход с кодом ошибки, если тесты не пройдены
if incorrect > 0:
    exit(1)
else:
    exit(0)