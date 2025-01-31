#!/opt/anaconda3/envs/itmo-bot/bin/python
import requests
import json
import time

# URL API вашего бота
BOT_URL = "http://localhost:8081/api/request"

# Количество тестов для исполнения (по умолчанию = все)
NUM_TESTS = 1  # Измените число, если хотите выполнить меньше тестов

# Тестовые запросы с ожидаемыми ответами
test_cases = [
    {"id": 1, "query": "В каком рейтинге (по состоянию на 2021 год) ИТМО впервые вошёл в топ-400 мировых университетов?\n1. ARWU (Shanghai Ranking)\n2. Times Higher Education (THE) World University Rankings\n3. QS World University Rankings\n4. U.S. News & World Report Best Global Universities", "expected_answer": 3},
    {"id": 2, "query": "В каком году Университет ИТМО был включён в число Национальных исследовательских университетов России?\n1. 2007\n2. 2009\n3. 2011\n4. 2015", "expected_answer": 2},
    {"id": 3, "query": "В каком рейтинге (по состоянию на 2021 год) ИТМО впервые вошёл в топ-400 мировых университетов?\n1. ARWU (Shanghai Ranking)\n2. Times Higher Education (THE) World University Rankings\n3. QS World University Rankings\n4. U.S. News & World Report Best Global Universities", "expected_answer": 3},
    {"id": 4, "query": "Какая была первоначальная специализация учебного заведения, которое позже стало Университетом ИТМО?\n1. Военная академия\n2. Школа точной механики и оптики\n3. Кафедра информатики и вычислительной техники\n4. Институт химических технологий", "expected_answer": 2},
    {"id": 5, "query": "Кто является основателем одной из первых научных школ Университета ИТМО в области оптики?\n1. Альберт Эйнштейн\n2. Сергей Вавилов\n3. Генрих Гейне\n4. Николай Басов", "expected_answer": 2},
    {"id": 6, "query": "Как называется одна из ключевых исследовательских направлений Университета ИТМО?\n1. Биотехнологии\n2. Генная инженерия\n3. Фотоника и оптоинформатика\n4. Космическая навигация", "expected_answer": 3},
    {"id": 7, "query": "В каком городе расположен главный кампус Университета ИТМО?\n1. Москва\n2. Санкт-Петербург\n3. Новосибирск\n4. Казань", "expected_answer": 2},
    {"id": 8, "query": "Сколько раз Университет ИТМО побеждал в студенческом чемпионате мира по программированию ACM ICPC (до 2020 года)?\n1. 4 раза\n2. 5 раз\n3. 7 раз\n4. Ни разу", "expected_answer": 3},
    {"id": 9, "query": "В какой области Университет ИТМО наиболее известен своими научными исследованиями?\n1. Физика высоких энергий\n2. Искусственный интеллект и робототехника\n3. Астрономия и астрофизика\n4. Юриспруденция и международное право", "expected_answer": 2},
    {"id": 10, "query": "Какая программа Университет ИТМО ориентирована на взаимодействие науки и бизнеса?\n1. StartUp Платформа\n2. Инновационная лаборатория химии\n3. Art & Science Институт\n4. Институт передовых производственных технологий (Advanced Manufacturing Technologies)", "expected_answer": 4},
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
