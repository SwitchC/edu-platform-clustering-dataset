import sys
import time
import json
import random
import copy
import numpy as np
import ollama
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QProgressBar, 
                             QTextEdit, QGroupBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt

# ==========================================
# ПОТІК 1: МАТЕМАТИЧНА ГЕНЕРАЦІЯ БАЗИ ТА ВОРОНКИ ВЗАЄМОДІЙ
# ==========================================
class MathGeneratorThread(QThread):
    log_updated = pyqtSignal(str)
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.n_users = 2000
        self.n_target_interactions = 150000  # Точна кількість фактичних взаємодій, яку ми хочемо зібрати
        self.n_comments_target = 600        # Скільки відгуків ми відберемо для текстової генерації

    def run(self):
        self.status_updated.emit("Етап 1: Завантаження та підготовка...")
        
        try:
            with open("courses.json", "r", encoding="utf-8") as f:
                courses = json.load(f)
        except FileNotFoundError:
            self.log_updated.emit("❌ Помилка: файл courses.json не знайдено!")
            self.status_updated.emit("Помилка завантаження!")
            return

        # --- [1] ПІДГОТОВКА КУРСІВ ---
        # Якщо вхідний файл має менше ніж 2000 курсів, дублюємо існуючі для досягнення цілі.
        if len(courses) < 2000:
            needed = 2000 - len(courses)
            for _ in range(needed):
                c = copy.deepcopy(random.choice(courses))
                c["Course_ID"] = len(courses) + 1
                courses.append(c)

        # Збираємо всі унікальні теги та формати з усіх курсів
        all_tags = list(set([tag for c in courses for tag in c.get("Теги", [])]))
        all_formats = list(set([fmt for c in courses for fmt in c.get("Формати матеріалів курсу", [])]))
        
        # Резервний список форматів (Fallback), якщо датасет курсів їх не містить
        if not all_formats: all_formats = ["Відео", "Текст", "Інтерактивний квіз"]

        # Створюємо словник для швидкого пошуку курсів за конкретною темою (Оптимізація швидкості O(1))
        courses_by_tag = {tag: [] for tag in all_tags}
        
        # Словник для конвертації текстової складності у число. 
        comp_map = {"базовий": 1, "середній": 2, "просунутий": 3}

        # Проходимося по кожному курсу і готуємо його до симуляції
        for idx, c in enumerate(courses):
            for tag in c.get("Теги", []):
                courses_by_tag[tag].append(idx)
            
            # ОПТИМІЗАЦІЯ: Перетворюємо формати курсу на множину (set). 
            c["_format_set"] = set(c.get("Формати матеріалів курсу", []))
            
            # Присвоюємо курсу числове значення складності
            c["_comp_val"] = comp_map.get(str(c.get("Складність", "Базовий")).lower(), 1)

        # РОЗПОДІЛ ПОПУЛЯРНОСТІ КУРСІВ (Закон Парето)
        c_weights = []
        for i, c in enumerate(courses):
            c["Кількість взаємодій"] = 0
            c["Кількість купівель"] = 0
            c["ratings_list"] = []
            
            # 10% курсів - нові, 15% - хіти (дуже висока вага), 75% - звичайні
            if i < len(courses) * 0.10:
                c["Type"], w = "New", 0.0
            elif i < len(courses) * 0.25:
                c["Type"], w = "Popular", 0.8
            else:
                c["Type"], w = "Rare", 0.2
            c_weights.append(w)
        
        # --- [2] ГЕНЕРАЦІЯ КОРИСТУВАЧІВ ---
        users = []
        
        # Уніфіковані рядкові константи (з великої літери)
        educations = ["Школяр", "Студент", "Бакалавр", "Магістр", "Аспірант"]
        goals = ["Професійний розвиток", "Хобі", "Академічні знання"]

        # Словник для зіставлення: яку складність курсу потенційно шукає людина з певною освітою
        edu_map = {"Школяр": 1, "Студент": 1, "Бакалавр": 2, "Магістр": 3, "Аспірант": 3}

        for i in range(1, self.n_users + 1):
            if i <= self.n_users * 0.1: u_type = "Новий"
            elif i <= self.n_users * 0.3: u_type = "Активний"
            else: u_type = "Пасивний"

            rm = random.randint(0, 10) 
            cm = random.randint(rm, 11) if u_type == "Пасивний" else 11 
            
            # ЛОГІКА ВІКУ ТА ОСВІТИ (Sanity Check)
            edu_choice = random.choice(educations)
            if edu_choice == "Школяр":
                u_age = random.randint(14, 75)
            elif edu_choice == "Студент":
                u_age = random.randint(17, 55)
            elif edu_choice == "Бакалавр":
                u_age = random.randint(21, 75)
            elif edu_choice == "Магістр":
                u_age = random.randint(22, 75)
            else: # Аспірант
                u_age = random.randint(24, 75)

            # ЛОГІКА МЕТИ НАВЧАННЯ (Захист від логічних помилок)
            if u_age < 17:
                u_goal = random.choice(["Хобі", "Академічні знання"])
            else:
                u_goal = random.choice(goals)

            # ПРОФІЛЬ ІНТЕРЕСІВ
            num_topics = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
            fav_topics = random.sample(all_tags, k=min(num_topics, len(all_tags)))
            
            if len(fav_topics) == 1: topic_probs = [1.0]
            elif len(fav_topics) == 2: topic_probs = [0.7, 0.3]
            else: topic_probs = [0.6, 0.3, 0.1]

            users.append({
                "User_ID": i, 
                "User_Type": u_type, 
                "Reg_Month": rm, 
                "Churn_Month": cm,
                "Вік": u_age,        
                "Рівень освіти": edu_choice,
                "Мета навчання": u_goal, 
                "Бажані формати навчання": random.sample(all_formats, k=random.randint(1, 3)),
                
                "Улюблені теми": fav_topics, 
                "Ваги тем": topic_probs,     
                "_mapped_edu": edu_map.get(edu_choice, 2), 
                
                "Загальний час навчання по місяцях": [0.0] * 12,
                "Загальні витрати по місяцях": [0.0] * 12,
                "Пройдені курси за тегами": {tag: [0]*12 for tag in all_tags},
                "Додано в бажане за тегами": {tag: 0 for tag in all_tags},
                "Деталізована історія взаємодій": {
                    "Куплені курси": [], "Пройдені курси": [], "Додані в бажане": [],
                    "Історія оцінювання": [], "Кліки за курсами": [], 
                    "Курси з коментарями": [], "ID залишених коментарів": []
                }
            })

        sim_users = [u for u in users if u["User_Type"] != "Новий"]
        u_weights = [0.8 if u["User_Type"] == "Активний" else 0.2 for u in sim_users]
        u_weights = np.array(u_weights) / sum(u_weights)

        # --- [3] ВОРОНКА ВЗАЄМОДІЙ (СИМУЛЯЦІЯ ЖИТТЯ ПЛАТФОРМИ) ---
        self.status_updated.emit(f"Етап 2: Симуляція {self.n_target_interactions} фактичних взаємодій...")
        raw_comments_pool = []
        
        actual_interactions = 0 
        loop_iterations = 0     
        all_course_indices = list(range(len(courses)))

        while actual_interactions < self.n_target_interactions:
            loop_iterations += 1
            
            u = np.random.choice(sim_users, p=u_weights)
            m = random.randint(u["Reg_Month"], u["Churn_Month"])

            u_formats_set = set(u.get("Бажані формати навчання", []))
            u_edu_level = u["_mapped_edu"] 

            def get_course_by_preferences(available_indices):
                adj_weights = []
                for idx in available_indices:
                    matches = len(u_formats_set & courses[idx]["_format_set"])
                    format_multiplier = 1.0 + (matches * 0.5) if matches > 0 else 0.3
                    
                    course_comp = courses[idx]["_comp_val"]
                    if u_edu_level == course_comp:
                        comp_multiplier = 1.3  
                    elif abs(u_edu_level - course_comp) == 1:
                        comp_multiplier = 1.0  
                    else:
                        comp_multiplier = 0.5  
                        
                    adj_weights.append(c_weights[idx] * format_multiplier * comp_multiplier)
                
                weight_sum = sum(adj_weights)
                if weight_sum > 0:
                    return np.random.choice(available_indices, p=np.array(adj_weights)/weight_sum)
                return random.choice(available_indices)

            if random.random() < 0.85 and u["Улюблені теми"]:
                chosen_topic = random.choices(u["Улюблені теми"], weights=u["Ваги тем"])[0]
                available_course_indices = courses_by_tag[chosen_topic]
                
                if available_course_indices:
                    c_idx = get_course_by_preferences(available_course_indices)
                    c = courses[c_idx]
                else:
                    c_idx = get_course_by_preferences(all_course_indices)
                    c = courses[c_idx]
            else:
                c_idx = get_course_by_preferences(all_course_indices)
                c = courses[c_idx]

            # --- ФАКТИЧНІ ВЗАЄМОДІЇ ---
            u["Деталізована історія взаємодій"]["Кліки за курсами"].append(c["Course_ID"])
            c["Кількість взаємодій"] += 1
            actual_interactions += 1

            if random.random() < (0.15 if u["User_Type"] == "Активний" else 0.05):
                u["Деталізована історія взаємодій"]["Додані в бажане"].append(c["Course_ID"])
                for tag in c.get("Теги", []): u["Додано в бажане за тегами"][tag] += 1
                actual_interactions += 1

            buy_probability = 0.25 if u["User_Type"] == "Активний" else 0.04
            
            if u["Вік"] > 21:
                buy_probability *= 1.5 
                
            spent_this_month = u["Загальні витрати по місяцях"][m]
            penalty_steps = int(spent_this_month // 6000)
            budget_multiplier = max(0.0, 1.0 - (penalty_steps * 0.25)) 
            
            buy_probability *= budget_multiplier
                
            if random.random() < buy_probability and c["Course_ID"] not in u["Деталізована історія взаємодій"]["Куплені курси"]:
                u["Деталізована історія взаємодій"]["Куплені курси"].append(c["Course_ID"])
                u["Загальні витрати по місяцях"][m] += c.get("Ціна", 0.0)
                c["Кількість купівель"] += 1
                actual_interactions += 1

                if u["Мета навчання"] == "Професійний розвиток":
                    base_completion_prob = 0.65
                elif u["Мета навчання"] == "Академічні знання":
                    base_completion_prob = 0.55
                else: 
                    base_completion_prob = 0.45
                
                if u["User_Type"] == "Активний":
                    base_completion_prob += 0.10
                    
                completed = random.random() < base_completion_prob
                
                if completed:
                    u["Деталізована історія взаємодій"]["Пройдені курси"].append(c["Course_ID"])
                    dur = c.get("Тривалість", 60)
                    if random.random() < 0.3 and m < 11:
                        u["Загальний час навчання по місяцях"][m] += dur * 0.4
                        u["Загальний час навчання по місяцях"][m+1] += dur * 0.6
                        cm_m = m + 1
                    else:
                        u["Загальний час навчання по місяцях"][m] += dur
                        cm_m = m
                    for tag in c.get("Теги", []): u["Пройдені курси за тегами"][tag][cm_m] += 1
                    actual_interactions += 1

                rated = False
                if completed and random.random() < 0.5: rated = True
                elif not completed and random.random() < 0.1: rated = True 

                if rated:
                    rating = random.choices([4, 5], weights=[0.3, 0.7])[0] if c["Type"] == "Popular" else random.randint(1, 5)
                    u["Деталізована історія взаємодій"]["Історія оцінювання"].append({"course_id": c["Course_ID"], "rating": rating})
                    c["ratings_list"].append(rating)
                    actual_interactions += 1

                    if random.random() < (0.3 if u["User_Type"] == "Активний" else 0.05):
                        raw_comments_pool.append({
                            "User_ID": u["User_ID"], "Course_ID": c["Course_ID"],
                            "Rating": rating, "Completed": completed
                        })
                        actual_interactions += 1
                        
            if loop_iterations % 5000 == 0: 
                self.progress_updated.emit(min(actual_interactions, self.n_target_interactions), self.n_target_interactions)
        
        self.progress_updated.emit(self.n_target_interactions, self.n_target_interactions)

        # --- [4] ДОДАВАННЯ ШУМУ ТА ДУБЛІКАТІВ ДЛЯ DATA SCIENCE ---
        self.status_updated.emit("Етап 3: Генерація шуму та дублікатів (завдання для датасету)...")
        
        # 4.1 Пропуски в даних КУРСІВ (Об'єкти інтересу описані неповністю)
        for c in courses:
            if random.random() < 0.05: c["Ціна"] = None
            if random.random() < 0.05: c["Тривалість"] = None
            if random.random() < 0.05: c["Складність"] = None
            # Знищуємо описові дані з імовірністю 5%, щоб дата-саєнтист з цим працював
            if random.random() < 0.05: c["Текстовий опис"] = "" 
            if random.random() < 0.05: c["Теги"] = []
            if random.random() < 0.05: c["Формати матеріалів курсу"] = []

        # 4.2 Пропуски в даних КОРИСТУВАЧІВ (до 5% імовірність)
        for u in users:
            if random.random() < 0.05: u["Вік"] = None
            if random.random() < 0.05: u["Рівень освіти"] = None
            if random.random() < 0.05: u["Мета навчання"] = None
            if random.random() < 0.05: u["Бажані формати навчання"] = []

        # 4.3 Дублікати КУРСІВ (2%)
        num_course_dupes = int(len(courses) * 0.02)
        course_dupes = copy.deepcopy(random.sample(courses, num_course_dupes))
        courses.extend(course_dupes) # Додаємо копії в кінець масиву

        # 4.4 Дублікати КОРИСТУВАЧІВ (2%)
        num_user_dupes = int(len(users) * 0.02)
        user_dupes = copy.deepcopy(random.sample(users, num_user_dupes))
        users.extend(user_dupes)


        # --- [5] АГРЕГАЦІЯ, ЗВ'ЯЗУВАННЯ ID ТА ОЧИЩЕННЯ ТЕХНІЧНИХ ДАНИХ ---
        self.log_updated.emit(f"📝 Зібрано {len(raw_comments_pool)} потенційних коментарів. Обираємо {self.n_comments_target}...")
        
        selected_comments = random.sample(raw_comments_pool, min(self.n_comments_target, len(raw_comments_pool)))
        comments_queue = []
        c_id_counter = 1000
        
        for item in selected_comments:
            item["Comment_ID"] = c_id_counter
            comments_queue.append(item)
            
            u_obj = next(u for u in users if u["User_ID"] == item["User_ID"])
            c_obj = next(c for c in courses if c["Course_ID"] == item["Course_ID"])
            
            u_obj["Деталізована історія взаємодій"]["Курси з коментарями"].append(item["Course_ID"])
            u_obj["Деталізована історія взаємодій"]["ID залишених коментарів"].append(c_id_counter)
            
            if "ID коментарів" not in c_obj["Деталізована історія взаємодій"]:
                c_obj["Деталізована історія взаємодій"]["ID коментарів"] = []
            c_obj["Деталізована історія взаємодій"]["ID коментарів"].append(c_id_counter)
            
            c_id_counter += 1

        self.status_updated.emit("Етап 4: Агрегація та збереження...")
        
        for c in courses:
            # Обережно перевіряємо, бо дублікати або стерті об'єкти можуть спричинити помилку
            if "ratings_list" in c and c["ratings_list"]:
                c["Середній рейтинг"] = round(sum(c["ratings_list"]) / len(c["ratings_list"]), 1)
            else:
                c["Середній рейтинг"] = 0.0
                
            c.pop("ratings_list", None)
            c.pop("Type", None)
            c.pop("_format_set", None)
            c.pop("_comp_val", None)

        for u in users:
            u.pop("Reg_Month", None)
            u.pop("Churn_Month", None)
            u.pop("_mapped_edu", None)
            
            u["Деталізована історія взаємодій"]["Кліки за курсами"] = list(set(u["Деталізована історія взаємодій"]["Кліки за курсами"]))
            u["Деталізована історія взаємодій"]["Додані в бажане"] = list(set(u["Деталізована історія взаємодій"]["Додані в бажане"]))

        with open("users_raw.json", "w", encoding="utf-8") as f: json.dump(users, f, ensure_ascii=False, indent=4)
        with open("courses_raw.json", "w", encoding="utf-8") as f: json.dump(courses, f, ensure_ascii=False, indent=4)
        with open("comments_queue.json", "w", encoding="utf-8") as f: json.dump(comments_queue, f, ensure_ascii=False, indent=4)

        self.log_updated.emit(f"✅ БАЗУ ЗГЕНЕРОВАНО! Фактично взаємодій: {actual_interactions}")
        self.log_updated.emit(f"⚠️ Додано шум: 2% дублікатів та ~5% пропущених значень (в т.ч. текстових).")
        self.status_updated.emit("Готово!")

# ==========================================
# ПОТІК 2: ГЕНЕРАЦІЯ ТЕКСТІВ КОМЕНТАРІВ (OLLAMA)
# ==========================================
class LLMCommentThread(QThread):
    log_updated = pyqtSignal(str)
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int, int)
    comment_completed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.is_running = True
        self.is_paused = False
        self.cooldown_interval = 15 * 60 
        self.cooldown_duration = 60      

    def run(self):
        self.status_updated.emit("Завантаження черги коментарів...")
        try:
            with open("comments_queue.json", "r", encoding="utf-8") as f: queue = json.load(f)
            with open("users_raw.json", "r", encoding="utf-8") as f: users = json.load(f)
            with open("courses_raw.json", "r", encoding="utf-8") as f: courses = json.load(f)
        except Exception as e:
            self.log_updated.emit("❌ Спочатку згенеруйте базу (Кнопка 1)!")
            return

        finished_comments = []
        try:
            with open("comments_final.json", "r", encoding="utf-8") as f:
                finished_comments = json.load(f)
        except: pass

        finished_ids = {c["Comment_ID"] for c in finished_comments}
        tasks = [q for q in queue if q["Comment_ID"] not in finished_ids]
        
        total = len(queue)
        completed = len(finished_comments)
        last_rest = time.time()

        for task in tasks:
            if not self.is_running: break
            while self.is_paused:
                time.sleep(0.5)
                if not self.is_running: return

            if time.time() - last_rest > self.cooldown_interval:
                self.status_updated.emit("Охолодження GPU (60 сек)...")
                self.log_updated.emit("--- ПАУЗА НА ОХОЛОДЖЕННЯ ---")
                time.sleep(self.cooldown_duration)
                last_rest = time.time()
                self.status_updated.emit("Генерація коментарів...")

            u = next(x for x in users if x["User_ID"] == task["User_ID"])
            c = next(x for x in courses if x["Course_ID"] == task["Course_ID"])
            
            sentiment = "позитивний" if task["Rating"] >= 4 else "нейтральний" if task["Rating"] == 3 else "негативний"
            status_str = "пройшов цей курс до кінця" if task["Completed"] else "купив цей курс, але кинув його на половині"
            
            # ЗАХИСТ ВІД ШУМУ: Якщо алгоритм на попередньому кроці стер дані, 
            # ми підставляємо фразу "не вказано", щоб LLM модель не зламалася.
            u_age_str = str(u.get("Вік")) if u.get("Вік") is not None else "не вказано"
            u_edu_str = u.get("Рівень освіти") if u.get("Рівень освіти") else "не вказано"
            u_goal_str = u.get("Мета навчання") if u.get("Мета навчання") else "не вказано"
            
            c_title = c.get('Назва курсу', 'Без назви')
            c_tags_str = ", ".join(c.get('Теги', [])) if c.get('Теги') else "не вказана"
            c_desc_str = c.get('Текстовий опис', '')[:200] if c.get('Текстовий опис') else "опис відсутній"

            prompt = f"""Ти - реальний користувач платформи онлайн-курсів. Напиши відгук про курс від першої особи.
Твій профіль: вік {u_age_str} років, освіта: {u_edu_str}, мета навчання: {u_goal_str}.
Курс: "{c_title}" (Тема: {c_tags_str}). Опис курсу: {c_desc_str}...
Статус: ти {status_str}. Твоя оцінка: {task["Rating"]} з 5.

Завдання:
Напиши {sentiment} відгук. Довжина строго 40-80 слів.
ОБОВ'ЯЗКОВО згадай свій досвід або мету. 
НЕ пиши привітань, заголовків чи хештегів. Тільки сам текст коментаря."""

            try:
                response = ollama.chat(
                    model='qwen2.5:7b', 
                    messages=[{'role': 'user', 'content': prompt}],
                    options={
                        'num_predict': 300, 
                        'temperature': 0.7  
                    }
                )
                text = response['message']['content'].strip()
                
                final_obj = {
                    "Comment_ID": task["Comment_ID"],
                    "User_ID": task["User_ID"],
                    "Course_ID": task["Course_ID"],
                    "Rating": task["Rating"],
                    "Text": text
                }
                finished_comments.append(final_obj)
                
                with open("comments_final.json", "w", encoding="utf-8") as f:
                    json.dump(finished_comments, f, ensure_ascii=False, indent=4)
                    
                completed += 1
                self.progress_updated.emit(completed, total)
                self.comment_completed.emit(f"Оцінка {task['Rating']}/5 | {text}")
                
            except Exception as e:
                self.log_updated.emit(f"Помилка: {e}")
                time.sleep(2)

        if self.is_running: self.status_updated.emit("Всі коментарі згенеровано!")

    def pause(self): self.is_paused = True
    def resume(self): self.is_paused = False
    def stop(self): self.is_running, self.is_paused = False, False

# ==========================================
# GUI 
# ==========================================
class DatasetAppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Генератор Датасету + Коментарі (Qwen)")
        self.setGeometry(100, 100, 900, 700)
        self.initUI()
        
    def initUI(self):
        w = QWidget()
        self.setCentralWidget(w)
        layout = QVBoxLayout(w)
        
        self.status_label = QLabel("Статус: Готовий")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        btn_layout = QHBoxLayout()
        self.btn_math = QPushButton("1. Згенерувати базу (Швидко)")
        self.btn_math.clicked.connect(self.run_math)
        btn_layout.addWidget(self.btn_math)
        
        self.btn_llm = QPushButton("2. Згенерувати коментарі (LLM)")
        self.btn_llm.clicked.connect(self.run_llm)
        btn_layout.addWidget(self.btn_llm)
        
        self.btn_pause = QPushButton("⏸ Пауза LLM")
        self.btn_pause.clicked.connect(self.pause_llm)
        self.btn_pause.setEnabled(False)
        btn_layout.addWidget(self.btn_pause)
        
        self.btn_resume = QPushButton("⏯ Продовжити")
        self.btn_resume.clicked.connect(self.resume_llm)
        self.btn_resume.setEnabled(False)
        btn_layout.addWidget(self.btn_resume)
        layout.addLayout(btn_layout)

        self.preview_box = QTextEdit()
        self.preview_box.setReadOnly(True)
        self.preview_box.setStyleSheet("background-color: #1E1E1E; color: #4CAF50; font-family: Segoe UI; font-size: 13px;")
        layout.addWidget(self.preview_box, stretch=2)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(150)
        self.log_edit.setStyleSheet("background-color: #1A1A1A; color: #00BCD4; font-family: Consolas; font-size: 11px;")
        layout.addWidget(self.log_edit, stretch=1)

    def append_log(self, text):
        self.log_edit.append(text)
        self.log_edit.verticalScrollBar().setValue(self.log_edit.verticalScrollBar().maximum())

    def append_preview(self, text):
        self.preview_box.append(text + "\n" + "-"*50 + "\n")
        self.preview_box.verticalScrollBar().setValue(self.preview_box.verticalScrollBar().maximum())

    def update_prog(self, c, t):
        self.progress_bar.setMaximum(t)
        self.progress_bar.setValue(c)

    def run_math(self):
        self.btn_math.setEnabled(False)
        self.math_thread = MathGeneratorThread()
        self.math_thread.log_updated.connect(self.append_log)
        self.math_thread.status_updated.connect(self.status_label.setText)
        self.math_thread.progress_updated.connect(self.update_prog)
        self.math_thread.start()

    def run_llm(self):
        self.btn_llm.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.llm_thread = LLMCommentThread()
        self.llm_thread.log_updated.connect(self.append_log)
        self.llm_thread.status_updated.connect(self.status_label.setText)
        self.llm_thread.progress_updated.connect(self.update_prog)
        self.llm_thread.comment_completed.connect(self.append_preview)
        self.llm_thread.start()

    def pause_llm(self):
        self.llm_thread.pause()
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(True)

    def resume_llm(self):
        self.llm_thread.resume()
        self.btn_resume.setEnabled(False)
        self.btn_pause.setEnabled(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DatasetAppWindow()
    window.show()
    sys.exit(app.exec_())