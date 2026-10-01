import sys
import time
import json
import re
import glob
import ollama
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QProgressBar, 
                             QTextEdit, QGroupBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt


class JSONBuilderThread(QThread):
    progress_updated = pyqtSignal(int, int)
    log_updated = pyqtSignal(str)
    status_updated = pyqtSignal(str)
    course_completed = pyqtSignal(str, str)

    def __init__(self, target_json="courses.json"):
        super().__init__()
        self.target_json = target_json
        self.is_running = True
        self.is_paused = False
        
        self.model_name = 'qwen2.5:7b'
        self.cooldown_interval = 15 * 60  
        self.cooldown_duration = 60     
        
        self.prev_course_json = ""
        self.curr_course_json = ""

    def load_raw_courses(self):
        """Зчитує всі текстові файли .txt у поточній папці."""
        files = glob.glob("*.txt")
        courses_data = []
        
        for filepath in files:
            if filepath == "requirements.txt":
                continue
                
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            blocks = content.split('## ТЕГ:')[1:]
            for block in blocks:
                try:
                    lines = block.strip().split('\n')
                    tags_raw = lines[0].strip()
                    tags = [t.strip() for t in tags_raw.split(',') if t.strip()]
                    
                    title_match = re.search(r'НАЗВА:\s*(.+)', block)
                    title = title_match.group(1).strip() if title_match else "Без назви"
                    
                    intro_match = re.search(r'(ВСТУП:.*?)(?=ЗМІСТ:|$)', block, re.DOTALL | re.IGNORECASE)
                    content_match = re.search(r'(ЗМІСТ:.*?)(?=ДЕТАЛІ:|DETAI:|$)', block, re.DOTALL | re.IGNORECASE)
                    details_match = re.search(r'(ДЕТАЛІ:.*|DETAI:.*)', block, re.DOTALL | re.IGNORECASE)
                    
                    intro = intro_match.group(1).strip() if intro_match else ""
                    content_text = content_match.group(1).strip() if content_match else ""
                    details = details_match.group(1).strip() if details_match else ""
                    
                    intro = re.sub(r'^((?:ВСТУП:)\s*)+', 'ВСТУП: ', intro, flags=re.IGNORECASE).strip()
                    content_text = re.sub(r'^((?:ЗМІСТ:)\s*)+', 'ЗМІСТ: ', content_text, flags=re.IGNORECASE).strip()
                    details = re.sub(r'^((?:ДЕТАЛІ:|DETAI:)\s*)+', 'ДЕТАЛІ: ', details, flags=re.IGNORECASE).strip()
                    
                    full_desc = f"{intro}\n\n{content_text}\n\n{details}".strip()
                    
                    if len(full_desc) > 50:
                        courses_data.append({
                            "title": title,
                            "tags": tags,
                            "description": full_desc
                        })
                except Exception:
                    continue
        return courses_data

    def validate_llm_json(self, text):
        """Перевіряє, чи Qwen повернув валідний JSON з потрібними ключами."""
        try:
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            
            data = json.loads(text.strip())
            
            required_keys = ["Price", "Difficulty", "Duration", "Formats"]
            for key in required_keys:
                if key not in data:
                    return None
            
            if not isinstance(data["Formats"], list):
                return None
                
            return data
        except Exception:
            return None

    def run(self):
        self.status_updated.emit("Зчитування сирих файлів...")
        raw_courses = self.load_raw_courses()
        
        if len(raw_courses) == 0:
            self.status_updated.emit("Не знайдено жодного курсу у текстових файлах!")
            return

        processed_courses = []
        course_id_counter = 1
        try:
            with open(self.target_json, "r", encoding="utf-8") as f:
                processed_courses = json.load(f)
                course_id_counter = len(processed_courses) + 1
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        processed_titles = {c.get("Назва курсу") for c in processed_courses}
        tasks_to_do = [c for c in raw_courses if c["title"] not in processed_titles]
        
        completed_tasks = len(processed_courses)
        total_tasks = completed_tasks + len(tasks_to_do)

        last_rest_time = time.time()
        self.status_updated.emit("Генерація метаданих активна (Ollama)...")

        for item in tasks_to_do:
            if not self.is_running:
                self.status_updated.emit("Генерацію скасовано.")
                break
            
            while self.is_paused:
                time.sleep(0.5)
                if not self.is_running:
                    return

            if time.time() - last_rest_time > self.cooldown_interval:
                self.status_updated.emit("Охолодження GPU (1 хв)...")
                self.log_updated.emit("\n--- ПАУЗА НА ОХОЛОДЖЕННЯ (60 сек) ---")
                time.sleep(self.cooldown_duration)
                last_rest_time = time.time()
                self.status_updated.emit("Генерація метаданих активна (Ollama)...")

            title = item["title"]
            tags = ", ".join(item["tags"])
            
            prompt = f"""Ти аналітична система. Твоє завдання згенерувати УНІКАЛЬНІ метадані для освітнього курсу на основі його назви та тегів.
Назва курсу: {title}
Теги: {tags}

Проаналізуй складність теми та згенеруй ВИКЛЮЧНО валідний JSON об'єкт. НЕ пиши жодного тексту окрім JSON.
Ключі у JSON мають бути строго такими: "Price", "Difficulty", "Duration", "Formats".

Правила генерації (унікальні для кожного курсу!):
- Price: Float (Згенеруй випадкову, але логічну ціну в ГРИВНЯХ від 400.00 до 15000.00 залежно від складності та ніші)
- Difficulty: String (Обери одне: "Базовий", "Середній", або "Просунутий")
- Duration: Integer (Згенеруй тривалість у хвилинах від 60 до 3000, уникай банальних круглих чисел)
- Formats: Array of Strings (Підбери від 1 до 5 форматів навчання, що ідеально підходять темі, наприклад: "Відеолекції", "Практичні завдання", "Інтерактивні тести", "Код-рев'ю", "Аудіо-подкасти", "Лонгріди" тощо).

Формат відповіді:
{{
  "Price": число,
  "Difficulty": "рядок", 
  "Duration": число,
  "Formats": ["формат_1", "формат_2", "..."]
}}
"""
            
            llm_data = None
            retries = 0
            while llm_data is None and retries < 3:
                try:
                    response = ollama.chat(
                        model=self.model_name, 
                        messages=[{'role': 'user', 'content': prompt}],
                        options={'temperature': 0.6} 
                    )
                    llm_data = self.validate_llm_json(response['message']['content'])
                except Exception as e:
                    self.log_updated.emit(f"Помилка з'єднання: {e}")
                
                if llm_data is None:
                    retries += 1
                    self.log_updated.emit(f"⚠️ Невдалий JSON для '{title}', спроба {retries}/3...")
                    time.sleep(1)

            if llm_data is None:
                llm_data = {
                    "Price": 99.99, "Difficulty": "Середній", "Duration": 180,
                    "Formats": ["Відео", "Текст"]
                }
                self.log_updated.emit(f"⚠️ Використано дефолтні метадані для '{title}'")

            final_description = item["description"]
            
            if "[СКЛАДНІСТЬ]" in final_description:
                final_description = final_description.replace("[СКЛАДНІСТЬ]", llm_data["Difficulty"].lower())
            
            if "[ТРИВАЛІСТЬ]" in final_description:
                final_description = final_description.replace("[ТРИВАЛІСТЬ]", f"{llm_data['Duration']} хвилин")
                
            if "[ФОРМАТИ]" in final_description:
                formats_str = ", ".join(llm_data["Formats"]).lower() if llm_data["Formats"] else "відео та тексти"
                final_description = final_description.replace("[ФОРМАТИ]", formats_str)

            final_course = {
                "Course_ID": course_id_counter,
                "Назва курсу": title,
                "Текстовий опис": final_description,
                "Теги": item["tags"],
                "Ціна": llm_data["Price"],
                "Складність": llm_data["Difficulty"],
                "Тривалість": llm_data["Duration"],
                "Формати матеріалів курсу": llm_data["Formats"],
                "Кількість взаємодій": 0,
                "Середній рейтинг": 0.0,
                "Деталізована історія взаємодій": {
                    "ID коментарів": []
                }
            }

            processed_courses.append(final_course)
            
            with open(self.target_json, "w", encoding="utf-8") as f:
                json.dump(processed_courses, f, ensure_ascii=False, indent=4)

            self.prev_course_json = self.curr_course_json
            self.curr_course_json = json.dumps(final_course, ensure_ascii=False, indent=2)

            course_id_counter += 1
            completed_tasks += 1
            
            self.progress_updated.emit(completed_tasks, total_tasks)
            self.course_completed.emit(self.curr_course_json, self.prev_course_json)
            self.log_updated.emit(f"✅ Оброблено: {title} (Ціна: {llm_data['Price']}, Час: {llm_data['Duration']})")

        if self.is_running:
            self.status_updated.emit(f"Успішно завершено! Файл {self.target_json} готовий.")

    def pause(self):
        self.is_paused = True
        self.status_updated.emit("Пауза...")

    def resume(self):
        self.is_paused = False
        self.status_updated.emit("Генерація активна (Ollama)...")
        
    def stop(self):
        self.is_running = False
        self.is_paused = False

class JSONAppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Генератор бази курсів (JSON)")
        self.setGeometry(100, 100, 1200, 800)
        self.initUI()
        
    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        self.status_label = QLabel("Статус: Готовий до запуску", self)
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)
        
        self.progress_text = QLabel("0 / 0 курсів оброблено", self)
        self.progress_text.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_text)

        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("▶ Запустити")
        self.btn_start.clicked.connect(self.start_generation)
        btn_layout.addWidget(self.btn_start)
        
        self.btn_pause = QPushButton("⏸ Пауза")
        self.btn_pause.clicked.connect(self.pause_generation)
        self.btn_pause.setEnabled(False)
        btn_layout.addWidget(self.btn_pause)
        
        self.btn_resume = QPushButton("⏯ Продовжити")
        self.btn_resume.clicked.connect(self.resume_generation)
        self.btn_resume.setEnabled(False)
        btn_layout.addWidget(self.btn_resume)
        
        self.btn_stop = QPushButton("⏹ Зупинити")
        self.btn_stop.clicked.connect(self.stop_generation)
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)

        preview_layout = QHBoxLayout()
        
        self.prev_box = QGroupBox("Попередній збережений JSON")
        prev_layout = QVBoxLayout()
        self.prev_text_edit = QTextEdit()
        self.prev_text_edit.setReadOnly(True)
        self.prev_text_edit.setStyleSheet("background-color: #2E2E2E; color: #D4D4D4; font-family: Consolas; font-size: 12px;")
        prev_layout.addWidget(self.prev_text_edit)
        self.prev_box.setLayout(prev_layout)
        
        self.curr_box = QGroupBox("Поточний збережений JSON")
        curr_layout = QVBoxLayout()
        self.curr_text_edit = QTextEdit()
        self.curr_text_edit.setReadOnly(True)
        self.curr_text_edit.setStyleSheet("background-color: #1E1E1E; color: #4CAF50; font-family: Consolas; font-size: 12px;")
        curr_layout.addWidget(self.curr_text_edit)
        self.curr_box.setLayout(curr_layout)
        
        preview_layout.addWidget(self.prev_box)
        preview_layout.addWidget(self.curr_box)
        layout.addLayout(preview_layout, stretch=1)

        self.log_edit = QTextEdit(self)
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(120)
        self.log_edit.setStyleSheet("background-color: #1A1A1A; color: #00BCD4; font-family: Consolas; font-size: 11px;")
        layout.addWidget(self.log_edit)

    def update_displays(self, curr_json, prev_json):
        self.prev_text_edit.setPlainText(prev_json)
        self.curr_text_edit.setPlainText(curr_json)

    def append_log(self, text):
        self.log_edit.append(text)
        scrollbar = self.log_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_text.setText(f"{current} / {total} оброблено")

    def update_status(self, text):
        self.status_label.setText(f"Статус: {text}")

    def start_generation(self):
        self.thread = JSONBuilderThread()
        self.thread.progress_updated.connect(self.update_progress)
        self.thread.log_updated.connect(self.append_log)
        self.thread.status_updated.connect(self.update_status)
        self.thread.course_completed.connect(self.update_displays)
        self.thread.start()
        
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)

    def pause_generation(self):
        self.thread.pause()
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(True)

    def resume_generation(self):
        self.thread.resume()
        self.btn_resume.setEnabled(False)
        self.btn_pause.setEnabled(True)

    def stop_generation(self):
        self.thread.stop()
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.append_log("--- ЗУПИНЕНО ---")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = JSONAppWindow()
    window.show()
    sys.exit(app.exec_())