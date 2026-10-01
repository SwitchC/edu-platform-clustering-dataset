import sys
import time
import json
import ollama
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QProgressBar, 
                             QTextEdit, QGroupBox, QGridLayout, QScrollArea)
from PyQt5.QtCore import QThread, pyqtSignal, Qt


def format_time(seconds):
    if seconds < 60:
        return f"{int(seconds)} сек."
    elif seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m} хв. {s} сек."
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h} год. {m} хв."


class LLMGeneratorThread(QThread):
    progress_updated = pyqtSignal(int, int)
    log_updated = pyqtSignal(str)
    status_updated = pyqtSignal(str)
    tag_distribution_updated = pyqtSignal(dict, int)
    course_generated = pyqtSignal(str, str) 
    eta_updated = pyqtSignal(str)

    def __init__(self, target_file="gemini_texts_2.txt", tasks_file="tasks.json"):
        super().__init__()
        self.target_file = target_file
        self.tasks_file = tasks_file
        self.is_running = True
        self.is_paused = False
        
        self.model_name = 'qwen2.5:7b'
        self.cooldown_interval = 15 * 60
        self.cooldown_duration = 60      
    def run(self):
        self.status_updated.emit("Завантаження завдань...")
        
        try:
            with open(self.tasks_file, "r", encoding="utf-8") as f:
                tasks = json.load(f)
        except FileNotFoundError:
            self.status_updated.emit(f"Помилка: Не знайдено {self.tasks_file}")
            return

        tag_counts, hybrid_count = self.get_current_counts()
        self.tag_distribution_updated.emit(tag_counts, hybrid_count)
        
        total_tasks = len(tasks)
        completed_tasks = 0
        last_rest_time = time.time()
        recent_task_times = []

        self.status_updated.emit("Генерація активна (Ollama)...")

        with open(self.target_file, "a", encoding="utf-8") as f:
            for item in tasks:
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
                    self.status_updated.emit("Генерація активна (Ollama)...")

                tags = item.get("tags", [])
                title = item.get("title", "Без назви")
                
                tag_str = ", ".join(tags)
                tag_key = " + ".join(tags)

                context_instruction = (
                    f"Курс є міждисциплінарним на перетині тем: {tag_str}." 
                    if len(tags) > 1 
                    else f"Тема курсу: '{tag_str}'."
                )

                prompt = f"""Ти методист освітньої платформи. Напиши логічний опис навчального курсу.
{context_instruction}
Назва курсу: {title}

Вимоги:
1. Текст обов'язково має розкривати зв'язок зазначених тем із назвою курсу.
2. Структура — строго 3 блоки: ВСТУП (3-4 речення), ЗМІСТ (3-4 речення), ДЕТАЛІ (3-4 речення).

Формат відповіді:
ВСТУП: [текст]
ЗМІСТ: [текст]
ДЕТАЛІ: [текст]"""
                
                task_start_time = time.time()
                try:
                    response = ollama.chat(
                        model=self.model_name, 
                        messages=[{'role': 'user', 'content': prompt}],
                        options={'num_predict': 400} 
                    )
                    generated_text = response['message']['content'].strip()
                    
                    entry = f"\n\n## ТЕГ: {tag_str}\nНАЗВА: {title}\n{generated_text}"
                    f.write(entry)
                    f.flush()
                    
                    tag_counts[tag_key] = tag_counts.get(tag_key, 0) + 1
                    if len(tags) > 1:
                        hybrid_count += 1
                        
                    completed_tasks += 1
                    
                    task_duration = time.time() - task_start_time
                    recent_task_times.append(task_duration)
                    if len(recent_task_times) > 15:
                        recent_task_times.pop(0)
                        
                    avg_time_per_task = sum(recent_task_times) / len(recent_task_times)
                    remaining_tasks = total_tasks - completed_tasks
                    estimated_remaining_seconds = remaining_tasks * avg_time_per_task
                    
                    if avg_time_per_task > 0:
                        work_time_left = remaining_tasks * avg_time_per_task
                        cooldowns_left = work_time_left / self.cooldown_interval
                        estimated_remaining_seconds += cooldowns_left * self.cooldown_duration

                    eta_string = format_time(estimated_remaining_seconds)
                    
                    self.progress_updated.emit(completed_tasks, total_tasks)
                    self.tag_distribution_updated.emit(tag_counts, hybrid_count)
                    self.eta_updated.emit(eta_string)
                    self.log_updated.emit(f"✅ [{tag_key}] {title} ({task_duration:.1f}с)")
                    self.course_generated.emit(f"[{tag_key}] {title}", generated_text)
                    
                except Exception as e:
                    self.log_updated.emit(f"❌ Помилка для '{title}': {str(e)}")
                    time.sleep(2)

        if self.is_running:
            self.status_updated.emit("Генерацію завершено!")

    def get_current_counts(self):
        counts = {}
        hybrid_count = 0
        try:
            with open(self.target_file, 'r', encoding='utf-8') as file:
                for line in file:
                    if "ТЕГ:" in line:
                        raw_tags = line.split("ТЕГ:")[-1].strip()
                        tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
                        tag_key = " + ".join(tags)
                        counts[tag_key] = counts.get(tag_key, 0) + 1
                        if len(tags) > 1:
                            hybrid_count += 1
        except FileNotFoundError:
            pass
        return counts, hybrid_count

    def pause(self):
        self.is_paused = True
        self.status_updated.emit("Пауза...")

    def resume(self):
        self.is_paused = False
        self.status_updated.emit("Генерація активна...")
        
    def stop(self):
        self.is_running = False
        self.is_paused = False

class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Генератор датасету мікрокурсів (Ollama + ETA)")
        self.setGeometry(100, 100, 1000, 850)
        self.initUI()
        
    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        top_layout = QHBoxLayout()
        self.status_label = QLabel("Статус: Готовий до запуску", self)
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        top_layout.addWidget(self.status_label)
        
        self.eta_label = QLabel("Очікуваний час: —", self)
        self.eta_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #007ACC;")
        self.eta_label.setAlignment(Qt.AlignRight)
        top_layout.addWidget(self.eta_label)
        layout.addLayout(top_layout)
        
        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)
        
        self.progress_text = QLabel("0 / 0 курсів", self)
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

        self.tag_group = QGroupBox("Покриття тем та суміжних курсів у файлі")
        group_layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        self.tag_layout = QGridLayout(scroll_widget)
        scroll.setWidget(scroll_widget)
        group_layout.addWidget(scroll)
        self.tag_group.setLayout(group_layout)
        layout.addWidget(self.tag_group, stretch=1)

        preview_layout = QHBoxLayout()
        self.prev_course_box = QGroupBox("Попередній згенерований курс")
        prev_layout = QVBoxLayout()
        self.prev_text_edit = QTextEdit()
        self.prev_text_edit.setReadOnly(True)
        self.prev_text_edit.setStyleSheet("background-color: #2E2E2E; color: #D4D4D4; font-size: 13px; font-family: Segoe UI;")
        prev_layout.addWidget(self.prev_text_edit)
        self.prev_course_box.setLayout(prev_layout)
        
        self.last_course_box = QGroupBox("Останній згенерований курс")
        last_layout = QVBoxLayout()
        self.last_text_edit = QTextEdit()
        self.last_text_edit.setReadOnly(True)
        self.last_text_edit.setStyleSheet("background-color: #1E1E1E; color: #4CAF50; font-size: 13px; font-weight: bold; font-family: Segoe UI;")
        last_layout.addWidget(self.last_text_edit)
        self.last_course_box.setLayout(last_layout)
        
        preview_layout.addWidget(self.prev_course_box)
        preview_layout.addWidget(self.last_course_box)
        layout.addLayout(preview_layout, stretch=2)

        self.log_edit = QTextEdit(self)
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(80)
        self.log_edit.setStyleSheet("background-color: #1A1A1A; color: #00BCD4; font-family: Consolas; font-size: 11px;")
        layout.addWidget(self.log_edit)

    def update_course_displays(self, title, text):
        self.prev_text_edit.setPlainText(self.last_text_edit.toPlainText())
        formatted_text = f"НАЗВА: {title}\n\n{text}"
        self.last_text_edit.setPlainText(formatted_text)

    def update_eta(self, eta_str):
        self.eta_label.setText(f"Очікуваний час: {eta_str}")

    def update_tag_distribution(self, tag_counts, hybrid_count):
        for i in reversed(range(self.tag_layout.count())): 
            self.tag_layout.itemAt(i).widget().setParent(None)
            
        row, col = 0, 0
        hybrid_label = QLabel(f"<b style='color: #007ACC;'>🔗 Всього суміжних:</b> {hybrid_count} шт.")
        self.tag_layout.addWidget(hybrid_label, row, col)
        col += 1

        sorted_tags = sorted(tag_counts.items(), key=lambda x: ("+" in x[0], x[0]))

        for tag, count in sorted_tags:
            lbl = QLabel(f"<b>{tag}:</b> {count}")
            self.tag_layout.addWidget(lbl, row, col)
            col += 1
            if col > 2:  
                col = 0
                row += 1

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
        self.thread = LLMGeneratorThread()
        self.thread.progress_updated.connect(self.update_progress)
        self.thread.log_updated.connect(self.append_log)
        self.thread.status_updated.connect(self.update_status)
        self.thread.tag_distribution_updated.connect(self.update_tag_distribution)
        self.thread.course_generated.connect(self.update_course_displays)
        self.thread.eta_updated.connect(self.update_eta)
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
    window = AppWindow()
    window.show()
    sys.exit(app.exec_())