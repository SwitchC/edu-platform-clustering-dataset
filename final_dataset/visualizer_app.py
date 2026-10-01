import sys
import json
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QTabWidget, QScrollArea, QGridLayout, QMessageBox)

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig, self.axes = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.setParent(parent)
        fig.patch.set_facecolor('#f0f0f0')
        self.axes.set_facecolor('#ffffff')
        
        self.setMinimumHeight(400)

    def plot_histogram(self, data, title, xlabel, ylabel, bins=20, color='skyblue'):
        """Будує гістограму розподілу для числових даних"""
        self.axes.clear()
        clean_data = [x for x in data if x is not None]
        
        if clean_data:
            self.axes.hist(clean_data, bins=bins, color=color, edgecolor='black', alpha=0.7)
        
        self.axes.set_title(title, fontsize=12, fontweight='bold')
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.grid(axis='y', linestyle='--', alpha=0.7)
        
        self.figure.tight_layout(pad=2.0)
        self.draw()

    def plot_bar(self, data, title, xlabel, ylabel, color='lightgreen', rotation=0, ha='center', labels_fontsize=10):
        """Будує стовпчикову діаграму з підрахунком частоти входжень (категоріальні дані)"""
        self.axes.clear()
        clean_data = [x for x in data if x is not None]
        
        if clean_data:
            counts = Counter(clean_data)
            counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
            labels = [str(k) for k in counts.keys()]
            values = list(counts.values())
            
            self.axes.bar(labels, values, color=color, edgecolor='black', alpha=0.7)
            
            self.axes.set_xticks(range(len(labels)))
            self.axes.set_xticklabels(labels, rotation=rotation, ha=ha, fontsize=labels_fontsize)
            
        self.axes.set_title(title, fontsize=12, fontweight='bold')
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.grid(axis='y', linestyle='--', alpha=0.7)
        
        self.figure.tight_layout(pad=2.0)
        self.draw()

    def plot_bar_direct(self, labels, values, title, xlabel, ylabel, color='gold'):
        self.axes.clear()
        self.axes.bar(labels, values, color=color, edgecolor='black', alpha=0.7)
        self.axes.tick_params(axis='x', rotation=0)
        
        self.axes.set_title(title, fontsize=12, fontweight='bold')
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.grid(axis='y', linestyle='--', alpha=0.7)
        
        self.figure.tight_layout(pad=2.0)
        self.draw()


class VisualizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Science Visualizer: Аналіз Датасету")
        self.setGeometry(100, 100, 1200, 800)

        self.users = self.load_json("users_raw.json")
        self.courses = self.load_json("courses_raw.json")

        if not self.users or not self.courses:
            QMessageBox.critical(self, "Помилка", "Не знайдено файли users_raw.json або courses_raw.json.")
            sys.exit()

        self.initUI()

    def load_json(self, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Помилка завантаження {filename}: {e}")
            return None

    def initUI(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tab_users = QWidget()
        self.setup_users_tab()
        self.tabs.addTab(self.tab_users, "👤 Користувачі")

        self.tab_courses = QWidget()
        self.setup_courses_tab()
        self.tabs.addTab(self.tab_courses, "📚 Курси")

    def setup_users_tab(self):
        layout = QVBoxLayout(self.tab_users)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QWidget()
        grid = QGridLayout(content_widget)

        ages = [u.get("Вік") for u in self.users]
        edus = [u.get("Рівень освіти") for u in self.users]
        goals = [u.get("Мета навчання") for u in self.users]
        u_types = [u.get("User_Type") for u in self.users]

        c_age = PlotCanvas(self, width=5, height=4)
        c_age.plot_histogram(ages, "Розподіл за Віком", "Вік", "К-ть користувачів", bins=20, color='#ff9999')
        grid.addWidget(c_age, 0, 0)

        c_edu = PlotCanvas(self, width=5, height=4)
        c_edu.plot_bar(edus, "Рівень освіти", "Освіта", "К-ть користувачів", color='#66b3ff', rotation=15, ha='right')
        grid.addWidget(c_edu, 0, 1)

        c_goal = PlotCanvas(self, width=5, height=4)
        c_goal.plot_bar(goals, "Мета навчання", "Мета", "К-ть користувачів", color='#99ff99', rotation=15, ha='right')
        grid.addWidget(c_goal, 1, 0)

        c_type = PlotCanvas(self, width=5, height=4)
        c_type.plot_bar(u_types, "Активність (User Type)", "Тип", "К-ть користувачів", color='#ffcc99')
        grid.addWidget(c_type, 1, 1)

        bought = [len(u.get("Деталізована історія взаємодій", {}).get("Куплені курси", [])) for u in self.users]
        completed = [len(u.get("Деталізована історія взаємодій", {}).get("Пройдені курси", [])) for u in self.users]
        wishlist = [len(u.get("Деталізована історія взаємодій", {}).get("Додані в бажане", [])) for u in self.users]
        clicks = [len(u.get("Деталізована історія взаємодій", {}).get("Кліки за курсами", [])) for u in self.users]
        ratings = [len(u.get("Деталізована історія взаємодій", {}).get("Історія оцінювання", [])) for u in self.users]
        comments = [len(u.get("Деталізована історія взаємодій", {}).get("ID залишених коментарів", [])) for u in self.users]

        c_bought = PlotCanvas(self, width=5, height=4)
        c_bought.plot_histogram(bought, "К-ть куплених курсів (на юзера)", "Куплено", "К-ть юзерів", bins=15, color='#c2c2f0')
        grid.addWidget(c_bought, 2, 0)

        c_comp = PlotCanvas(self, width=5, height=4)
        c_comp.plot_histogram(completed, "К-ть пройдених курсів", "Пройдено", "К-ть юзерів", bins=15, color='#ffb3e6')
        grid.addWidget(c_comp, 2, 1)

        c_wish = PlotCanvas(self, width=5, height=4)
        c_wish.plot_histogram(wishlist, "Обсяг Wishlist", "Курсів у бажаному", "К-ть юзерів", bins=15, color='#c2f0c2')
        grid.addWidget(c_wish, 3, 0)

        c_clicks = PlotCanvas(self, width=5, height=4)
        c_clicks.plot_histogram(clicks, "Кількість унікальних кліків", "Кліки", "К-ть юзерів", bins=20, color='#f0c2c2')
        grid.addWidget(c_clicks, 3, 1)

        c_ratings = PlotCanvas(self, width=5, height=4)
        c_ratings.plot_histogram(ratings, "Кількість залишених оцінок", "Оцінки", "К-ть юзерів", bins=10, color='#f0e6c2')
        grid.addWidget(c_ratings, 4, 0)

        c_comments = PlotCanvas(self, width=5, height=4)
        c_comments.plot_histogram(comments, "Кількість залишених коментарів", "Коментарі", "К-ть юзерів", bins=5, color='#c2f0f0')
        grid.addWidget(c_comments, 4, 1)

        monthly_spent_events = []
        monthly_time_events = []
        
        for u in self.users:
            spent = u.get("Загальні витрати по місяцях")
            if spent:
                monthly_spent_events.extend([s for s in spent if s is not None and s > 0])
                
            time_spent = u.get("Загальний час навчання по місяцях")
            if time_spent:
                monthly_time_events.extend([t for t in time_spent if t is not None and t > 0])

        c_spent_monthly = PlotCanvas(self, width=5, height=4)
        c_spent_monthly.plot_histogram(monthly_spent_events, "Витрачені кошти (за один активний місяць)", "Витрати (грн)", "К-ть випадків", bins=30, color='#8fbc8f')
        grid.addWidget(c_spent_monthly, 5, 0)

        c_time_monthly = PlotCanvas(self, width=5, height=4)
        c_time_monthly.plot_histogram(monthly_time_events, "Витрачений час (за один активний місяць)", "Час (хвилини)", "К-ть випадків", bins=30, color='#dda0dd')
        grid.addWidget(c_time_monthly, 5, 1)

        monthly_global_spent = [0.0] * 12
        for u in self.users:
            user_months = u.get("Загальні витрати по місяцях", [0]*12)
            for m in range(12):
                if user_months[m] is not None:
                    monthly_global_spent[m] += user_months[m]

        months_labels = [f"Міс. {i+1}" for i in range(12)]
        
        c_spent_month_total = PlotCanvas(self, width=10, height=4)
        c_spent_month_total.setMinimumHeight(450)
        c_spent_month_total.plot_bar_direct(months_labels, monthly_global_spent, "Загальні витрати всіх юзерів по місяцях (Прибуток платформи)", "Місяць", "Сума (грн)", color='#ffd700')
        grid.addWidget(c_spent_month_total, 6, 0, 1, 2)

        scroll.setWidget(content_widget)
        layout.addWidget(scroll)

    def setup_courses_tab(self):
        layout = QVBoxLayout(self.tab_courses)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QWidget()
        grid = QGridLayout(content_widget)

        prices = [c.get("Ціна") for c in self.courses]
        durations = [c.get("Тривалість") for c in self.courses]
        difficulties = [c.get("Складність") for c in self.courses]
        avg_ratings = [c.get("Середній рейтинг") for c in self.courses if c.get("Середній рейтинг", 0) > 0]

        c_price = PlotCanvas(self, width=5, height=4)
        c_price.plot_histogram(prices, "Розподіл Ціни курсів", "Ціна (Грн)", "К-ть курсів", bins=20, color='#8c564b')
        grid.addWidget(c_price, 0, 0)

        c_dur = PlotCanvas(self, width=5, height=4)
        c_dur.plot_histogram(durations, "Розподіл Тривалості", "Хвилини", "К-ть курсів", bins=20, color='#e377c2')
        grid.addWidget(c_dur, 0, 1)

        c_diff = PlotCanvas(self, width=5, height=4)
        c_diff.plot_bar(difficulties, "Розподіл за Складністю", "Складність", "К-ть курсів", color='#7f7f7f')
        grid.addWidget(c_diff, 1, 0)

        c_rating = PlotCanvas(self, width=5, height=4)
        c_rating.plot_histogram(avg_ratings, "Розподіл Середнього Рейтингу", "Рейтинг (1-5)", "К-ть курсів", bins=10, color='#bcbd22')
        grid.addWidget(c_rating, 1, 1)

        combined_tags = []
        for c in self.courses:
            tags = c.get("Теги")
            if tags and len(tags) > 0:
                combined_str = " + ".join(sorted(tags))
                combined_tags.append(combined_str)
            else:
                combined_tags.append("Без тегів")

        c_tags = PlotCanvas(self, width=10, height=4)
        c_tags.setMinimumHeight(750) 
        
        c_tags.plot_bar(combined_tags, "Кількість курсів за комбінаціями тегів", "Комбінація тегів", "К-ть курсів", 
                        color='#1f77b4', rotation=45, ha='right', labels_fontsize=8)
        
        grid.addWidget(c_tags, 2, 0, 1, 2)

        course_clicks = {c["Course_ID"]: 0 for c in self.courses}
        course_buys = {c["Course_ID"]: 0 for c in self.courses}
        course_completed = {c["Course_ID"]: 0 for c in self.courses}
        course_wishlist = {c["Course_ID"]: 0 for c in self.courses}

        for u in self.users:
            history = u.get("Деталізована історія взаємодій", {})
            for cid in history.get("Кліки за курсами", []):
                if cid in course_clicks: course_clicks[cid] += 1
            for cid in history.get("Куплені курси", []):
                if cid in course_buys: course_buys[cid] += 1
            for cid in history.get("Пройдені курси", []):
                if cid in course_completed: course_completed[cid] += 1
            for cid in history.get("Додані в бажане", []):
                if cid in course_wishlist: course_wishlist[cid] += 1

        c_cclicks = PlotCanvas(self, width=5, height=4)
        c_cclicks.plot_histogram(list(course_clicks.values()), "Розподіл Кліків по курсах", "К-ть кліків на курс", "К-ть курсів", bins=30, color='#17becf')
        grid.addWidget(c_cclicks, 3, 0)

        c_cbuys = PlotCanvas(self, width=5, height=4)
        c_cbuys.plot_histogram(list(course_buys.values()), "Розподіл Купівель по курсах", "К-ть купівель курсу", "К-ть курсів", bins=30, color='#ff7f0e')
        grid.addWidget(c_cbuys, 3, 1)

        c_ccomp = PlotCanvas(self, width=5, height=4)
        c_ccomp.plot_histogram(list(course_completed.values()), "Розподіл Проходжень", "К-ть разів пройдено", "К-ть курсів", bins=30, color='#2ca02c')
        grid.addWidget(c_ccomp, 4, 0)

        c_cwish = PlotCanvas(self, width=5, height=4)
        c_cwish.plot_histogram(list(course_wishlist.values()), "Популярність у Wishlist", "Додавань у бажане", "К-ть курсів", bins=30, color='#d62728')
        grid.addWidget(c_cwish, 4, 1)

        scroll.setWidget(content_widget)
        layout.addWidget(scroll)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = VisualizerApp()
    window.show()
    sys.exit(app.exec_())