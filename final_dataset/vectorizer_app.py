import sys
import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QTabWidget, QTableWidget, QTableWidgetItem, 
                             QTextEdit, QDialog, QListWidget, QSplitter, QMessageBox)
from PyQt5.QtCore import Qt

import spacy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class DataVectorizer:
    def __init__(self):
        self.users_orig = []
        self.courses_orig = []
        self.users_vec = []
        self.courses_vec = []
        
        self.user_vec_fields = []
        self.course_vec_fields = []
        
        try:
            self.nlp = spacy.load("uk_core_news_sm")
        except OSError:
            self.nlp = None

    def load_data(self):
        users_path = os.path.join(BASE_DIR, "users_raw.json")
        courses_path = os.path.join(BASE_DIR, "courses_raw.json")
        try:
            with open(users_path, "r", encoding="utf-8") as f:
                self.users_orig = json.load(f)
            with open(courses_path, "r", encoding="utf-8") as f:
                self.courses_orig = json.load(f)
            return True
        except Exception as e:
            print(f"Помилка читання оригінальних файлів: {e}")
            return False

    def load_vectorized_data(self):
        vec_dir = os.path.join(BASE_DIR, 'vectorized')
        u_path = os.path.join(vec_dir, 'users_vectorized.json')
        c_path = os.path.join(vec_dir, 'courses_vectorized.json')
        try:
            with open(u_path, "r", encoding="utf-8") as f:
                self.users_vec = json.load(f)
            with open(c_path, "r", encoding="utf-8") as f:
                self.courses_vec = json.load(f)
                
            if self.users_vec:
                self.user_vec_fields = list(self.users_vec[0].keys())
            for c in self.courses_vec:
                if c and "Статус" not in c:
                    self.course_vec_fields = list(c.keys())
                    break
            return True
        except Exception as e:
            print(f"Помилка читання векторизованих файлів: {e}")
            return False

    def vectorize_users(self):
        df = pd.DataFrame(self.users_orig)
        df_vec = pd.DataFrame()
        
        df_vec['User_ID'] = df['User_ID']
        
        df['Вік'] = pd.to_numeric(df['Вік'], errors='coerce')
        df['Вік'] = df['Вік'].fillna(df['Вік'].mean())
        df_vec['Вік_norm'] = StandardScaler().fit_transform(df[['Вік']])
        
        edu_map = {'школяр': 1, 'студент': 2, 'бакалавр': 3, 'магістр': 4, 'аспірант': 5}
        mode_edu = df['Рівень освіти'].mode()[0] if not df['Рівень освіти'].empty else 'студент'
        df['Рівень освіти'] = df['Рівень освіти'].fillna(mode_edu).astype(str).str.lower()
        df['edu_num'] = df['Рівень освіти'].map(edu_map).fillna(1)
        df_vec['Освіта_norm'] = StandardScaler().fit_transform(df[['edu_num']])
        
        df['Мета навчання'] = df['Мета навчання'].fillna('Unknown')
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        goal_encoded = ohe.fit_transform(df[['Мета навчання']])
        for i, col_name in enumerate(ohe.get_feature_names_out(['Мета навчання'])):
            df_vec[col_name] = goal_encoded[:, i]
            
        df['Бажані формати навчання'] = df['Бажані формати навчання'].apply(lambda x: x if isinstance(x, list) else [])
        mlb_format = MultiLabelBinarizer()
        format_encoded = mlb_format.fit_transform(df['Бажані формати навчання'])
        for i, col_name in enumerate(mlb_format.classes_):
            df_vec[f'Формат_{col_name}'] = format_encoded[:, i]
            
        df['час_avg'] = df['Загальний час навчання по місяцях'].apply(
            lambda x: np.mean(x[-6:]) if isinstance(x, list) and len(x) >= 6 else (np.mean(x) if isinstance(x, list) and x else 0.0))
        df_vec['Час_навчання_norm'] = StandardScaler().fit_transform(df[['час_avg']])
        
        df['витрати_avg'] = df['Загальні витрати по місяцях'].apply(
            lambda x: np.mean(x[-6:]) if isinstance(x, list) and len(x) >= 6 else (np.mean(x) if isinstance(x, list) and x else 0.0))
        df_vec['Витрати_norm'] = StandardScaler().fit_transform(df[['витрати_avg']])
        
        all_passed_tags = set()
        for d in df['Пройдені курси за тегами']:
            if isinstance(d, dict): all_passed_tags.update(d.keys())
            
        for tag in all_passed_tags:
            col_data = df['Пройдені курси за тегами'].apply(
                lambda d: np.mean(d.get(tag, [0]*6)[-6:]) if isinstance(d, dict) else 0.0)
            df_vec[f'Пройдено_тег_{tag}'] = StandardScaler().fit_transform(col_data.values.reshape(-1, 1)).flatten()

        all_wish_tags = set()
        for d in df['Додано в бажане за тегами']:
            if isinstance(d, dict): all_wish_tags.update(d.keys())
            
        for tag in all_wish_tags:
            col_data = df['Додано в бажане за тегами'].apply(
                lambda d: d.get(tag, 0) if isinstance(d, dict) else 0.0)
            df_vec[f'Бажане_тег_{tag}'] = StandardScaler().fit_transform(col_data.values.reshape(-1, 1)).flatten()
            
        self.users_vec = df_vec.to_dict(orient='records')
        self.user_vec_fields = list(df_vec.columns)
        
        print(f"✅ Успішно векторизовано користувачів: {len(self.users_vec)}")

    def vectorize_courses(self):
        df = pd.DataFrame(self.courses_orig)
        
        # --- Фільтрація китайських символів (> 50%) ---
        def is_too_chinese(title, desc):
            text = str(title).strip() + " " + str(desc).strip()
            if len(text.replace(" ", "")) == 0: return False
            ch_chars = re.findall(r'[\u4e00-\u9fff]', text)
            return (len(ch_chars) / len(text.replace(" ", ""))) > 0.5

        df['is_valid'] = df.apply(lambda row: not is_too_chinese(row.get('Назва курсу'), row.get('Текстовий опис')), axis=1)
        
        valid_df = df[df['is_valid']].copy()
        invalid_df = df[~df['is_valid']].copy()
        
        valid_vec = pd.DataFrame(index=valid_df.index)
        valid_vec['Course_ID'] = valid_df['Course_ID']
        
        # --- NLP Pipeline ---
        def clean_text(row):
            title = str(row.get('Назва курсу', '')).strip()
            desc = str(row.get('Текстовий опис', '')).strip()
            text = (title + " " + desc).lower()
            
            # Видаляємо ієрогліфи та залишаємо лише кирилицю/латиницю
            text = re.sub(r'[\u4e00-\u9fff]', ' ', text)
            text = re.sub(r'[^a-zа-яієїґ\s]', ' ', text)
            
            doc = self.nlp(text)
            clean_tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
            return " ".join(clean_tokens)

        valid_df['nlp_text'] = valid_df.apply(clean_text, axis=1)
        texts = valid_df['nlp_text'].tolist()

        temp_vect = TfidfVectorizer()
        temp_vect.fit(texts)
        idfs = temp_vect.idf_
        vocab = temp_vect.get_feature_names_out()
        
        n_drop = max(1, int(len(vocab) * 0.05))
        lowest_idf_indices = np.argsort(idfs)[:n_drop]
        drop_words = [vocab[i] for i in lowest_idf_indices]
        
        # Оптимізований TF-IDF з обмеженням словника
        final_vect = TfidfVectorizer(stop_words=drop_words, min_df=2, max_features=2500)
        tfidf_matrix = final_vect.fit_transform(texts)
        tfidf_dense = tfidf_matrix.toarray()
        
        # Виводимо всі токени словника TF-IDF у консоль
        feature_names = final_vect.get_feature_names_out()
        print("\n" + "="*50)
        print(f"📋 ПОВНИЙ СПИСОК ТОКЕНІВ TF-IDF ({len(feature_names)} шт.):")
        print(list(feature_names))
        print("="*50)
        
        print("\n--- СТАТИСТИКА ОБРОБКИ КУРСІВ ---")
        print(f"✅ Успішно векторизовано: {len(valid_df)}")
        print(f"❌ Відхилено (понад 50% ієрогліфів): {len(invalid_df)}")
        print(f"Відкинуто {len(drop_words)} найчастіших стоп-слів.")
        print(f"Фінальна довжина вектора TF-IDF: {tfidf_matrix.shape[1]}")
        print("="*50 + "\n")
        
        # --- Масштабування інших ознак ---
        valid_df['Теги'] = valid_df['Теги'].apply(lambda x: x if isinstance(x, list) else [])
        mlb_tags = MultiLabelBinarizer()
        tags_enc = mlb_tags.fit_transform(valid_df['Теги'])
        for i, col_name in enumerate(mlb_tags.classes_):
            valid_vec[f'Тег_{col_name}'] = tags_enc[:, i]
            
        valid_df['Ціна'] = pd.to_numeric(valid_df['Ціна'], errors='coerce')
        valid_df['Ціна'] = valid_df['Ціна'].fillna(valid_df['Ціна'].median())
        valid_vec['Ціна_norm'] = StandardScaler().fit_transform(valid_df[['Ціна']])
        
        comp_map = {'базовий': 1, 'середній': 2, 'просунутий': 3}
        mode_comp = valid_df['Складність'].mode()[0] if not valid_df['Складність'].empty else 'базовий'
        valid_df['Складність'] = valid_df['Складність'].fillna(mode_comp).astype(str).str.lower()
        valid_df['comp_num'] = valid_df['Складність'].map(comp_map).fillna(1)
        valid_vec['Складність_norm'] = StandardScaler().fit_transform(valid_df[['comp_num']])
        
        valid_df['Тривалість'] = pd.to_numeric(valid_df['Тривалість'], errors='coerce')
        valid_df['Тривалість'] = valid_df['Тривалість'].fillna(valid_df['Тривалість'].median())
        valid_vec['Тривалість_norm'] = StandardScaler().fit_transform(valid_df[['Тривалість']])
        
        valid_df['Формати матеріалів курсу'] = valid_df['Формати матеріалів курсу'].apply(lambda x: x if isinstance(x, list) else [])
        mlb_f = MultiLabelBinarizer()
        f_enc = mlb_f.fit_transform(valid_df['Формати матеріалів курсу'])
        for i, col_name in enumerate(mlb_f.classes_):
            valid_vec[f'ФорматКурсу_{col_name}'] = f_enc[:, i]

        self.course_vec_fields = list(valid_vec.columns) + [f"TF-IDF_vector (Length: {tfidf_matrix.shape[1]})"]
        
        self.courses_vec = [None] * len(df)
        valid_idx_map = {idx: i for i, idx in enumerate(valid_df.index)}
        
        for orig_idx, row in df.iterrows():
            if row['is_valid']:
                v_idx = valid_idx_map[orig_idx]
                rec = valid_vec.loc[orig_idx].to_dict()
                rec['TF-IDF_vector'] = tfidf_dense[v_idx].tolist()
                self.courses_vec[orig_idx] = rec
            else:
                self.courses_vec[orig_idx] = {
                    "Course_ID": row['Course_ID'],
                    "Статус": "Відхилено (понад 50% китайських символів)"
                }

    def save_vectorized(self):
        vec_dir = os.path.join(BASE_DIR, 'vectorized')
        os.makedirs(vec_dir, exist_ok=True)
        with open(os.path.join(vec_dir, 'users_vectorized.json'), 'w', encoding='utf-8') as f:
            json.dump(self.users_vec, f, ensure_ascii=False, indent=2)
        with open(os.path.join(vec_dir, 'courses_vectorized.json'), 'w', encoding='utf-8') as f:
            json.dump(self.courses_vec, f, ensure_ascii=False, indent=2)


class PCADialog(QDialog):
    def __init__(self, orig_data, vec_data, title):
        super().__init__()
        self.setWindowTitle(f"PCA Візуалізація: {title}")
        self.setGeometry(200, 200, 1000, 700)
        self.orig_data = orig_data
        self.vec_data = vec_data
        
        self.layout = QHBoxLayout(self)
        
        self.left_panel = QVBoxLayout()
        self.left_panel.addWidget(QLabel("Оберіть поле для розфарбовування:"))
        self.list_widget = QListWidget()
        
        sample = self.orig_data[0]
        for key, val in sample.items():
            if not isinstance(val, dict) and key != "Деталізована історія взаємодій":
                self.list_widget.addItem(key)
                
        self.list_widget.currentItemChanged.connect(self.update_plot)
        self.left_panel.addWidget(self.list_widget)
        self.layout.addLayout(self.left_panel, 1)
        
        self.figure, self.ax = plt.subplots(figsize=(7, 5))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas, 3)
        
        self.perform_pca()

    def perform_pca(self):
        matrix = []
        self.pca_orig_data = []
        
        for orig_row, vec_row in zip(self.orig_data, self.vec_data):
            if not vec_row or ("Статус" in vec_row and "Відхилено" in vec_row["Статус"]):
                continue
                
            num_row = []
            for k, v in vec_row.items():
                if k == 'User_ID' or k == 'Course_ID': continue
                if isinstance(v, list): 
                    num_row.extend(v)
                elif isinstance(v, (int, float)):
                    num_row.append(v)
            matrix.append(num_row)
            self.pca_orig_data.append(orig_row)
            
        pca = PCA(n_components=2)
        self.pca_result = pca.fit_transform(matrix)
        
        self.ax.clear()
        self.ax.scatter(self.pca_result[:, 0], self.pca_result[:, 1], c='gray', alpha=0.6)
        self.ax.set_title("PCA: Дані без розфарбовування")
        self.canvas.draw()

    def update_plot(self, current):
        if not current: return
        field = current.text()
        
        unique_vals = set()
        for row in self.pca_orig_data:
            val = row.get(field)
            if isinstance(val, list):
                for v in val: unique_vals.add(str(v))
            else:
                unique_vals.add(str(val))
                
        cmap = plt.get_cmap('tab20')
        color_map = {val: cmap(i / len(unique_vals))[:3] for i, val in enumerate(unique_vals)}
        
        point_colors = []
        for row in self.pca_orig_data:
            val = row.get(field)
            if isinstance(val, list):
                if not val:
                    point_colors.append((0.5, 0.5, 0.5)) 
                else:
                    colors = [color_map.get(str(v), (0,0,0)) for v in val]
                    avg_color = np.mean(colors, axis=0)
                    point_colors.append(tuple(avg_color))
            else:
                point_colors.append(color_map.get(str(val), (0.5, 0.5, 0.5)))

        self.ax.clear()
        self.ax.scatter(self.pca_result[:, 0], self.pca_result[:, 1], c=point_colors, alpha=0.8, edgecolors='k', linewidth=0.5)
        self.ax.set_title(f"PCA (Колір: {field})")
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[k], label=k) for k in list(unique_vals)[:10]]
        if len(unique_vals) > 10:
             legend_elements.append(Patch(facecolor='white', label='... (лише топ 10)'))
        self.ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        self.figure.tight_layout()
        self.canvas.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Векторизація Датасету (Контентна Фільтрація)")
        self.setGeometry(100, 100, 1300, 800)
        self.processor = DataVectorizer()
        self.initUI()
        
        if self.processor.nlp is None:
            QMessageBox.critical(self, "Помилка spaCy", 
                                 "Модель 'uk_core_news_sm' не знайдена!\n\n"
                                 "Будь ласка, відкрийте термінал і виконайте команду:\n"
                                 "python -m spacy download uk_core_news_sm\n\n"
                                 "Після цього перезапустіть програму.")
            sys.exit()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        top_panel = QHBoxLayout()
        self.btn_load = QPushButton("📥 Згенерувати нові вектори")
        self.btn_load.clicked.connect(self.run_pipeline)
        
        self.btn_open_existing = QPushButton("📂 Відкрити готові вектори")
        self.btn_open_existing.clicked.connect(self.load_existing_pipeline)
        
        top_panel.addWidget(self.btn_load)
        top_panel.addWidget(self.btn_open_existing)
        main_layout.addLayout(top_panel)

        self.tabs = QTabWidget()
        
        self.tab_users = QWidget()
        self.setup_data_tab(self.tab_users, "users")
        self.tabs.addTab(self.tab_users, "👤 Користувачі")
        
        self.tab_courses = QWidget()
        self.setup_data_tab(self.tab_courses, "courses")
        self.tabs.addTab(self.tab_courses, "📚 Курси")
        
        main_layout.addWidget(self.tabs)

    def setup_data_tab(self, tab, data_type):
        layout = QVBoxLayout(tab)
        
        btn_pca = QPushButton("📊 Відкрити PCA Візуалізацію")
        btn_pca.clicked.connect(lambda: self.open_pca(data_type))
        layout.addWidget(btn_pca)
        
        splitter = QSplitter(Qt.Vertical)
        
        table = QTableWidget()
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.SingleSelection)
        table.itemSelectionChanged.connect(lambda: self.on_row_selected(data_type))
        splitter.addWidget(table)
        
        bottom_spliter = QSplitter(Qt.Horizontal)
        
        orig_widget = QWidget()
        orig_layout = QVBoxLayout(orig_widget)
        orig_layout.addWidget(QLabel("📦 ОРИГІНАЛЬНІ ДАНІ (JSON):"))
        text_orig = QTextEdit()
        text_orig.setReadOnly(True)
        text_orig.setStyleSheet("background: #2b2b2b; color: #a9b7c6; font-family: Consolas;")
        orig_layout.addWidget(text_orig)
        bottom_spliter.addWidget(orig_widget)
        
        vec_widget = QWidget()
        vec_layout = QVBoxLayout(vec_widget)
        lbl_vec = QLabel("🔢 ВЕКТОРИЗОВАНІ ДАНІ (Поля не обрано):")
        lbl_vec.setWordWrap(True)
        vec_layout.addWidget(lbl_vec)
        text_vec = QTextEdit()
        text_vec.setReadOnly(True)
        text_vec.setStyleSheet("background: #2b2b2b; color: #6a8759; font-family: Consolas;")
        vec_layout.addWidget(text_vec)
        bottom_spliter.addWidget(vec_widget)
        
        splitter.addWidget(bottom_spliter)
        layout.addWidget(splitter)
        
        if data_type == "users":
            self.tbl_users = table
            self.txt_u_orig = text_orig
            self.txt_u_vec = text_vec
            self.lbl_u_vec = lbl_vec
        else:
            self.tbl_courses = table
            self.txt_c_orig = text_orig
            self.txt_c_vec = text_vec
            self.lbl_c_vec = lbl_vec

    def run_pipeline(self):
        self.btn_load.setText("⏳ Обробка... Зачекайте (Лематизація)")
        self.btn_load.setEnabled(False)
        QApplication.processEvents()
        
        if not self.processor.load_data():
            QMessageBox.critical(self, "Помилка", f"Оригінальні файли не знайдено у:\n{BASE_DIR}")
            self.btn_load.setText("📥 Згенерувати нові вектори")
            self.btn_load.setEnabled(True)
            return

        print("\nПочаток генерації...")
        self.processor.vectorize_users()
        self.processor.vectorize_courses()
        self.processor.save_vectorized()
        
        self.update_gui_after_load()
        self.btn_load.setText("✅ Дані оброблено та збережено")
        QMessageBox.information(self, "Готово", "Генерацію завершено! Дані збережено у папці vectorized.")

    def load_existing_pipeline(self):
        self.btn_open_existing.setText("⏳ Завантаження...")
        self.btn_open_existing.setEnabled(False)
        QApplication.processEvents()
        
        if not self.processor.load_data() or not self.processor.load_vectorized_data():
            QMessageBox.critical(self, "Помилка", "Не знайдено оригінальних або векторизованих файлів.")
            self.btn_open_existing.setText("📂 Відкрити готові вектори")
            self.btn_open_existing.setEnabled(True)
            return
            
        print("\n✅ Успішно завантажено існуючі вектори.")
        self.update_gui_after_load()
        self.btn_open_existing.setText("✅ Файли завантажено")
        
    def update_gui_after_load(self):
        self.populate_table("users")
        self.populate_table("courses")
        
        self.lbl_u_vec.setText(f"🔢 ВЕКТОРИЗОВАНІ ПОЛЯ (К-ть {len(self.processor.user_vec_fields)}):\n{', '.join(self.processor.user_vec_fields)}")
        self.lbl_c_vec.setText(f"🔢 ВЕКТОРИЗОВАНІ ПОЛЯ (К-ть {len(self.processor.course_vec_fields)}):\n{', '.join(self.processor.course_vec_fields)}")
        self.btn_load.setEnabled(True)
        self.btn_open_existing.setEnabled(True)

    def populate_table(self, data_type):
        table = self.tbl_users if data_type == "users" else self.tbl_courses
        data = self.processor.users_orig if data_type == "users" else self.processor.courses_orig
        
        if not data: return
        
        columns = list(data[0].keys())
        columns = [c for c in columns if c != "Деталізована історія взаємодій"]
        
        table.setRowCount(len(data))
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)
        
        for r_idx, row in enumerate(data):
            for c_idx, col in enumerate(columns):
                val = row.get(col, "")
                if isinstance(val, (list, dict)):
                    val = str(val)[:50] + "..." 
                table.setItem(r_idx, c_idx, QTableWidgetItem(str(val)))
                
        table.resizeColumnsToContents()

    def on_row_selected(self, data_type):
        if data_type == "users":
            table = self.tbl_users
            orig_data = self.processor.users_orig
            vec_data = self.processor.users_vec
            txt_orig = self.txt_u_orig
            txt_vec = self.txt_u_vec
        else:
            table = self.tbl_courses
            orig_data = self.processor.courses_orig
            vec_data = self.processor.courses_vec
            txt_orig = self.txt_c_orig
            txt_vec = self.txt_c_vec
            
        selected_rows = table.selectionModel().selectedRows()
        if not selected_rows: return
        idx = selected_rows[0].row()
        
        txt_orig.setText(json.dumps(orig_data[idx], ensure_ascii=False, indent=4))
        
        vec_dict = vec_data[idx]
        if vec_dict is None:
            txt_vec.setText("{\n  \"Статус\": \"Об'єкт відхилено через перевищення ліміту китайських символів\"\n}")
            return
            
        if "TF-IDF_vector" in vec_dict:
            display_dict = vec_dict.copy()
            v_len = len(display_dict["TF-IDF_vector"])
            preview = display_dict["TF-IDF_vector"][:5]
            display_dict["TF-IDF_vector"] = f"[ {', '.join(map(lambda x: f'{x:.3f}', preview))} ... та ще {v_len - 5} значень ]"
            txt_vec.setText(json.dumps(display_dict, ensure_ascii=False, indent=4))
        else:
            txt_vec.setText(json.dumps(vec_dict, ensure_ascii=False, indent=4))

    def open_pca(self, data_type):
        if not self.processor.users_orig:
            QMessageBox.warning(self, "Увага", "Спочатку завантажте дані!")
            return
            
        if data_type == "users":
            dialog = PCADialog(self.processor.users_orig, self.processor.users_vec, "Користувачі")
        else:
            dialog = PCADialog(self.processor.courses_orig, self.processor.courses_vec, "Курси")
        dialog.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())