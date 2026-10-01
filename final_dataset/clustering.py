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
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import cdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QTabWidget, QTableWidget, QTableWidgetItem, 
                             QTextEdit, QDialog, QListWidget, QSplitter, QMessageBox, QComboBox, QSpinBox)
from PyQt5.QtCore import Qt

import spacy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# ЯДРО ОБРОБКИ ДАНИХ (DATA PIPELINE)
# ==========================================
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
                raw_users = json.load(f)
            with open(courses_path, "r", encoding="utf-8") as f:
                raw_courses = json.load(f)

            # === АВТОМАТИЧНЕ ВИДАЛЕННЯ ДУБЛІКАТІВ КОРИСТУВАЧІВ ===
            seen_u_ids = set()
            self.users_orig = []
            for u in raw_users:
                uid = u.get("User_ID")
                if uid not in seen_u_ids:
                    if uid is not None:
                        seen_u_ids.add(uid)
                    self.users_orig.append(u)

            # === АВТОМАТИЧНЕ ВИДАЛЕННЯ ДУБЛІКАТІВ КУРСІВ ===
            seen_c_ids = set()
            self.courses_orig = []
            for c in raw_courses:
                cid = c.get("Course_ID")
                if cid not in seen_c_ids:
                    if cid is not None:
                        seen_c_ids.add(cid)
                    self.courses_orig.append(c)

            print(f"Завантажено: {len(self.users_orig)} унікальних користувачів (з {len(raw_users)}).")
            print(f"Завантажено: {len(self.courses_orig)} унікальних курсів (з {len(raw_courses)}).")
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
                raw_u_vec = json.load(f)
            with open(c_path, "r", encoding="utf-8") as f:
                raw_c_vec = json.load(f)

            # === ВИДАЛЕННЯ ДУБЛІКАТІВ У ГОТОВИХ ВЕКТОРАХ ===
            seen_u_ids = set()
            self.users_vec = []
            for u in raw_u_vec:
                uid = u.get("User_ID")
                if uid not in seen_u_ids:
                    if uid is not None:
                        seen_u_ids.add(uid)
                    self.users_vec.append(u)

            seen_c_ids = set()
            self.courses_vec = []
            for c in raw_c_vec:
                cid = c.get("Course_ID") if isinstance(c, dict) else None
                if cid not in seen_c_ids:
                    if cid is not None:
                        seen_c_ids.add(cid)
                    self.courses_vec.append(c)
                
            if self.users_vec:
                self.user_vec_fields = list(self.users_vec[0].keys())
            for c in self.courses_vec:
                if isinstance(c, dict) and "Статус" not in c:
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

    def vectorize_courses(self):
        df = pd.DataFrame(self.courses_orig)
        
        def is_too_chinese(title, desc):
            text = str(title).strip() + " " + str(desc).strip()
            if len(text.replace(" ", "")) == 0: return False
            ch_chars = re.findall(r'[\u4e00-\u9fff]', text)
            return (len(ch_chars) / len(text.replace(" ", ""))) > 0.5

        df['is_valid'] = df.apply(lambda row: not is_too_chinese(row.get('Назва курсу'), row.get('Текстовий опис')), axis=1)
        valid_df = df[df['is_valid']].copy()
        
        valid_vec = pd.DataFrame(index=valid_df.index)
        valid_vec['Course_ID'] = valid_df['Course_ID']
        
        def clean_text(row):
            title = str(row.get('Назва курсу', '')).strip()
            desc = str(row.get('Текстовий опис', '')).strip()
            text = (title + " " + desc).lower()
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
        
        final_vect = TfidfVectorizer(stop_words=drop_words, min_df=2, max_features=2500)
        tfidf_matrix = final_vect.fit_transform(texts)
        tfidf_dense = tfidf_matrix.toarray()
        
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


# ==========================================
# ВІКНО КЛАСТЕРИЗАЦІЇ ТА МЕТРИК
# ==========================================
class ClusteringDialog(QDialog):
    def __init__(self, processor, entity_type):
        super().__init__()
        self.processor = processor
        self.entity_type = entity_type
        self.last_clusters = None
        self.setWindowTitle(f"Ієрархічна кластеризація: {'Користувачі' if entity_type=='users' else 'Курси'}")
        self.setGeometry(150, 150, 1200, 800)
        self.initUI()

    def initUI(self):
        layout = QHBoxLayout(self)
        
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("<b>Оберіть метрику схожості (відстані):</b>"))
        
        self.metric_combo = QComboBox()
        if self.entity_type == "users":
            self.metric_combo.addItems([
                "1. Евклідова норма (за усім числовим вектором)",
                "2. Косинусна норма (1 - d за усім числовим вектором)",
                "3. Зважене порівняння атрибутів (Інтегральна метрика S)",
                "4. Крос-косинусна відстань (Пройдені <-> Бажане)",
                "5. Манхеттенська відстань (L1-норма)"
            ])
        else:
            self.metric_combo.addItems([
                "1. Косинусна відстань для текстових описів (1 - d)",
                "2. Евклідова відстань (за вектором)",
                "3. Зважена комбінована відстань (теги + числові: 0.7 / 0.3)",
                "4. Відстань за історією взаємодій з користувачами (Interaction)",
                "5. Зважена комбінована відстань (текст + теги + числові: 0.5 / 0.4 / 0.1)"
            ])
        left_layout.addWidget(self.metric_combo)

        left_layout.addWidget(QLabel("<b>Метод зв'язку (Linkage Method):</b>"))
        self.linkage_combo = QComboBox()
        self.linkage_combo.addItems([
            "ward",
            "average",
            "complete",
            "single"
        ])
        self.linkage_combo.setCurrentIndex(0)  # ward за замовчуванням
        left_layout.addWidget(self.linkage_combo)
        
        left_layout.addWidget(QLabel("<b>Кількість кластерів (k):</b>"))
        self.k_spinner = QSpinBox()
        self.k_spinner.setRange(2, 30)
        self.k_spinner.setValue(3)
        left_layout.addWidget(self.k_spinner)
        
        run_btn = QPushButton("🚀 Виконати ієрархічну кластеризацію")
        run_btn.setStyleSheet("background-color: #2b5797; color: white; font-weight: bold; padding: 12px;")
        run_btn.clicked.connect(self.run_clustering)
        left_layout.addWidget(run_btn)
        
        self.btn_pca_plot = QPushButton("📊 Показати графік кластерів на PCA")
        self.btn_pca_plot.setStyleSheet("background-color: #00838f; color: white; font-weight: bold; padding: 10px;")
        self.btn_pca_plot.setEnabled(False)
        self.btn_pca_plot.clicked.connect(self.show_pca_clusters)
        left_layout.addWidget(self.btn_pca_plot)
        
        left_layout.addStretch()
        layout.addLayout(left_layout, 1)
        
        right_layout = QVBoxLayout()
        self.lbl_result = QLabel("<b>Результати кластеризації:</b>")
        right_layout.addWidget(self.lbl_result)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("background: #1e1e1e; color: #dcdcdc; font-family: Consolas; font-size: 13px;")
        right_layout.addWidget(self.result_text)
        
        layout.addLayout(right_layout, 2)

    def _sanitize_matrix(self, d_matrix):
        """Гарантує відсутність від'ємних чисел, симетрію та нулі на діагоналі"""
        d_matrix = np.clip(d_matrix, 0.0, 1.0)
        d_matrix = (d_matrix + d_matrix.T) / 2.0
        np.fill_diagonal(d_matrix, 0.0)
        return d_matrix
        
    def _get_favorite_theme(self, user_data):
        """Отримує улюблені теми з даних JSON користувача"""
        themes = user_data.get("Улюблені теми")
        
        # Перевіряємо альтернативний ключ, якщо перший порожній
        if not themes:
            themes = user_data.get("Улюблена тема")
            
        if isinstance(themes, list) and len(themes) > 0:
            return ", ".join(themes)
        elif isinstance(themes, str) and themes.strip():
            return themes
            
        return "Не вказано"

    def compute_user_distance_matrix(self, metric_idx):
        users_orig = self.processor.users_orig
        users_vec = self.processor.users_vec
        n = len(users_orig)
        
        if metric_idx == 0:
            mat = np.array([[v for k, v in u.items() if k != 'User_ID' and isinstance(v, (int, float))] for u in users_vec], dtype=float)
            dists = euclidean_distances(mat)
            max_d = np.max(dists)
            return self._sanitize_matrix(dists / max_d if max_d > 0 else dists)

        elif metric_idx == 1:
            mat = np.array([[v for k, v in u.items() if k != 'User_ID' and isinstance(v, (int, float))] for u in users_vec], dtype=float)
            cos_sim = cosine_similarity(mat)
            return self._sanitize_matrix(1.0 - cos_sim)

        elif metric_idx == 2:
            passed_cols = [k for k in users_vec[0].keys() if 'Пройдено_тег_' in k]
            wish_cols = [k for k in users_vec[0].keys() if 'Бажане_тег_' in k]
            
            passed_mat = np.array([[u.get(col, 0.0) for col in passed_cols] for u in users_vec], dtype=float) if passed_cols else np.zeros((n, 1))
            wish_mat = np.array([[u.get(col, 0.0) for col in wish_cols] for u in users_vec], dtype=float) if wish_cols else np.zeros((n, 1))
            num_mat = np.array([[u.get('Вік_norm', 0), u.get('Час_навчання_norm', 0), u.get('Витрати_norm', 0)] for u in users_vec], dtype=float)
            edu_arr = np.array([u.get('Освіта_norm', 0) for u in users_vec], dtype=float)
            
            mlb_f = MultiLabelBinarizer()
            formats_bin = mlb_f.fit_transform([u.get("Бажані формати навчання", []) for u in users_orig])
            if formats_bin.shape[1] > 0:
                inter_f = formats_bin @ formats_bin.T
                sums_f = formats_bin.sum(axis=1)
                union_f = sums_f[:, None] + sums_f[None, :] - inter_f
                d_formats = 1.0 - np.divide(inter_f, union_f, out=np.zeros_like(inter_f, dtype=float), where=union_f!=0)
            else:
                d_formats = np.zeros((n, n))
                
            goal_map = {"Хобі": 0, "Професійний розвиток": 1, "Академічні цілі": 2}
            goal_indices = np.array([goal_map.get(u.get("Мета навчання", "Хобі"), 0) for u in users_orig])
            goal_penalty = np.array([[0.0, 1.0, 0.75], [1.0, 0.0, 0.25], [0.75, 0.25, 0.0]])
            d_goal = goal_penalty[goal_indices[:, None], goal_indices[None, :]]
            
            d_passed = 1.0 - cosine_similarity(passed_mat)
            d_wish = 1.0 - cosine_similarity(wish_mat)
            d_num = euclidean_distances(num_mat) / np.sqrt(num_mat.shape[1])
            d_edu = np.abs(edu_arr[:, None] - edu_arr[None, :])
            
            dist_matrix = (0.40 * d_passed + 0.12 * d_formats + 0.12 * d_num + 
                           0.12 * d_wish + 0.12 * d_edu + 0.12 * d_goal)
            return self._sanitize_matrix(dist_matrix)

        elif metric_idx == 3:
            passed_cols = [k for k in users_vec[0].keys() if 'Пройдено_тег_' in k]
            wish_cols = [k for k in users_vec[0].keys() if 'Бажане_тег_' in k]
            
            passed_mat = np.array([[u.get(col, 0.0) for col in passed_cols] for u in users_vec], dtype=float) if passed_cols else np.zeros((n, 1))
            wish_mat = np.array([[u.get(col, 0.0) for col in wish_cols] for u in users_vec], dtype=float) if wish_cols else np.zeros((n, 1))
            
            C_PW = cosine_similarity(passed_mat, wish_mat)
            d_cross = 1.0 - (C_PW + C_PW.T) / 2.0
            return self._sanitize_matrix(d_cross)

        elif metric_idx == 4:
            mat = np.array([[v for k, v in u.items() if k != 'User_ID' and isinstance(v, (int, float))] for u in users_vec], dtype=float)
            dists = cdist(mat, mat, metric='cityblock')
            return self._sanitize_matrix(dists / mat.shape[1])

        return np.zeros((n, n))

    def compute_course_distance_matrix(self, metric_idx):
        courses_orig = self.processor.courses_orig
        courses_vec = self.processor.courses_vec
        
        valid_indices = [i for i, c in enumerate(courses_vec) if c and "Статус" not in c]
        n = len(valid_indices)
        if n == 0: return np.zeros((0, 0))

        if metric_idx == 0:
            texts = [courses_vec[i].get('TF-IDF_vector', [0]) for i in valid_indices]
            sim = cosine_similarity(texts)
            return self._sanitize_matrix(1.0 - sim)

        elif metric_idx == 1:
            mat = np.array([[v for k, v in courses_vec[i].items() if k not in ['Course_ID', 'TF-IDF_vector'] and isinstance(v, (int, float))] for i in valid_indices], dtype=float)
            dists = euclidean_distances(mat)
            max_d = np.max(dists)
            return self._sanitize_matrix(dists / max_d if max_d > 0 else dists)

        elif metric_idx == 2:
            valid_orig = [courses_orig[i] for i in valid_indices]
            mlb_t = MultiLabelBinarizer()
            tags_bin = mlb_t.fit_transform([c.get("Теги", []) for c in valid_orig])
            if tags_bin.shape[1] > 0:
                inter_t = tags_bin @ tags_bin.T
                sums_t = tags_bin.sum(axis=1)
                union_t = sums_t[:, None] + sums_t[None, :] - inter_t
                d_tags = 1.0 - np.divide(inter_t, union_t, out=np.zeros_like(inter_t, dtype=float), where=union_t!=0)
            else:
                d_tags = np.zeros((n, n))
                
            num_c = np.array([[courses_vec[i].get('Ціна_norm', 0), courses_vec[i].get('Тривалість_norm', 0)] for i in valid_indices], dtype=float)
            d_num = euclidean_distances(num_c) / np.sqrt(num_c.shape[1]) if num_c.shape[1] > 0 else np.zeros((n, n))
            
            return self._sanitize_matrix(0.7 * d_tags + 0.3 * d_num)

        elif metric_idx == 3:
            valid_orig = [courses_orig[i] for i in valid_indices]
            all_users_hist = [set(c.get("Деталізована історія", {}).get("Куплені курси", [orig_idx])) for orig_idx, c in zip(valid_indices, valid_orig)]
            unique_users = sorted(list(set(u for hist in all_users_hist for u in hist)))
            
            if len(unique_users) > 0:
                user_to_idx = {u: idx for idx, u in enumerate(unique_users)}
                hist_bin = np.zeros((n, len(unique_users)), dtype=int)
                for idx, hist in enumerate(all_users_hist):
                    for u in hist:
                        if u in user_to_idx:
                            hist_bin[idx, user_to_idx[u]] = 1
                inter_u = hist_bin @ hist_bin.T
                sums_u = hist_bin.sum(axis=1)
                union_u = sums_u[:, None] + sums_u[None, :] - inter_u
                jacc_u = np.divide(inter_u, union_u, out=np.zeros_like(inter_u, dtype=float), where=union_u!=0)
                d_matrix = 1.0 - jacc_u
            else:
                d_matrix = np.random.rand(n, n) * 0.1
                d_matrix = (d_matrix + d_matrix.T) / 2
            
            return self._sanitize_matrix(d_matrix)

        elif metric_idx == 4:
            texts = [courses_vec[i].get('TF-IDF_vector', [0]) for i in valid_indices]
            d_text = 1.0 - cosine_similarity(texts)
            
            valid_orig = [courses_orig[i] for i in valid_indices]
            mlb_t = MultiLabelBinarizer()
            tags_bin = mlb_t.fit_transform([c.get("Теги", []) for c in valid_orig])
            if tags_bin.shape[1] > 0:
                inter_t = tags_bin @ tags_bin.T
                sums_t = tags_bin.sum(axis=1)
                union_t = sums_t[:, None] + sums_t[None, :] - inter_t
                d_tags = 1.0 - np.divide(inter_t, union_t, out=np.zeros_like(inter_t, dtype=float), where=union_t!=0)
            else:
                d_tags = np.zeros((n, n))
                
            num_c = np.array([[courses_vec[i].get('Ціна_norm', 0), courses_vec[i].get('Тривалість_norm', 0)] for i in valid_indices], dtype=float)
            d_num = euclidean_distances(num_c) / np.sqrt(num_c.shape[1]) if num_c.shape[1] > 0 else np.zeros((n, n))
            
            return self._sanitize_matrix(0.5 * d_text + 0.4 * d_tags + 0.1 * d_num)

        return np.zeros((n, n))

    def run_clustering(self):
        metric_idx = self.metric_combo.currentIndex()
        k = self.k_spinner.value()
        linkage_method = self.linkage_combo.currentText()
        
        valid_indices = []
        if self.entity_type == "users":
            dist_matrix = self.compute_user_distance_matrix(metric_idx)
            names = [f"Користувач #{u.get('User_ID', i)} [idx {i}]" for i, u in enumerate(self.processor.users_orig)]
        else:
            dist_matrix = self.compute_course_distance_matrix(metric_idx)
            courses_orig = self.processor.courses_orig
            courses_vec = self.processor.courses_vec
            valid_indices = [i for i, c in enumerate(courses_vec) if c and "Статус" not in c]
            names = [f"Курс: {courses_orig[i].get('Назва курсу', f'ID {i}')} [idx {i}]" for i in valid_indices]

        n = len(dist_matrix)
        if n < k:
            QMessageBox.warning(self, "Помилка", f"Кількість об'єктів ({n}) менша за кількість кластерів k ({k})!")
            return
            
        if n != len(names):
            QMessageBox.critical(self, "Розсинхронізація", 
                                 f"Кількість об'єктів у матриці ({n}) не збігається з оригіналами ({len(names)}).\n\n"
                                 f"Будь ласка, закрийте це вікно і натисніть синю кнопку 'Згенерувати нові вектори'!")
            return

        try:
            condensed_dist = squareform(dist_matrix, checks=False)
            Z = linkage(condensed_dist, method=linkage_method)
            clusters = fcluster(Z, t=k, criterion='maxclust')
            self.last_clusters = clusters
            self.btn_pca_plot.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Помилка кластеризації", f"Метод {linkage_method} викликав помилку:\n{str(e)}")
            return

        # --- РОЗРАХУНОК МЕТРИК ЯКОСТІ (WSS, BSS, Rand Index) ---
        
        # 1. Формування матриці ознак X для WSS та BSS
        X = []
        for i in range(len(valid_indices)) if self.entity_type == 'courses' else range(n):
            if self.entity_type == "users":
                vec_dict = self.processor.users_vec[i]
            else:
                vec_dict = self.processor.courses_vec[valid_indices[i]]
            
            num_row = []
            for key, val in vec_dict.items():
                if key not in ['User_ID', 'Course_ID']:
                    if isinstance(val, list): num_row.extend(val)
                    elif isinstance(val, (int, float)): num_row.append(val)
            X.append(num_row)
        X = np.array(X, dtype=float)
        
        # Обчислення WSS (Компактність) та BSS (Відокремленість)
        global_centroid = np.mean(X, axis=0)
        WSS = 0.0
        BSS = 0.0
        
        for cluster_id in range(1, k + 1):
            mask = (clusters == cluster_id)
            points_in_cluster = X[mask]
            if len(points_in_cluster) > 0:
                cluster_centroid = np.mean(points_in_cluster, axis=0)
                WSS += np.sum((points_in_cluster - cluster_centroid) ** 2)
                BSS += len(points_in_cluster) * np.sum((cluster_centroid - global_centroid) ** 2)

        # 2. Розрахунок Індексу Rand (TP, FP, FN, TN)
        true_labels = []
        for i in range(n):
            if self.entity_type == "users":
                u = self.processor.users_orig[i]
                true_labels.append(self._get_favorite_theme(u))
            else:
                c = self.processor.courses_orig[valid_indices[i]]
                tags = c.get('Теги', [])
                true_labels.append(", ".join(tags) if tags else "Немає тегів")
                
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        true_labels_int = le.fit_transform(true_labels)
        
        # Матриці попарної рівності
        pred_pairs = (clusters[:, None] == clusters[None, :])
        true_pairs = (true_labels_int[:, None] == true_labels_int[None, :])
        
        # Беремо лише верхній трикутник матриці (без діагоналі) для унікальних пар
        iu = np.triu_indices(n, k=1)
        pred_pairs_flat = pred_pairs[iu]
        true_pairs_flat = true_pairs[iu]
        
        TP = np.sum(pred_pairs_flat & true_pairs_flat)
        TN = np.sum((~pred_pairs_flat) & (~true_pairs_flat))
        FP = np.sum(pred_pairs_flat & (~true_pairs_flat))
        FN = np.sum((~pred_pairs_flat) & true_pairs_flat)
        
        rand_index = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

        # --- ФОРМУВАННЯ ВИВЕДЕННЯ ---
        
        self.lbl_result.setText(f"<b>Результати кластеризації (Метод: {linkage_method.upper()}):</b>")
        
        output = f"=== ІЄРАРХІЧНА КЛАСТЕРИЗАЦІЯ ({linkage_method.upper()} Linkage, k={k}) ===\n"
        output += f"Сутність: {'Користувачі' if self.entity_type=='users' else 'Курси'}\n"
        output += f"Обрана метрика: {self.metric_combo.currentText()}\n\n"

        output += "📊 ОЦІНКА ЯКОСТІ КЛАСТЕРИЗАЦІЇ:\n"
        output += "Внутрішні метрики:\n"
        output += f" • Компактність (WSS): {WSS:.4f}\n"
        output += f" • Відокремленість (BSS): {BSS:.4f}\n\n"
        
        output += "Зовнішні метрики (Індекс Rand):\n"
        output += f" • Базовий критерій (Клас): {'Улюблена тема' if self.entity_type=='users' else 'Теги курсу'}\n"
        output += f" • TP (один кластер, один клас): {TP}\n"
        output += f" • FP (один кластер, різні класи): {FP}\n"
        output += f" • FN (різні кластери, один клас): {FN}\n"
        output += f" • TN (різні кластери, різні класи): {TN}\n"
        output += f" • Індекс Rand: {rand_index:.4f}\n\n"

        output += "📋 СПИСОК КЛАСТЕРІВ:\n"
        for cluster_id in range(1, k + 1):
            count = np.sum(clusters == cluster_id)
            output += f" • Кластер #{cluster_id}: {count} елементів\n"
        output += "\n" + "="*50 + "\n\n"

        for cluster_id in range(1, k + 1):
            cl_indices = np.where(clusters == cluster_id)[0]
            output += f"--------------------------------------------------\n"
            output += f"📁 КЛАСТЕР #{cluster_id} (Елементів у кластері: {len(cl_indices)}):\n"
            
            if len(cl_indices) == 0:
                output += "Кластер порожній.\n\n"
                continue

            cl_names = [names[idx] for idx in cl_indices]
            output += f"Об'єкти: {', '.join(cl_names[:10])}{' ... та інші' if len(cl_names) > 10 else ''}\n"

            pairs = []
            for i in range(len(cl_indices)):
                for j in range(i + 1, len(cl_indices)):
                    idx1 = cl_indices[i]
                    idx2 = cl_indices[j]
                    d_val = dist_matrix[idx1, idx2]
                    pairs.append((idx1, idx2, d_val))

            pairs.sort(key=lambda x: x[2])
            top10 = pairs[:10]

            output += f"\nТОП-10 найсхожіших пар у Кластері #{cluster_id}:\n"
            if not top10:
                output += "  (У кластері замало елементів для формування пар)\n"
                
            for rank, (idx1, idx2, val) in enumerate(top10, 1):
                if self.entity_type == "users":
                    u1 = self.processor.users_orig[idx1]
                    u2 = self.processor.users_orig[idx2]
                    
                    f_name1 = f"👤 Користувач #{u1.get('User_ID', idx1)}"
                    f_name2 = f"👤 Користувач #{u2.get('User_ID', idx2)}"
                    
                    fav_theme1 = self._get_favorite_theme(u1)
                    fav_theme2 = self._get_favorite_theme(u2)
                    
                    det1 = f"Вік: {u1.get('Вік')} | Освіта: {u1.get('Рівень освіти')} | Теми: {fav_theme1} | Мета: {u1.get('Мета навчання')}"
                    det2 = f"Вік: {u2.get('Вік')} | Освіта: {u2.get('Рівень освіти')} | Теми: {fav_theme2} | Мета: {u2.get('Мета навчання')}"
                else:
                    c1 = self.processor.courses_orig[valid_indices[idx1]]
                    c2 = self.processor.courses_orig[valid_indices[idx2]]
                    
                    # ПОВНА НАЗВА БЕЗ ОБРІЗКИ
                    title1 = str(c1.get('Назва курсу', f'ID {idx1}'))
                    title2 = str(c2.get('Назва курсу', f'ID {idx2}'))
                    f_name1 = f"📚 {title1}"
                    f_name2 = f"📚 {title2}"
                    
                    det1 = f"Ціна: {c1.get('Ціна')} | Складність: {c1.get('Складність')} | Трив: {c1.get('Тривалість')}хв | Теги: {c1.get('Теги')}"
                    det2 = f"Ціна: {c2.get('Ціна')} | Складність: {c2.get('Складність')} | Трив: {c2.get('Тривалість')}хв | Теги: {c2.get('Теги')}"

                # Деревоподібний візуальний вивід
                output += f"{rank}. Відстань: {val:.4f}\n"
                output += f"   ├── {f_name1}\n"
                output += f"   │   ↳ {det1}\n"
                output += f"   └── {f_name2}\n"
                output += f"       ↳ {det2}\n\n"
            output += "\n"

        self.result_text.setPlainText(output)

    def show_pca_clusters(self):
        if self.last_clusters is None:
            return
        title = "Користувачі" if self.entity_type == "users" else "Курси"
        dialog = ClusterPCADialog(self.processor, self.entity_type, self.last_clusters, title)
        dialog.exec_()


# ==========================================
# ВІКНО PCA ВІЗУАЛІЗАЦІЇ КЛАСТЕРІВ
# ==========================================
class ClusterPCADialog(QDialog):
    def __init__(self, processor, entity_type, clusters, title):
        super().__init__()
        self.processor = processor
        self.entity_type = entity_type
        self.clusters = clusters
        self.setWindowTitle(f"PCA Карта Кластерів: {title}")
        self.setGeometry(200, 200, 950, 700)
        
        layout = QVBoxLayout(self)
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.plot_clusters()

    def plot_clusters(self):
        orig_data = self.processor.users_orig if self.entity_type == "users" else self.processor.courses_orig
        vec_data = self.processor.users_vec if self.entity_type == "users" else self.processor.courses_vec
        
        matrix = []
        valid_clusters = []
        for orig_row, vec_row, cl_id in zip(orig_data, vec_data, self.clusters):
            if not vec_row or ("Статус" in vec_row and "Відхилено" in vec_row["Статус"]):
                continue
            num_row = [v for k, v in vec_row.items() if k not in ['User_ID', 'Course_ID'] and (isinstance(v, (int, float)) or isinstance(v, list))]
            flat_row = []
            for item in num_row:
                if isinstance(item, list): flat_row.extend(item)
                else: flat_row.append(item)
            matrix.append(flat_row)
            valid_clusters.append(cl_id)
            
        matrix = np.array(matrix, dtype=float)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(matrix)
        valid_clusters = np.array(valid_clusters)
        
        self.ax.clear()
        unique_clusters = np.unique(valid_clusters)
        cmap = plt.get_cmap('tab10')
        
        for idx, cl_id in enumerate(unique_clusters):
            mask = (valid_clusters == cl_id)
            color = cmap(idx % 10)
            self.ax.scatter(coords[mask, 0], coords[mask, 1], color=color, label=f"Кластер #{cl_id}", alpha=0.8, edgecolors='k', linewidth=0.5)
            
        self.ax.set_title("PCA Карта кластерів (Колір відповідає кластеру)")
        self.ax.set_xlabel("Головна компонента 1")
        self.ax.set_ylabel("Головна компонента 2")
        self.ax.legend(loc='upper right', fontsize='small')
        self.figure.tight_layout()
        self.canvas.draw()


# ==========================================
# ВІКНО PCA (ВІЗУАЛІЗАЦІЯ ДАНИХ)
# ==========================================
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
            num_row = [v for k, v in vec_row.items() if k not in ['User_ID', 'Course_ID'] and (isinstance(v, (int, float)) or isinstance(v, list))]
            flat_row = []
            for item in num_row:
                if isinstance(item, list): flat_row.extend(item)
                else: flat_row.append(item)
            matrix.append(flat_row)
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
                if not val: point_colors.append((0.5, 0.5, 0.5)) 
                else:
                    colors = [color_map.get(str(v), (0,0,0)) for v in val]
                    point_colors.append(tuple(np.mean(colors, axis=0)))
            else:
                point_colors.append(color_map.get(str(val), (0.5, 0.5, 0.5)))

        self.ax.clear()
        self.ax.scatter(self.pca_result[:, 0], self.pca_result[:, 1], c=point_colors, alpha=0.8, edgecolors='k', linewidth=0.5)
        self.ax.set_title(f"PCA (Колір: {field})")
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[k], label=k) for k in list(unique_vals)[:10]]
        if len(unique_vals) > 10: legend_elements.append(Patch(facecolor='white', label='... (лише топ 10)'))
        self.ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        self.figure.tight_layout()
        self.canvas.draw()


# ==========================================
# ГОЛОВНЕ ВІКНО (GUI)
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Векторизація та Ієрархічна Кластеризація (Контентна Фільтрація)")
        self.setGeometry(100, 100, 1350, 850)
        self.processor = DataVectorizer()
        self.initUI()
        
        if self.processor.nlp is None:
            QMessageBox.critical(self, "Помилка spaCy", "Виконайте у терміналі:\npython -m spacy download uk_core_news_sm")
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
        btns_layout = QHBoxLayout()
        
        btn_pca = QPushButton("📊 Відкрити PCA Візуалізацію")
        btn_pca.clicked.connect(lambda: self.open_pca(data_type))
        btns_layout.addWidget(btn_pca)
        
        btn_cluster = QPushButton("🔬 Ієрархічна кластеризація та ТОП-10 пар")
        btn_cluster.setStyleSheet("background-color: #2b5797; color: white; font-weight: bold;")
        btn_cluster.clicked.connect(lambda: self.open_clustering(data_type))
        btns_layout.addWidget(btn_cluster)
        
        layout.addLayout(btns_layout)
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
        self.btn_load.setText("⏳ Обробка... Зачекайте")
        self.btn_load.setEnabled(False)
        QApplication.processEvents()
        
        if not self.processor.load_data():
            QMessageBox.critical(self, "Помилка", f"Оригінальні файли не знайдено у:\n{BASE_DIR}")
            self.btn_load.setText("📥 Згенерувати нові вектори")
            self.btn_load.setEnabled(True)
            return

        self.processor.vectorize_users()
        self.processor.vectorize_courses()
        self.processor.save_vectorized()
        self.update_gui_after_load()
        self.btn_load.setText("✅ Дані оброблено та збережено")
        QMessageBox.information(self, "Готово", "Генерацію завершено!")

    def load_existing_pipeline(self):
        self.btn_open_existing.setText("⏳ Завантаження...")
        self.btn_open_existing.setEnabled(False)
        QApplication.processEvents()
        
        if not self.processor.load_data() or not self.processor.load_vectorized_data():
            QMessageBox.critical(self, "Помилка", "Не знайдено файлів у папці vectorized/.")
            self.btn_open_existing.setText("📂 Відкрити готові вектори")
            self.btn_open_existing.setEnabled(True)
            return
            
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
        
        columns = [c for c in data[0].keys() if c != "Деталізована історія взаємодій"]
        table.setRowCount(len(data))
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)
        
        for r_idx, row in enumerate(data):
            for c_idx, col in enumerate(columns):
                val = str(row.get(col, ""))
                if len(val) > 50: val = val[:50] + "..."
                table.setItem(r_idx, c_idx, QTableWidgetItem(val))
        table.resizeColumnsToContents()

    def on_row_selected(self, data_type):
        table = self.tbl_users if data_type == "users" else self.tbl_courses
        orig_data = self.processor.users_orig if data_type == "users" else self.processor.courses_orig
        vec_data = self.processor.users_vec if data_type == "users" else self.processor.courses_vec
        txt_orig = self.txt_u_orig if data_type == "users" else self.txt_c_orig
        txt_vec = self.txt_u_vec if data_type == "users" else self.txt_c_vec
            
        selected_rows = table.selectionModel().selectedRows()
        if not selected_rows: return
        idx = selected_rows[0].row()
        
        txt_orig.setText(json.dumps(orig_data[idx], ensure_ascii=False, indent=4))
        if idx >= len(vec_data) or vec_data[idx] is None:
            txt_vec.setText("{\n  \"Статус\": \"Об'єкт відхилено через перевищення ліміту китайських символів\"\n}")
            return
            
        vec_dict = vec_data[idx]
        if "TF-IDF_vector" in vec_dict:
            display_dict = vec_dict.copy()
            v_len = len(display_dict["TF-IDF_vector"])
            preview = display_dict["TF-IDF_vector"][:5]
            display_dict["TF-IDF_vector"] = f"[ {', '.join(map(lambda x: f'{x:.3f}', preview))} ... та ще {v_len - 5} значень ]"
            txt_vec.setText(json.dumps(display_dict, ensure_ascii=False, indent=4))
        else:
            txt_vec.setText(json.dumps(vec_dict, ensure_ascii=False, indent=4))

    def open_pca(self, data_type):
        if not self.processor.users_orig or not self.processor.users_vec:
            QMessageBox.warning(self, "Увага", "Спочатку завантажте або згенеруйте дані!")
            return
        dialog = PCADialog(self.processor.users_orig, self.processor.users_vec, "Користувачі" if data_type=="users" else "Курси")
        dialog.exec_()

    def open_clustering(self, data_type):
        if not self.processor.users_orig or not self.processor.users_vec:
            QMessageBox.warning(self, "Увага", "Спочатку завантажте або згенеруйте дані!")
            return
        dialog = ClusteringDialog(self.processor, data_type)
        dialog.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())