# Note: run sekali di awal jika belum ada database.db

import sqlite3
import os

conn = sqlite3.connect(os.path.join('db', 'database.db'))

with open(os.path.join('db', 'schema.sql')) as f:
    conn.executescript(f.read())

cur = conn.cursor()

cur.execute('''
    INSERT INTO tbcmodels(path, description) VALUES
    ("svm_model_luas_singleUnet.pkl", "SVM Luas SU"),
    ("rf_model_luas_singleUnet.pkl", "Random Forest Luas SU"),
    ("svm_model_luas_posisi_singleUnet.pkl", "SVM Luas Posisi SU"),
    ("rf_model_luas_posisi_singleUnet.pkl", "Random Forest Luas Posisi SU"),
    ("svm_model_luas_doubleUnet.pkl", "SVM Luas DU"),
    ("rf_model_luas_doubleUnet.pkl", "Random Forest Luas DU"),
    ("svm_model_luas_posisi_doubleUnet.pkl", "SVM Luas Posisi DU"),
    ("rf_model_luas_posisi_doubleUnet.pkl", "Random Forest Luas Posisi DU")
''')

conn.commit()
conn.close()