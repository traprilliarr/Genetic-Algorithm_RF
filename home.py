import streamlit as st
import os
import base64
from preprocess import preprocess_data, split_data
from random_forest import RandomForestClassifier
from optimasi import generations
from sklearn.metrics import accuracy_score

script_directory = os.path.abspath(os.path.dirname(__file__))

def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['Home', 'Predict'])

    if page == 'Home':
        home_page()
    elif page == 'Predict':
        predict_page()

def home_page():
    st.title('Optimasi Algoritma Random Forest Menggunakan Algoritma Genetika Dalam Klasifikasi Stroke')

    # Upload file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        data = preprocess_data(uploaded_file)
        st.subheader('Data Preview:')
        st.write(data)

        option = st.selectbox('Choose an option', ['RF (Random Forest)', 'RF+GA (Random Forest + Genetic Algorithm)'])

        if option == 'RF (Random Forest)':
            estimators = st.number_input('Estimators (Number of Trees)', min_value=1, value=100)
            if st.button('Submit'):
                X_train, X_test, y_train, y_test = split_data(data)
                clf = RandomForestClassifier(n_trees=estimators)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f'Accuracy: {accuracy}')

        elif option == 'RF+GA (Random Forest + Genetic Algorithm)':
            estimators = st.number_input('Estimators (Number of Trees)', min_value=1, value=100)
            crossover_rate = st.slider('Crossover Rate', min_value=0.0, max_value=1.0, value=0.1)
            mutation_rate = st.slider('Mutation Rate', min_value=0.0, max_value=1.0, value=0.1)
            population = st.number_input('Population', min_value=1, value=200)
            generations_input = st.number_input('Generations', min_value=1, value=4)
            if st.button('Submit'):
                X_train, X_test, y_train, y_test = split_data(data)
                best_chromo, best_score = generations(data, data, size=population, n_feat=X_train.shape[1], crossover_rate=crossover_rate, mutation_rate=mutation_rate, n_gen=generations_input, X_rf_train=X_train, X_rf_test=X_test, y_rf_train=y_train, y_rf_test=y_test)
                clf = RandomForestClassifier(n_trees=estimators)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f'Accuracy of RF+GA: {accuracy}')


def predict_page():
    st.title('Prediksi Pasien Stroke')

    # Form input
    st.subheader('Masukkan detail pasien:')
    gender = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan', 'Lainnya'])
    age = st.number_input('Usia', min_value=0, max_value=150, value=30)
    hypertension = st.selectbox('Hipertensi', ['Tidak', 'Ya'])
    heart_disease = st.selectbox('Penyakit Jantung', ['Tidak', 'Ya'])
    ever_married = st.selectbox('Pernah Menikah', ['Tidak', 'Ya'])
    work_type = st.selectbox('Jenis Pekerjaan', ['Anak-anak', 'Pekerja Pemerintah', 'Belum Pernah Bekerja', 'Swasta', 'Wiraswasta'])
    residence_type = st.selectbox('Tipe Tempat Tinggal', ['Pedesaan', 'Perkotaan'])
    avg_glucose_level = st.number_input('Rata-rata Kadar Glukosa', min_value=0.0, value=100.0)
    bmi = st.number_input('Indeks Massa Tubuh (BMI)', min_value=0.0, value=20.0)
    smoking_status = st.selectbox('Status Merokok', ['Pernah Merokok', 'Tidak Pernah Merokok', 'Merokok', 'Tidak Diketahui'])

    if st.button('Prediksi'):
        input_data = {
            'gender': gender,
            'age': age,
            'hypertension': 1 if hypertension == 'Ya' else 0,
            'heart_disease': 1 if heart_disease == 'Ya' else 0,
            'ever_married': 'Ya' if ever_married == 'Ya' else 'Tidak',
            'work_type': work_type,
            'Residence_type': residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status
        }
        prediction = 1  
        if prediction == 1:
            st.write('Pasien memiliki stroke')
        else:
            st.write('Pasien tidak memiliki stroke')


if __name__ == '__main__':
    main()
