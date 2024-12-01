import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import plotly.express as px
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE

# Judul Aplikasi
st.title("Analisis Faktor yang Mempengaruhi Kepuasan Penumpang Pesawat Terbang ")
st.divider()
# Fitur Upload Dataset
with st.expander("Upload File CSV"):
    uploaded_file = st.file_uploader("", type=["csv"])

if uploaded_file is not None:

    #validasi input
    if uploaded_file.name != "maskapai.csv":
            st.error("File tidak sesuai! Harap unggah dataset yang sesuai.")
    else:
        # read dtaset dan menyimpang ke variabel df
        df = pd.read_csv(uploaded_file)
        st.balloons()

        #deklarasi dataframe yang akan digunakan secara global
        data = df.dropna(axis=0)
        new_df = data.drop(['Unnamed: 0', 'id', 'Age', 'Gender', 'Customer Type', 'Type of Travel', 'Class', 'Flight Distance'], axis=1)
        dff = new_df.drop(['Departure/Arrival time convenient', 'Gate location', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'], axis=1)
        df_cleaned = dff.drop_duplicates()
        # Sidebar Menu
        with st.sidebar:
            st.sidebar.title("Menu Utama")
            main_menu = st.sidebar.selectbox("Pilih Menu", ["⏳ Preprocessing", "🔍 Analisis Data", "📊 Visualisasi Data"])

        st.divider()
        # Menu Preprocessing
    if main_menu == "⏳ Preprocessing":
        st.header("Preprocessing Dataset")
        st.subheader("Informasi Dataset")
        txt = st.text_area(
            "Summary",
            "Dataset ini berisi survei kepuasan penumpang maskapai penerbangan. "
            "Terdapat berbagai faktor yang memengaruhi kepuasan penumpang, "
            "oleh sebab itu melalui aplikasi ini, maskapai penerbangan dapat mengetahui "
            "faktor apa saja yang memiliki korelasi tinggi dengan kepuasan "
            "(atau ketidakpuasan) penumpang\n\n"
            "- Gender: Gender of the passengers (Female, Male)\n"
            "- Customer Type: The customer type (Loyal customer, disloyal customer)\n"
            "- Age: The actual age of the passengers\n"
            "- Type of Travel: Purpose of the flight of the passengers (Personal Travel, Business Travel)\n"
            "- Class: Travel class in the plane of the passengers (Business, Eco, Eco Plus)\n"
            "- Flight distance: The flight distance of this journey\n"
            "- Inflight wifi service: Satisfaction level of the inflight wifi service (0:Not Applicable;1-5)\n"
            "- Departure/Arrival time convenient: Satisfaction level of Departure/Arrival time convenient\n"
            "- Ease of Online booking: Satisfaction level of online booking\n"
            "- Gate location: Satisfaction level of Gate location\n"
            "- Food and drink: Satisfaction level of Food and drink\n"
            "- Online boarding: Satisfaction level of online boarding\n"
            "- Seat comfort: Satisfaction level of Seat comfort\n"
            "- Inflight entertainment: Satisfaction level of inflight entertainment\n"
            "- On-board service: Satisfaction level of On-board service\n"
            "- Leg room service: Satisfaction level of Leg room service\n"
            "- Baggage handling: Satisfaction level of baggage handling\n"
            "- Check-in service: Satisfaction level of Check-in service\n"
            "- Inflight service: Satisfaction level of inflight service\n"
            "- Cleanliness: Satisfaction level of Cleanliness\n"
            "- Departure Delay in Minutes: Minutes delayed when departure\n"
            "- Arrival Delay in Minutes: Minutes delayed when Arrival\n"
            "- Satisfaction: Airline satisfaction level(Satisfaction, neutral or dissatisfaction)",
        )

        with st.expander("Tampilkan Data"):
            st.write(df)
            st.write(f"Dataset ini terdiri dari {df.shape[0]} baris dan {df.shape[1]} kolom.")
            st.write(data.dtypes)

        with st.expander("Drop Null Value"):
            data = df.dropna(axis=0)
            st.write(df.isnull().sum())
            if st.button("Drop Null Value"):
                st.write(data)
                st.success("Berhasil menghapus nilai null!!")
                st.write(f"Dataset ini terdiri dari {data.shape[0]} baris dan {data.shape[1]} kolom.")
                st.write(data.isnull().sum())

        #feature selection
        new_df = data.drop(['Unnamed: 0', 'id', 'Age', 'Gender', 'Customer Type', 'Type of Travel', 'Class', 'Flight Distance'], axis=1)
        with st.expander("Feature Elimination"):
            st.text_area(
                "",
                "Kolom 'Unnamed: 0', 'id', 'Age', 'Gender', 'Customer Type', 'Type of Travel', 'Class', 'Flight Distance'" 
                "tidak termasuk atribut yang memengaruhi tingkat kepuasan penumpang, maka untuk meningkatkan" 
                "akurasi model, kolom tersebut dapat dihapus.",
                )
            if st.button("Drop Atribut"):
                st.write(new_df)
                st.success(f"Berhasil menghapus atribut yang dipilih!!")
                st.write(f"Dataset ini terdiri dari {new_df.shape[0]} baris dan {new_df.shape[1]} kolom.")

        with st.expander("Transform Data"):
            st.write(new_df.dtypes)

            if st.button("Encoding Atribut Satisfaction"):
                new_df['satisfaction'] = new_df["satisfaction"].replace({"satisfied": 1, "neutral or dissatisfied": 0})
                new_df['satisfaction'] = new_df['satisfaction'].astype('int64')
                st.write(new_df)
                st.write(new_df.dtypes)
        
        new_df['satisfaction'] = new_df["satisfaction"].replace({"satisfied": 1, "neutral or dissatisfied": 0})
        new_df['satisfaction'] = new_df['satisfaction'].astype('int64')
        with st.expander("Matriks Korelasi"):
            fig = plt.figure(figsize=(12, 9))
            sns.heatmap(new_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
            st.pyplot(fig)

            dff = new_df.drop(['Departure/Arrival time convenient', 'Gate location', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'], axis=1)
            if st.button("Drop Atribut yang Berkorelasi Negatif"):
                st.success(f"Berhasil menghapus atribut yang dipilih!")
                st.write(dff)
                st.write(f"Dataset ini terdiri dari {dff.shape[0]} baris dan {dff.shape[1]} kolom.")

        with st.expander("Drop Duplicates"):
            duplicates = dff[dff.duplicated()]
            st.write(duplicates)
            st.write(f"Dataset ini terdiri dari {duplicates.shape[0]} baris dan {duplicates.shape[1]} kolom.")
             # Tombol untuk menghapus duplikat
            if st.button("Drop Duplicate Data"):
                # Hapus duplikat dari DataFrame
                df_cleaned = dff.drop_duplicates()
                st.success("Duplikat berhasil dihapus!")
                st.write("Data Setelah Duplikat Dihapus:")
                st.write(dff)
                st.write(f"Dataset ini terdiri dari {df_cleaned.shape[0]} baris dan {df_cleaned.shape[1]} kolom.")
        

    # Menu Analisis Data
    elif main_menu == "🔍 Analisis Data":
        st.header("Analisis Data")
        
        # Split data
        x = df_cleaned.drop('satisfaction', axis=1)
        y = df_cleaned['satisfaction']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Oversample data training dengan SMOTE
        smote = SMOTE(random_state=42)
        x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

        rdf_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,  # Lebih dangkal untuk mencegah overfitting
            min_samples_split=10,  # Minimum jumlah sample untuk split
            min_samples_leaf=4,  # Minimum jumlah sample pada leaf
            max_features='sqrt',  # Fitur yang dipertimbangkan pada setiap split
            random_state=42,
            class_weight=None  # Menyeimbangkan bobot kelas per subsample
        )
        rdf_model.fit(x_train_smote, y_train_smote)

        # Menghitung feature importances
        feature_importances = pd.Series(rdf_model.feature_importances_, index=x.columns).sort_values(ascending=True)
        feature_importances_df = feature_importances.reset_index()
        feature_importances_df.columns = ['Feature', 'Importance']

        # Membuat bar chart horizontal
        fig = px.bar(
            feature_importances_df,
            x='Importance',
            y='Feature',
            orientation='h',
            labels={'Importance': 'Nilai', 'Feature': 'Fitur Importance'},
            title='Feature Importance Analysis'
        )

        # Menampilkan chart di Streamlit
        st.plotly_chart(fig)

        # Show model accuracy
        predict = rdf_model.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, predict)
        st.write("Tingkat Akurasi Model:", round(accuracy * 100, 2), "%")

        selected_features = [
            "Inflight wifi service",
            "Ease of Online booking",
            "Food and drink",
            "Online boarding",
            "Seat comfort",
            "Inflight entertainment",
            "On-board service",
            "Leg room service",
            "Baggage handling",
            "Checkin service",
            "Inflight service",
            "Cleanliness",
        ]

        # Visualisasi distribusi fitur berdasarkan kelas
        for feature in selected_features:
            fig = px.histogram(df_cleaned, x=feature, color="satisfaction", barmode="overlay",
                            title=f"Distribusi {feature} Berdasarkan Satisfaction")
            st.plotly_chart(fig)

        # Pastikan kolom yang dipilih ada di dataset
        if all(feature in df_cleaned.columns for feature in selected_features) and 'satisfaction' in df_cleaned.columns:
            # Filter data hanya dengan fitur yang dipilih dan target
            dff_filtered = df_cleaned[selected_features + ['satisfaction']]

            # Pisahkan fitur dan target
            x = dff_filtered[selected_features]
            y = dff_filtered['satisfaction']

            # Train-test split
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            smote = SMOTE(random_state=42)
            x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

            # Model Random Forest dengan class weights
            rdf_model = RandomForestClassifier(
                n_estimators=50, 
                max_depth=10, 
                min_samples_split=5, 
                random_state=42, 
                class_weight="balanced"
            )
            rdf_model.fit(x_train_smote, y_train_smote)

            # # Menambahkan input manual untuk prediksi
            # st.subheader("Masukkan Nilai untuk Memprediksi Tingkat Kepuasan")
            # user_input = {}
            # for feature in selected_features:
            #     user_input[feature] = st.slider(
            #         f"Nilai {feature} (0-5):",
            #         min_value=0,
            #         max_value=5,
            #         value=3,
            #         step=1
            #     )

            # Membuat input sebagai DataFrame
            # user_input_df = pd.DataFrame([user_input])

            # Prediksi satisfaction berdasarkan input user
            # user_prediction = rdf_model.predict(user_input_df)
            # prediction_label = "Puas" if user_prediction[0] == 1 else "Tidak Puas"
            # st.write("Prediksi Kepuasan Berdasarkan Input Anda:", prediction_label)

        # Recursive Feature Elimination
        rfe = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=5)
        rfe.fit(x_train, y_train)

        # Confusion Matrix
        cm = confusion_matrix(y_test, predict)
        fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"), 
                        title="Confusion Matrix")
        st.plotly_chart(fig_cm)


    elif main_menu == "📊 Visualisasi Data":
        st.header("Visualisasi Data")

        # "Persentase Jumlah Penerbangan Berdasarkan Satisfaction"
        satisfaction_counts = df['satisfaction'].value_counts()
        fig = px.pie(
            names=satisfaction_counts.index,
            values=satisfaction_counts.values,
            title='Percentage of Satisfaction',
            color_discrete_map={ 'Fri': 'cyan','Thur': 'lightcyan'}
        )
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label'
        )

        fig.update_layout(showlegend=True, legend_title_text='Satisfaction')
        st.plotly_chart(fig)

    # "Persentase Jumlah Penerbangan Berdasarkan Customer Type"
        satisfaction_counts = df['Customer Type'].value_counts()
        fig = px.pie(
            names=satisfaction_counts.index,
            values=satisfaction_counts.values,
            title='Percentage of Customer Type',
            color_discrete_map={'Thur': 'lightcyan', 'Fri': 'cyan'}
        )
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label'
        )

        fig.update_layout(showlegend=True, legend_title_text='Customer Type')
        st.plotly_chart(fig)

    # Perbandingan Jumlah Customer Type Berdasarkan Class
        fig = px.histogram(
            data, x='Class', color='Customer Type',
            title='Perbandingan Jumlah Customer Type Berdasarkan Class Penerbangan',
            labels={'Class': 'Class Penerbangan', 'count': 'Jumlah Penumpang'},
            barmode='group',  # Mengelompokkan berdasarkan nilai 'satisfaction'
            color_discrete_map={
            'High': 'dodgerblue',    
            'Low': 'blueviolet'        
        }

        )
        fig.update_layout(
            xaxis_title="Class Penerbangan",
            yaxis_title="Jumlah Penumpang",
            title_font_size=16,
            title_x=0 
        )
        st.plotly_chart(fig)

    # Perbandingan Jumlah Satisfaction Berdasarkan Class Penerbangan
        fig = px.histogram(
            data, x='Class', color='satisfaction',
            title='Perbandingan Jumlah Satisfaction Berdasarkan Class Penerbangan',
            labels={'Class': 'Class Penerbangan', 'count': 'Jumlah Penumpang'},
            barmode='group', 
            color_discrete_map={
                'High': 'mediumslateblue',   
                'Low': 'mediumturquoise'   
            } 
        )
        fig.update_layout(
            xaxis_title="Class Penerbangan",
            yaxis_title="Jumlah Penumpang",
            title_font_size=16,
            title_x=0 
        )
        
        st.plotly_chart(fig)

        fig = px.box(
            data,
            x="satisfaction",
            y="Age",
            color="satisfaction",
            title="Distribusi Umur Berdasarkan Satisfaction"
        )
        st.plotly_chart(fig)

        # Distribusi Skor Layanan Berdasarkan Tingkat Kepuasan
        st.subheader("Distribusi Skor Layanan Berdasarkan Tingkat Kepuasan")
        selected_service = st.selectbox(
            "Pilih Layanan",
            [
                "Inflight wifi service",
                "Ease of Online booking",
                "Food and drink",
                "Online boarding",
                "Seat comfort",
                "Inflight entertainment",
                "On-board service",
                "Leg room service",
                "Baggage handling",
                "Checkin service",
                "Inflight service",
                "Cleanliness",
            ],
        )

        # Menghitung frekuensi untuk setiap tingkat skor berdasarkan tingkat kepuasan
        line_data = data.groupby(["satisfaction", selected_service]).size().reset_index(name="Count")

        # Membuat grafik garis
        fig = px.line(
            line_data,
            x=selected_service,
            y="Count",
            color="satisfaction",
            title=f"Distribusi {selected_service} Berdasarkan Tingkat Kepuasan",
            labels={
                "satisfaction": "Tingkat Kepuasan",
                selected_service: "Rating",
                "Count": "Jumlah Pelanggan",
            },
            markers=True,  # Menambahkan titik data pada garis
            color_discrete_map={
                "neutral or dissatisfied": "orange",
                "satisfied": "green",
            },
        )

        st.plotly_chart(fig)

        # Rata-rata Skor Layanan Berdasarkan Customer Type dan Class
        st.subheader("Rata-rata Skor Layanan Berdasarkan Customer Type dan Class")

        # Menghitung rata-rata skor layanan
        avg_scores = data.groupby(['Customer Type', 'Class'])[
            [
                "Inflight wifi service",
                "Ease of Online booking",
                "Food and drink",
                "Online boarding",
                "Seat comfort",
                "Inflight entertainment",
                "On-board service",
                "Leg room service",
                "Baggage handling",
                "Checkin service",
                "Inflight service",
                "Cleanliness",
            ]
        ].mean()

        # Mereset index agar dapat digunakan untuk visualisasi
        avg_scores_reset = avg_scores.reset_index()

        # Filter data hanya untuk loyal customer
        loyal_customer_data = avg_scores_reset[avg_scores_reset['Customer Type'] == 'Loyal Customer']

        # Mengubah data menjadi format panjang (long format) untuk plotly
        loyal_customer_data_melted = loyal_customer_data.melt(
            id_vars=["Customer Type", "Class"], 
            var_name="Service", 
            value_name="Average Score"
        )

        # Membuat grafik garis
        fig = px.line(
            loyal_customer_data_melted,
            x="Service",
            y="Average Score",
            color="Class",
            line_group="Customer Type",
            markers=True,
            title="Rata-rata Skor Layanan Berdasarkan Customer Type dan Class (Loyal Customer)",
            labels={
                "Service": "Jenis Layanan",
                "Average Score": "Rata-rata Skor",
                "Class": "Kelas Penerbangan",
                "Customer Type": "Tipe Pelanggan"
            },
        )

        # Menyesuaikan tata letak
        fig.update_layout(
            xaxis_title="Jenis Layanan",
            yaxis_title="Rata-rata Skor",
            title_font_size=16,
            title_x=0,  # Posisi judul di tengah
            legend_title="Kelas Penerbangan",
            yaxis=dict(
                rangemode="tozero",  # Menetapkan sumbu Y dari nilai terendah
            ),
            xaxis=dict(
                tickangle=45,  # Mengubah posisi label sumbu X agar datar
            ),
        )

        st.plotly_chart(fig)

        sentiment_mapping = ["one", "two", "three", "four", "five"]
        selected = st.feedback("stars")
        if selected is not None:
            st.markdown(f"You selected {sentiment_mapping[selected]} star(s).")
        st.write("Give your feedback 😊 ")

else:
    st.warning("Silakan upload file CSV untuk memulai analisis.")
