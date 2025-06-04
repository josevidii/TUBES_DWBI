import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi halaman
st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Fungsi untuk memuat model
@st.cache_resource
def load_models():
    # Load Naive Bayes model
    with open('naive_bayes_model.pkl', 'rb') as f:
        nb_model = pickle.load(f)
    
    # Load KMeans model
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans_output = joblib.load(f)
    
    return nb_model, kmeans_output

# Load models
nb_model_metrics, kmeans_output = load_models()
kmeans_model = kmeans_output['model']
df = kmeans_output['clustered_data']

# Sidebar untuk navigasi
st.sidebar.title("📊 Customer Analytics")
page = st.sidebar.radio("Pilih Section:", 
    ["🎯 Prediksi Pembelian", "👥 Segmentasi Pelanggan"])

if page == "🎯 Prediksi Pembelian":
    st.title("🎯 Prediksi Pembelian Pelanggan")
    st.markdown("### Prediksi kemungkinan pelanggan akan melakukan pembelian")

    # Buat 2 kolom dengan ukuran yang lebih proporsional
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Informasi Pelanggan")
        customer_age = st.number_input("Usia", min_value=18, max_value=100, value=30)
        customer_preferred_product_id = st.number_input("ID Produk Pilihan", value=1)

    with col2:
        st.markdown("#### Informasi Produk")
        product_base_price = st.number_input("Harga Dasar", min_value=0.0, value=100.0)
        product_default_discount = st.number_input("Diskon (%)", min_value=0.0, max_value=100.0, value=0.0)
        product_release_year = st.number_input("Tahun Rilis", min_value=2000, max_value=2025, value=2023)
        product_year_since_release = 2025 - product_release_year
        product_sales_factor = st.number_input("Faktor Penjualan", min_value=0.0, max_value=10.0, value=1.0)

    # Prediksi section dengan ukuran yang disesuaikan
    if st.button("Prediksi", use_container_width=True):
        try:
            input_data = pd.DataFrame({
                'customer_age': [customer_age],
                'product_base_price': [product_base_price],
                'product_default_discount': [product_default_discount],
                'product_release_year': [product_release_year],
                'product_year_since_release': [product_year_since_release],
                'product_sales_factor': [product_sales_factor],
                'customer_preferred_product_id': [customer_preferred_product_id],
                'sales_factor': [0],
                'discount': [product_default_discount],
                'year_since_release': [product_year_since_release],
                'price': [product_base_price],
                'release_year': [product_release_year],
                'preferred_product_id': [customer_preferred_product_id],
                'brand': ['unknown'],
                'storage': ['unknown'],
                'color': ['unknown'],
                'city': ['unknown'],
                'age': [customer_age],
                'category': ['unknown']
            })

            model = nb_model_metrics['model']
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)

            # Tampilkan hasil dalam container
            with st.container():
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Hasil Prediksi")
                    if prediction[0] == 1:
                        st.success("Kemungkinan akan membeli! 🎯")
                    else:
                        st.error("Kemungkinan tidak membeli 😔")

                with col2:
                    st.markdown("#### Probabilitas")
                    prob_yes = prediction_proba[0][1]
                    st.progress(prob_yes)
                    st.write(f"Probabilitas Pembelian: {prob_yes:.2%}")

        except Exception as e:
            st.error(f"Error: {str(e)}")

else:
    st.title("👥 Segmentasi Pelanggan")
    st.markdown("### Analisis Cluster menggunakan K-Means")
    
    # Tambahkan penjelasan clustering
    # st.markdown("""
    # #### Tentang K-Means Clustering
    # K-Means clustering adalah metode yang digunakan untuk membagi pelanggan ke dalam 4 segmen berdasarkan:
    # - **RFM Analysis**: Recency (waktu sejak pembelian terakhir), Frequency (frekuensi pembelian), 
    #   dan Monetary (total nilai pembelian)
    # - **Demografis**: Usia pelanggan
    # - **Perilaku**: Jumlah transaksi dan rata-rata kuantitas per transaksi
    
    # Setiap segmen memiliki karakteristik unik yang dapat membantu dalam strategi pemasaran yang lebih tepat sasaran.
    # """)

    try:
        # Analisis cluster code
        cluster_counts = df['Cluster'].value_counts().sort_index()
        
        selected_metrics = ['age', 'Total_Revenue', 'Transaction_Count', 
                          'Avg_Quantity_Per_Transaction', 'Recency', 'Frequency', 'Monetary']
        cluster_means = df.groupby('Cluster')[selected_metrics].mean()
        
        metrics = ['Recency', 'Frequency', 'Monetary', 'age', 'Transaction_Count']
        
        df_normalized = cluster_means[metrics].copy()
        for column in df_normalized.columns:
            df_normalized[column] = (df_normalized[column] - df_normalized[column].min()) / (df_normalized[column].max() - df_normalized[column].min())
        
        # Scatter plot dalam container
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("#### Distribusi Pelanggan berdasarkan RFM")
                fig, ax = plt.subplots(figsize=(8, 5))
                scatter = plt.scatter(df['Monetary'], 
                                    df['Frequency'],
                                    c=df['Cluster'],
                                    cmap='viridis',
                                    alpha=0.6)
                plt.colorbar(scatter, label='Cluster')
                plt.title("Segmentasi Pelanggan (Monetary vs Frequency)", pad=10, fontsize=12)
                plt.xlabel("Monetary (Total Nilai Pembelian)", fontsize=10)
                plt.ylabel("Frequency (Frekuensi Pembelian)", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)

        # Karakteristik cluster dengan interpretasi
        with st.container():
            st.markdown("### Karakteristik dan Interpretasi Segmen")
            
            # Gunakan tabs untuk memisahkan informasi
            tab1, tab2 = st.tabs(["📊 Data Metrik", "📋 Interpretasi"])
            
            with tab1:
                # Tampilkan dataframe tanpa styling gradien warna
                st.dataframe(
                    cluster_means.round(2),
                    use_container_width=True,
                    height=200
                )
                
            with tab2:
                # Bagi menjadi dua kolom yang seimbang
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    #### 📈 Interpretasi Metrik
                    - **Recency**: Waktu sejak pembelian terakhir
                    - **Frequency**: Frekuensi pembelian
                    - **Monetary**: Total nilai pembelian
                    - **Age**: Usia pelanggan
                    - **Transaction**: Jumlah transaksi
                    """)
                    
                with col2:
                    st.markdown("""
                    #### 👥 Karakteristik Segmen
                    - **Segmen Reguler**: Pelanggan setia dengan pembelian rutin
                    - **Segmen Premium**: High-value customers dengan transaksi tinggi
                    - **Segmen Casual**: Pelanggan occasional dengan nilai transaksi rendah
                    - **Segmen Potensial**: Pelanggan dengan potensi upgrade
                    """)

        # Radar charts dengan layout yang lebih rapi
        with st.container():
            st.markdown("### Profil Detail Segmen")
            
            # Ubah menjadi satu kolom penuh
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), subplot_kw=dict(projection='polar'))
            axes = axes.flatten()

            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))

            colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
            cluster_names = ['Reguler', 'Premium', 'Casual', 'Potensial']

            for idx in range(4):
                values = df_normalized.iloc[idx][metrics].values
                values = np.concatenate((values, [values[0]]))
                
                axes[idx].plot(angles, values, color=colors[idx], linewidth=2)
                axes[idx].fill(angles, values, color=colors[idx], alpha=0.25)
                axes[idx].set_xticks(angles[:-1])
                axes[idx].set_xticklabels(metrics, size=8)
                axes[idx].set_title(f'Segmen {cluster_names[idx]}', pad=15, size=10)
                axes[idx].grid(True, linestyle='--', alpha=0.7)
                axes[idx].set_ylim(0, 1)

            plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)
            st.pyplot(fig)
            
            with col2:
                pass

    except Exception as e:
        st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Dibuat oleh Kelompok 1 ❤️")