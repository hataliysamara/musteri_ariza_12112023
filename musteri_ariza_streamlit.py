import streamlit as st
import pandas as pd
import numpy as np

#streamlit run C:\Users\185681\Desktop\code_with_ai\proje3\streamlit\musteri_ariza_streamlit.py

# Sayfa Ayarları
st.set_page_config(
    page_title="Müşteri Ariza Classification",
    page_icon="data:image/jpeg;base64,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",
    menu_items={
        "Get help": "mailto:berk.kizilkanat@turktelekom.com.tr",
        "About": "For More Information\n" + "https://github.com/hataliysamara/TT_REPO1"
    }
)

# Başlık Ekleme
st.title("Müşteri Ariza Siniflandirma Projesi")

# Markdown Oluşturma
st.markdown("Türk Telekom Şirketininin Abonelerin Ariza Birakip Birakmayacagini Tahminleyen Uygulamadir")

# Resim Ekleme
st.image("https://ares.shiftdelete.net/2022/08/turk-telekom-2022-finansal-rapor.jpg")

# Header Ekleme
st.header("Veri İçeriği")

st.markdown("- **Müşteri Ariza**: Müşteri Son ay Ariza Birakmis mi?")
st.markdown("- **Altyapi**: Abonenin Altyapi Bilgisi(BAKIR,FTTC,FTTB,FTTH)")
st.markdown("- **Santral Mesafe**:Abone Santral Mesafesi")
st.markdown("- **Attenuation**:Hattaki Zayiflama Değeri")
st.markdown("- **Noise Margin**:   Hattin Kalite Değeri")
st.markdown("- **Hizmet Kalitesi**:  Abonenin sözleşme hizinin karsilanma orani")

# Pandasla veri setini okuma
df=pd.read_table(r'C:\Users\185681\Desktop\code_with_ai\proje3\streamlit\musteri_ariza_classification_data.txt')


# Tablo Gösterimi
st.table(df.sample(5, random_state=42))

# Sidebarda Markdown Oluşturma
st.sidebar.markdown("**Aşağidan Deger Seçin")

# Sidebarda Kullanıcıdan Girdileri Alma
ALTYAPI = st.sidebar.number_input("Altyapi Değeri Giriniz", min_value=1,max_value=250, format="%d")
SANTRAL_MESAFESİ = st.sidebar.number_input("Santral Mesafesi Giriniz", min_value=1,max_value=2000, format="%d")
ATTENUATION = st.sidebar.number_input("Attenuation Değeri Giriniz", min_value=1,max_value=40, format="%d")
NOISE_MARGIN = st.sidebar.number_input("Noise Margin Değeri Giriniz", min_value=1,max_value=40, format="%d")
HIZMET_KALITESI = st.sidebar.number_input("Hizmet Kalite Değeri Giriniz", min_value=0,max_value=1, format="%d")

# Pickle kütüphanesi kullanarak eğitilen modelin tekrardan kullanılması
from joblib import load

ros_model = load(r'C:\Users\185681\Desktop\code_with_ai\proje3\streamlit\svc_adasyn.pkl')

input_df = pd.DataFrame({
    'ALTYAPI': [ALTYAPI],
    'SANTRAL_MESAFESİ': [SANTRAL_MESAFESİ],
    'ATTENUATION': [ATTENUATION],
    'NOISE_MARGIN': [NOISE_MARGIN],
    'HIZMET_KALITESI': [HIZMET_KALITESI]
})

pred = ros_model.predict(input_df.values)

#---------------------------------------------------------------------------------------------------------------------

st.header("Results")

# Sonuç Ekranı
if st.sidebar.button("Submit"):

    # Info mesajı oluşturma
    st.info("Sonucu Aşagida Görebilirsiniz.")

    # Sonuçları Görüntülemek için DataFrame
    results_df = pd.DataFrame({
    'ALTYAPI': [ALTYAPI],
    'SANTRAL_MESAFESİ': [SANTRAL_MESAFESİ],
    'ATTENUATION': [ATTENUATION],
    'NOISE_MARGIN': [NOISE_MARGIN],
    'HIZMET_KALITESI': [HIZMET_KALITESI],
    'Prediction': [pred]
    })

    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("0","ARIZA_YOK"))
    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("1","ARIZA_VAR"))

    st.table(results_df)
else:
    st.markdown("Please click the *Submit Button*!")
    





