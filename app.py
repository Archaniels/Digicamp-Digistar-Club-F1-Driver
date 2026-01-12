import streamlit as st
import joblib
import pandas as pd
import numpy as np

@st.cache_resource
def load_models():
    model = joblib.load("f1_champion_model.pkl")
    scaler = joblib.load("f1_scaler.pkl")
    features = joblib.load("f1_features.pkl")
    return model, scaler, features

model, scaler, features = load_models()

st.set_page_config(
    page_title="F1 Champion Predictor",
    page_icon="🏎️",
    layout="wide"
)

st.title("🏎️ Formula 1 World Champion Predictor")
st.markdown("""
Aplikasi ini memprediksi **apakah seorang driver Formula 1 berpotensi menjadi Juara Dunia** 
berdasarkan statistik performa mereka menggunakan Machine Learning.
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Statistik Balapan")
    
    race_entries = st.number_input(
        "Total Balapan Diikuti (Race Entries)",
        min_value=0,
        max_value=400,
        value=100,
        help="Jumlah total balapan yang diikuti driver"
    )
    
    race_starts = st.number_input(
        "Total Race Starts",
        min_value=0,
        max_value=400,
        value=95,
        help="Jumlah balapan yang benar-benar dimulai"
    )
    
    pole_positions = st.number_input(
        "Pole Positions",
        min_value=0,
        max_value=110,
        value=10,
        help="Jumlah start dari posisi pertama"
    )
    
    race_wins = st.number_input(
        "Race Wins (Kemenangan)",
        min_value=0,
        max_value=110,
        value=8,
        help="Jumlah kemenangan di balapan"
    )
    
    podiums = st.number_input(
        "Podiums (Podium)",
        min_value=0,
        max_value=200,
        value=25,
        help="Jumlah finish di posisi 1-3"
    )
    
    fastest_laps = st.number_input(
        "Fastest Laps",
        min_value=0,
        max_value=80,
        value=5,
        help="Jumlah lap tercepat dalam balapan"
    )
    
    points = st.number_input(
        "Total Points",
        min_value=0.0,
        max_value=5000.0,
        value=500.0,
        step=10.0,
        help="Total poin yang dikumpulkan sepanjang karir"
    )

with col2:
    st.subheader("📈 Rasio Performa")
    
    pole_rate = pole_positions / race_entries if race_entries > 0 else 0
    start_rate = race_starts / race_entries if race_entries > 0 else 0
    win_rate = race_wins / race_entries if race_entries > 0 else 0
    podium_rate = podiums / race_entries if race_entries > 0 else 0
    fastlap_rate = fastest_laps / race_entries if race_entries > 0 else 0
    points_per_entry = points / race_entries if race_entries > 0 else 0
    
    st.metric("Pole Rate", f"{pole_rate:.3f}")
    st.metric("Start Rate", f"{start_rate:.3f}")
    st.metric("Win Rate", f"{win_rate:.3f}")
    st.metric("Podium Rate", f"{podium_rate:.3f}")
    st.metric("Fast Lap Rate", f"{fastlap_rate:.3f}")
    st.metric("Points Per Entry", f"{points_per_entry:.2f}")
    
    st.divider()
    
    years_active = st.slider(
        "Tahun Aktif di F1",
        min_value=1,
        max_value=25,
        value=5,
        help="Jumlah tahun karir di Formula 1"
    )

st.divider()

input_data = {
    'Race_Entries': race_entries,
    'Race_Starts': race_starts,
    'Pole_Positions': pole_positions,
    'Race_Wins': race_wins,
    'Podiums': podiums,
    'Fastest_Laps': fastest_laps,
    'Points': points,
    'Pole_Rate': pole_rate,
    'Start_Rate': start_rate,
    'Win_Rate': win_rate,
    'Podium_Rate': podium_rate,
    'FastLap_Rate': fastlap_rate,
    'Points_Per_Entry': points_per_entry,
    'Years_Active': years_active
}

if st.button("🔮 Prediksi Potensi Juara Dunia", type="primary", use_container_width=True):
    df = pd.DataFrame([input_data])
    
    df_scaled = scaler.transform(df)
    
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0]
    
    st.divider()
    st.subheader("🎯 Hasil Prediksi")
    
    col_result1, col_result2, col_result3 = st.columns(3)
    
    with col_result1:
        if prediction == 1:
            st.success("### ✅ CHAMPION POTENTIAL")
            st.markdown("Driver ini memiliki **karakteristik juara dunia!**")
        else:
            st.warning("### ❌ NON-CHAMPION")
            st.markdown("Driver ini **belum menunjukkan** statistik juara dunia.")
    
    with col_result2:
        champion_prob = probability[1] * 100
        st.metric(
            "Probabilitas Champion",
            f"{champion_prob:.1f}%",
            help="Tingkat keyakinan model"
        )
        
        st.progress(champion_prob / 100)
    
    with col_result3:
        if champion_prob >= 80:
            st.info("🌟 **Sangat Tinggi**\n\nStatistik luar biasa!")
        elif champion_prob >= 60:
            st.info("🔥 **Tinggi**\n\nPotensi besar!")
        elif champion_prob >= 40:
            st.info("📊 **Sedang**\n\nPerlu peningkatan.")
        else:
            st.info("📉 **Rendah**\n\nMasih berkembang.")
    
    st.divider()
    st.subheader("💡 Insight Tambahan")
    
    col_i1, col_i2 = st.columns(2)
    
    with col_i1:
        st.markdown("**📌 Statistik Kunci:**")
        if win_rate >= 0.15:
            st.success(f"✅ Win Rate tinggi: {win_rate:.2%}")
        else:
            st.info(f"ℹ️ Win Rate: {win_rate:.2%} (Champion avg: ~15-30%)")
            
        if podium_rate >= 0.25:
            st.success(f"✅ Podium Rate bagus: {podium_rate:.2%}")
        else:
            st.info(f"ℹ️ Podium Rate: {podium_rate:.2%} (Champion avg: ~25-50%)")
    
    with col_i2:
        st.markdown("**🎯 Rekomendasi:**")
        if win_rate < 0.10:
            st.write("🔸 Fokus meningkatkan kemenangan")
        if podium_rate < 0.20:
            st.write("🔸 Konsistensi podium perlu ditingkatkan")
        if pole_rate < 0.05:
            st.write("🔸 Kualifikasi bisa lebih baik")
        if points_per_entry < 2.0:
            st.write("🔸 Perlu lebih banyak poin per balapan")