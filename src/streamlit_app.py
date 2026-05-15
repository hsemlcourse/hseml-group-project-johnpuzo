import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


st.set_page_config(
    page_title="Recipe Quality Predictor",
    page_icon="🍽️",
    layout="centered",
)

st.title("🍽️ Recipe Quality Predictor")

st.write(
    "Приложение предсказывает качество рецепта: "
    "**bad**, **normal** или **good**."
)

st.markdown("### Проверка API")

try:
    health_response = requests.get(f"{API_URL}/health", timeout=5)

    if health_response.status_code == 200:
        st.success("API доступен, модель загружена.")
    else:
        st.error(f"API ответил с ошибкой: {health_response.status_code}")
        st.code(health_response.text)
        st.stop()

except requests.RequestException as exc:
    st.error("Не удалось подключиться к API.")
    st.code(str(exc))
    st.stop()


try:
    model_info = requests.get(f"{API_URL}/model-info", timeout=5).json()
    model_features = model_info.get("features", [])
except requests.RequestException:
    model_features = []


base_feature_names = {
    "minutes",
    "n_steps",
    "n_ingredients",
    "calories",
    "total_fat_pdv",
    "sugar_pdv",
    "sodium_pdv",
    "protein_pdv",
    "saturated_fat_pdv",
    "carbs_pdv",
}


st.markdown("### Режим ввода")

use_advanced = st.checkbox(
    "Показать расширенные признаки модели",
    value=False,
    help=(
        "Если выключено, вводятся только базовые признаки. "
        "Остальные признаки API заполнит пропусками, а модель обработает их через pipeline."
    ),
)


st.markdown("### Основные параметры рецепта")

with st.form("prediction_form"):
    minutes = st.number_input("Время приготовления, минут", min_value=1.0, value=45.0)
    n_steps = st.number_input("Количество шагов", min_value=1.0, value=8.0)
    n_ingredients = st.number_input("Количество ингредиентов", min_value=1.0, value=10.0)

    calories = st.number_input("Калории", min_value=0.0, value=350.0)
    total_fat_pdv = st.number_input("Total fat, %DV", min_value=0.0, value=20.0)
    sugar_pdv = st.number_input("Sugar, %DV", min_value=0.0, value=10.0)
    sodium_pdv = st.number_input("Sodium, %DV", min_value=0.0, value=15.0)
    protein_pdv = st.number_input("Protein, %DV", min_value=0.0, value=12.0)
    saturated_fat_pdv = st.number_input("Saturated fat, %DV", min_value=0.0, value=8.0)
    carbs_pdv = st.number_input("Carbs, %DV", min_value=0.0, value=14.0)

    advanced_values = {}

    if use_advanced and model_features:
        st.markdown("### Расширенные признаки")

        with st.expander("Дополнительные признаки модели", expanded=True):
            for feature in model_features:
                if feature in base_feature_names:
                    continue

                advanced_values[feature] = st.number_input(
                    label=feature,
                    value=0.0,
                    key=f"advanced_{feature}",
                )

    submitted = st.form_submit_button("Предсказать качество рецепта")


if submitted:
    features = {
        "minutes": minutes,
        "n_steps": n_steps,
        "n_ingredients": n_ingredients,
        "calories": calories,
        "total_fat_pdv": total_fat_pdv,
        "sugar_pdv": sugar_pdv,
        "sodium_pdv": sodium_pdv,
        "protein_pdv": protein_pdv,
        "saturated_fat_pdv": saturated_fat_pdv,
        "carbs_pdv": carbs_pdv,
    }

    if use_advanced:
        features.update(advanced_values)

    payload = {
        "features": features
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10,
        )

        if response.status_code != 200:
            st.error(f"Ошибка предсказания: {response.status_code}")
            st.code(response.text)
        else:
            result = response.json()

            label = result["prediction_label"]
            prediction_id = result["prediction_id"]
            probabilities = result.get("probabilities")

            st.markdown("## Результат")
            st.metric("Предсказанный класс", label)

            st.write(f"ID класса: `{prediction_id}`")

            if probabilities:
                st.markdown("### Вероятности классов")
                st.json(probabilities)

            st.markdown("### Техническая информация")
            st.write(f"Использовано признаков моделью: `{result['used_features_count']}`")
            st.write(f"Пропущено признаков: `{result['missing_features_count']}`")

            ignored_features = result.get("ignored_features", [])
            if ignored_features:
                st.write("Игнорированные признаки:")
                st.write(ignored_features)

    except requests.RequestException as exc:
        st.error("Не удалось отправить запрос к API.")
        st.code(str(exc))