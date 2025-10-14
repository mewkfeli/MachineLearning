import streamlit as st
import requests
from PIL import Image

st.set_page_config(
    page_title="Классификатор изображений",
    layout="wide"
)

def main():
    st.title("Классификатор изображений")
    st.write("Загрузите изображение для распознавания крыс, хомяков и шиншилл")

    # Загрузка изображения
    uploaded_file = st.file_uploader(
        "Выберите изображение для распознавания",
        type=['jpg', 'jpeg', 'png', 'bmp', 'gif']
    )

    if uploaded_file is not None:
        # Показываем изображение с фиксированным размером
        image = Image.open(uploaded_file)
        st.image(image, caption="Загруженное изображение", width=400)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Распознать изображение", type="primary", use_container_width=True):
                recognize_image(uploaded_file, image)

        with col2:
            if st.button("Очистить", use_container_width=True):
                st.rerun()


def recognize_image(uploaded_file, image):
    """Распознавание изображения"""
    with st.spinner("Идет распознавание..."):
        files = {
            'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }

        response = requests.post(
            "http://localhost:8000/predict",
            files=files,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            display_results(result, image)
        else:
            error_data = response.json()
            st.error(f"Ошибка API: {error_data.get('detail', 'Unknown error')}")


def display_results(result, image):
    """Отображение результатов"""
    st.success("Распознавание завершено!")

    col1, col2 = st.columns([1, 2])

    with col2:
        st.subheader("Результаты")

        confidence = float(result['confidence'])
        confidence_percent = result['confidence_percentage']

        st.write(f"**Класс:** {result['class_name']}")
        st.write(f"**Точность:** {confidence_percent}")

        st.progress(confidence, text=f"Уверенность модели: {confidence_percent}")

        if confidence > 0.8:
            st.success("Высокая уверенность в результате")
        elif confidence > 0.6:
            st.warning("Средняя уверенность в результате")
        else:
            st.error("Низкая уверенность в результате")

        with st.expander("Детальная информация"):
            st.write(f"**Вероятность:** {confidence:.4f}")
            st.write(f"**Процент уверенности:** {confidence_percent}")
            st.write(f"**Имя класса:** {result['class_name']}")

if __name__ == "__main__":
    main()