import streamlit as st
import requests
from PIL import Image, ImageDraw
import io
import matplotlib.pyplot as plt
import base64
import numpy as np

st.set_page_config(
    page_title="Распознавание цифр MNIST",
    layout="wide"
)

st.title("Распознавание рукописных цифр")

API_URL = "http://localhost:8000"


def check_server_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


if not check_server_health():
    st.error("Сервер не запущен! Запустите сначала FastAPI сервер:")
    st.code("uvicorn main:app --reload --host 0.0.0.0 --port 8000")
    st.stop()

# Инициализация session_state
if 'drawing' not in st.session_state:
    st.session_state.drawing = None
if 'draw_result' not in st.session_state:
    st.session_state.draw_result = None
if 'upload_result' not in st.session_state:
    st.session_state.upload_result = None

# Создаем одну вкладку
tab1 = st.tabs(["Загрузка изображения"])[0]

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Загрузите изображение")

        uploaded_file = st.file_uploader(
            "Выберите изображение с цифрой",
            type=['png', 'jpg', 'jpeg']        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Загруженное изображение", use_container_width=True)

            if st.button("Распознать цифру", type="primary"):
                with st.spinner("Обрабатываем изображение..."):
                    try:
                        uploaded_file.seek(0)
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        response = requests.post(f"{API_URL}/predict", files=files)

                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.upload_result = result

                            st.metric(
                                label="Распознанная цифра",
                                value=f"{result['predicted_digit']}",
                                delta=f"Уверенность: {result['confidence']:.2%}"
                            )
                        else:
                            st.error(f"Ошибка при распознавании: {response.text}")

                    except requests.exceptions.RequestException as e:
                        st.error(f"Ошибка подключения к API: {e}")
                    except Exception as e:
                        st.error(f"Неожиданная ошибка: {e}")

    with col2:
        if st.session_state.upload_result is not None:
            result = st.session_state.upload_result

            st.subheader("Вероятности по всем классам:")
            fig, ax = plt.subplots(figsize=(10, 4))
            classes = list(result['all_probabilities'].keys())
            probabilities = list(result['all_probabilities'].values())

            bars = ax.bar(classes, probabilities, color='skyblue')
            ax.set_ylabel('Вероятность')
            ax.set_xlabel('Цифры')
            ax.set_title('Распределение вероятностей по классам')
            ax.set_ylim(0, 1)

            predicted_index = classes.index(result['predicted_class'])
            bars[predicted_index].set_color('red')

            for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{prob:.2%}', ha='center', va='bottom', fontsize=8)

            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)

        # Холст для рисования цифры
        st.subheader("Или нарисуйте цифру")
        html_canvas = """
        <!DOCTYPE html>
        <html>
        <body>
        <canvas id="myCanvas" width="280" height="280" style="border:2px solid #000000; cursor: crosshair; background-color: white;"></canvas>
        <br>
        <button onclick="clearCanvas()" style="padding: 8px 15px; margin: 5px; background: #ff4444; color: white; border: none; border-radius: 4px; cursor: pointer;">Очистить</button>
        <button onclick="saveCanvas()" style="padding: 8px 15px; margin: 5px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer;">Сохранить</button>

        <script>
        var canvas = document.getElementById('myCanvas');
        var ctx = canvas.getContext('2d');
        var painting = false;

        // Настройки
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.lineWidth = 15;
        ctx.lineCap = 'round';
        ctx.strokeStyle = 'black';

        function startPosition(e) {
            painting = true;
            draw(e);
        }

        function finishedPosition() {
            painting = false;
            ctx.beginPath();
        }

        function draw(e) {
            if(!painting) return;

            var rect = canvas.getBoundingClientRect();
            var x = e.clientX - rect.left;
            var y = e.clientY - rect.top;

            ctx.lineTo(x, y);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(x, y);
        }

        function clearCanvas() {
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.lineWidth = 15;
            ctx.lineCap = 'round';
            ctx.strokeStyle = 'black';
        }

        function saveCanvas() {
            var image = canvas.toDataURL('image/png');
            var link = document.createElement('a');
            link.download = 'digit.png';
            link.href = image;
            link.click();
        }

        // События
        canvas.addEventListener('mousedown', startPosition);
        canvas.addEventListener('mouseup', finishedPosition);
        canvas.addEventListener('mousemove', draw);
        </script>
        </body>
        </html>
        """

        st.components.v1.html(html_canvas, height=350)