import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message, InputFile
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import matplotlib.pyplot as plt
import io
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Инициализация бота и диспетчера
API_TOKEN = YOUR_TOKEN
bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# Пример данных для анализа тональности (обучающий набор)
train_data = [
    ("Я люблю футбол!", "positive"),
    ("Эта команда ужасна", "negative"),
    ("Матч был норм", "neutral")
]
texts, labels = zip(*train_data)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(texts)
clf = MultinomialNB().fit(X_train, labels)

# Обработчик команды /start
@dp.message(Command("start"))
async def send_welcome(message: Message):
    # Клавиатура с опциями
    builder = ReplyKeyboardBuilder()
    builder.button(text="Прогноз Золотого мяча 🏆")
    builder.button(text="Анализ тональности")
    builder.adjust(2)
    await message.answer(
        "Привет! Я бот для спортивных прогнозов и диалога ⚽\nВыбери опцию:",
        reply_markup=builder.as_markup(resize_keyboard=True)
    )

# Обработчик текстовых сообщений
@dp.message()
async def handle_message(message: Message):
    text = message.text.lower()
    
    # Анализ тональности
    if "анализ тональности" in text:
        await message.answer("Напиши сообщение, я определю его тональность 😊")
        return
    
    # Прогноз Золотого мяча
    if "прогноз золотого мяча" in text:
        try:
            # Актуальные фавориты на основе реальных прогнозов (Ousmane Dembele лидирует после успеха PSG в ЛЧ)
            players = ["Дембеле", "Ямаль", "Витинья", "Салах", "Рафинья"]
            probabilities = [45, 25, 15, 10, 5]  # Примерные % на основе букмекеров и СМИ
            
            # Создание графика
            plt.figure(figsize=(8, 5))
            bars = plt.bar(players, probabilities, color=['#36A2EB', '#FF6384', '#FFCE56', '#4BC0C0', '#9966FF'])
            plt.title("Прогноз: Золотой мяч 2025")
            plt.ylabel("Вероятность (%)")
            plt.ylim(0, 50)
            plt.xticks(rotation=45)
            
            # Добавь значения на бары для наглядности
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval}%', ha='center', va='bottom')
            
            # Сохранение в буфер
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plt.close()
            
            # InputFile для BytesIO
            from aiogram.types import InputFile  # Убедись, что импортировано
            photo = InputFile(buf, filename="forecast.png")
            
            # Отправка
            await message.answer_photo(photo=photo, caption="Прогноз на Золотой мяч 2025: Дембеле - главный фаворит после триумфа PSG в ЛЧ! 🏆")
            
            buf.close()  # Закрытие после отправки
            
        except Exception as e:
            logging.error(f"Подробная ошибка: {str(e)}")  # Для дебага
            await message.answer("Ошибка графика. Текстом: Фавориты - Дембеле (45%), Ямаль (25%), Витинья (15%). Дембеле выиграл благодаря ЛЧ и треблу PSG.<grok-card data-id=\"c43af3\" data-type=\"citation_card\"></grok-card><grok-card data-id=\"7931c6\" data-type=\"citation_card\"></grok-card>")
        return
    
    # Анализ тональности текста
    try:
        X_test = vectorizer.transform([text])
        prediction = clf.predict(X_test)[0]
        await message.answer(f"Тональность твоего сообщения: {prediction} 😎")
    except Exception as e:
        logging.error(f"Ошибка анализа тональности: {e}")
        await message.answer("Не удалось проанализировать тональность. Попробуй другое сообщение! 😅")
    
    # Имитация печати
    await bot.send_chat_action(message.chat.id, "typing")
    await asyncio.sleep(1)
    
    # Ответ на обычное сообщение
    await message.answer("Круто, расскажи ещё что-нибудь! ⚽")

# Запуск бота
async def main():
    try:
        await dp.start_polling(bot)
    except Exception as e:
        logging.error(f"Ошибка при запуске бота: {e}")

if __name__ == "__main__":
    asyncio.run(main())
