import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
from sklearn.linear_model import LogisticRegression
import numpy as np

API_TOKEN = YOUR_TOKEN

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# ==== Команды и их сила/опыт ====
teams = {
    "Real Madrid": [90, 30],
    "Barcelona": [88, 28],
    "PSG": [85, 25],
    "Bayern": [87, 27],
    "Man City": [92, 29],
    "Arsenal": [80, 22],
    "Inter": [83, 23],
    "Chelsea": [78, 20],
}

# ==== Обучающие данные ====
# [разница силы, разница опыта] -> исход
X = np.array([
    [10, 5], [5, 2], [-10, -5], [0, 0],
    [15, 10], [-20, -10], [8, 3], [-5, 0],
    [12, 7], [0, -2], [-15, -8], [3, 1]
])
y = np.array([2, 2, 0, 1, 2, 0, 2, 0, 2, 1, 0, 1])  # 0=поражение, 1=ничья, 2=победа

model = LogisticRegression(max_iter=200)
model.fit(X, y)

labels = {0: "❌ Победа гостей", 1: "😐 Ничья", 2: "🏆 Победа хозяев"}

user_data = {}

@dp.message(Command("start"))
async def start(message: types.Message):
    user_data[message.from_user.id] = {}
    team_list = "\n".join([f"⚽ {t}" for t in teams.keys()])
    await message.answer(
        f"Привет! ⚽\nВот список доступных команд:\n\n{team_list}\n\n"
        "Введи домашнюю команду:"
    )

@dp.message()
async def handle_message(message: types.Message):
    user_id = message.from_user.id
    text = message.text.strip()

    if user_id not in user_data:
        user_data[user_id] = {}

    data = user_data[user_id]

    if "home" not in data:
        if text not in teams:
            await message.answer("❌ Такой команды нет. Попробуй снова.")
            return
        data["home"] = text
        await message.answer("Теперь введи гостевую команду:")
    elif "away" not in data:
        if text not in teams:
            await message.answer("❌ Такой команды нет. Попробуй снова.")
            return
        data["away"] = text
        home = teams[data["home"]]
        away = teams[data["away"]]

        diff = [home[0] - away[0], home[1] - away[1]]
        pred = model.predict([diff])[0]
        prob = model.predict_proba([diff]).max() * 100

        await message.answer(
            f"📊 Прогноз:\n"
            f"🏠 {data['home']} vs 🏃 {data['away']}\n\n"
            f"{labels[pred]} ({prob:.1f}% уверенности)"
        )
        user_data[user_id] = {}  # очистка для нового прогноза
        await message.answer("Хочешь предсказать ещё матч? Введи домашнюю команду:")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
