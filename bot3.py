import os
import asyncio
import sqlite3
import random
import requests
import pandas as pd
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

TELEGRAM_TOKEN = "8141508085:AAGvUM92ApTNpJHW_YNcyJlvYuk5jUCDsLU"
DB_PATH = "premier_league_2008_2009.db"
CSV_URL = "https://www.football-data.co.uk/mmz4281/0809/E0.csv"

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

def ensure_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        home_team TEXT,
        away_team TEXT,
        home_goals INTEGER,
        away_goals INTEGER,
        winner TEXT
    )""")
    conn.commit()
    cur.execute("SELECT COUNT(*) FROM matches")
    if cur.fetchone()[0] == 0:
        try:
            r = requests.get(CSV_URL, timeout=10)
            r.raise_for_status()
            df = pd.read_csv(r.content.decode('utf-8'))
        except Exception:
            df = pd.DataFrame([
                ["2008-08-16", "Arsenal", "West Brom", 1, 0, "home"],
                ["2008-08-16", "Man United", "Newcastle", 1, 1, "draw"],
                ["2008-08-17", "Liverpool", "Sunderland", 1, 0, "home"]
            ], columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Winner"])
        if "Winner" not in df.columns:
            df["Winner"] = df.apply(
                lambda x: "home" if x["FTHG"] > x["FTAG"] else ("away" if x["FTAG"] > x["FTHG"] else "draw"), axis=1
            )
        rows = [(x["Date"], x["HomeTeam"], x["AwayTeam"], int(x["FTHG"]), int(x["FTAG"]), x["Winner"]) for _, x in df.iterrows()]
        cur.executemany("INSERT INTO matches (date, home_team, away_team, home_goals, away_goals, winner) VALUES (?,?,?,?,?,?)", rows)
        conn.commit()
    conn.close()

def load_df():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM matches", conn)
    conn.close()
    return df

def build_model(df):
    if df.empty:
        return None, None
    le = LabelEncoder()
    teams = pd.concat([df["home_team"], df["away_team"]]).unique()
    le.fit(teams)
    df["home_enc"] = le.transform(df["home_team"])
    df["away_enc"] = le.transform(df["away_team"])
    X = df[["home_enc", "away_enc", "home_goals", "away_goals"]]
    y = df["winner"]
    model = LogisticRegression(max_iter=300)
    model.fit(X, y)
    return model, le

ensure_db()
df_global = load_df()
model_global, le_global = build_model(df_global)

@dp.message(Command("start"))
async def cmd_start(m: types.Message):
    kb = [
        [types.KeyboardButton(text="📜 Последние 5 матчей"), types.KeyboardButton(text="🔎 Поиск по команде")],
        [types.KeyboardButton(text="🎱 8 Ball"), types.KeyboardButton(text="🤖 Прогноз: TeamA vs TeamB")]
    ]
    keyboard = types.ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)
    await m.answer("Привет! База Premier League 2008/09 готова ⚽", reply_markup=keyboard)

@dp.message()
async def handle_message(m: types.Message):
    txt = m.text.strip()
    if txt == "📜 Последние 5 матчей":
        df = df_global.sort_values("date", ascending=False).head(5)
        await m.answer("\n".join(f"{r['date']} — {r['home_team']} {r['home_goals']}:{r['away_goals']} {r['away_team']}" for _, r in df.iterrows()))
    elif txt == "🎱 8 Ball":
        await m.answer(random.choice(["Да", "Нет", "Возможно", "Сомневаюсь", "Позже", "Сегодня не твой день","Сегодня тебе можно все"]))
    elif "vs" in txt:
        try:
            home, away = [x.strip() for x in txt.split("vs")]
            if home not in le_global.classes_ or away not in le_global.classes_:
                await m.answer("Команды не найдены в базе.")
                return
            h = int(le_global.transform([home])[0])
            a = int(le_global.transform([away])[0])
            X = pd.DataFrame([[h, a, 0, 0]], columns=["home_enc", "away_enc", "home_goals", "away_goals"])
            pred = model_global.predict(X)[0]
            await m.answer(f"Прогноз: {pred}")
        except Exception:
            await m.answer("Не удалось сделать прогноз.")
    else:
        df = df_global[(df_global["home_team"].str.lower() == txt.lower()) | (df_global["away_team"].str.lower() == txt.lower())].head(5)
        if df.empty:
            await m.answer("Матчи не найдены.")
        else:
            await m.answer("\n".join(f"{r['date']} — {r['home_team']} {r['home_goals']}:{r['away_goals']} {r['away_team']}" for _, r in df.iterrows()))

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
