import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


df_raw = None # исх данные
df_work = None # Рабочая копия
fig = plt.Figure(figsize=(9,5.5),dpi=100) 
canvas = None
current_chart = "line" #линейная диаграмма
#Поддержка русского
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    print("Загрузка файла...")
    global df_raw 
    df_raw = pd.read_csv("data.csv",encoding="utf-8").sample(n = 250000,random_state=42) 
    #.sample(n = 50000,random_state=42) # Берёт 50тыс случайных строк из таблицы
    print("Данные загружены")

def preprocess_data():
    global df_raw
    df = df_raw.copy()
    #Очистка от аномалий
    df = df[(df["time_min"] >= 0) & (df["score"] >= 0) &  (df["score"] <= 100)]
    df = df.dropna(subset=["score","time_min"]) #Удаляет строки в которых есть пропуски (NaN)
    df["attempts"] = df["attempts"].replace(0,1)
    #Обрезка по IQR
    q1 = df["time_min"].quantile(0.25)
    q3 = df["time_min"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df["time_min"] = df["time_min"].clip(lower,upper)
    # дата и время
    df["datetime"] = pd.to_datetime(df["ts"],unit='s') # из unix секундв в нормальное время
    # День недели
    df["weekday"] = df["datetime"].dt.day_name() # превращает дату в текстовый день недели
    # Оценка
    df["grade_category"] = pd.cut(df["score"], # cut нарезает цифры на категории(интервалы)
                                  bins=[-np.inf,50,75,90,np.inf],
                                  labels=["Неуд","Удовл","Хор","Отл"])
    #Сложность задачи
    df["diff_text"] = pd.cut(df["diff"],
                             bins = 3,
                             labels=["Лёгкие","Средние","Сложные"])
    #Оптимизация
    category_cols = ["weekday","grade_category","diff_text"]
    for col in category_cols:
        df[col] = df[col].astype("category") #переводит в формат категорий

    selected_diff = diff_var.get()
    if selected_diff != "Все":
            df = df[df["diff_text"] == selected_diff]
   
    return df

def clear_figure():
    fig.clear() 

def plot_line(): #линейный график
    global current_chart,df_work
    current_chart = "line"
    clear_figure()
    #Создает оси X и Y на холсте. 111 означает "сетка 1x1, график номер 1".
    ax = fig.add_subplot(111) 
    #чтобы работать со временем, дата должна стать индексом
    df_ts = df_work.set_index("datetime").sort_index()
    #разбивает все наши точки на недели (W) и считает среднюю оценку за каждую неделю.
    weekly_scores = df_ts["score"].resample("W").mean().dropna()
    # скользящее среднее. сглаживаем график за 4 недели 
    rolling_mean = weekly_scores.rolling(window=4,min_periods=1).mean()
    # отрисовка графиков
    sns.lineplot(x=weekly_scores.index,y=weekly_scores.values,label="Средний бал (неделя)",ax=ax,color="blue",alpha=0.4)
    sns.lineplot(x=rolling_mean.index,y=rolling_mean.values,label="Тренд(1 месяц)",ax = ax,color = "red",linewidth=2)
    ax.set_title("Динамика успеваемости во времени")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Средний балл")
    ax.tick_params(axis='x',rotation=45) # Поворачиваем подписи дат на 45 градусов

    fig.tight_layout()
    if canvas:
        canvas.draw_idle()

def plot_bar(): #столбчатая диаграмма
    global current_chart,df_work
    current_chart = "bar"
    clear_figure()
    ax=fig.add_subplot(111)
    #собираем все строки таблицы в кучки по дням недели
    # считаем ср кол попыток для каждого дня
    agg_data=df_work.groupby("weekday",observed=True)["attempts"].mean().reset_index()
    #Рисуем столбцы
    sns.barplot(data=agg_data,x="weekday",y="attempts",ax=ax,palette="viridis")
    
    ax.set_title("Среднее количество попыток по дням недели")
    ax.set_xlabel("День недели")
    ax.set_ylabel("Среднее число попыток")
    ax.tick_params(axis='x',rotation=45)
    fig.tight_layout()
    if canvas:
        canvas.draw_idle()

def plot_scatter(): #Точечная диаграмма
    global current_chart,df_work
    current_chart = "scatter"
    clear_figure()
    ax=fig.add_subplot(111)
    # Если нарисовать 50 000 точек, они сольются в одно огромное непроглядное пятно. 
    # Поэтому мы берем случайную тысячу точек только для этого графика.
    sample_size=min(1000,len(df_work))
    sample_df = df_work.sample(n=sample_size,random_state=42)
    #Отрисовка графика
    custom_colors = {"Лёгкие": "#28a745", "Средние": "#fd7e14", "Сложные": "#dc3545"}
    sns.scatterplot(data=sample_df,x="time_min",y="score",hue="diff_text",palette=custom_colors,alpha=0.7,ax=ax)

    ax.set_title("Зависимость балла от времени решения")
    ax.set_xlabel("Время выплнения (мин)")
    ax.set_ylabel("Оценка (баллы)")
    ax.legend(title="Сложность")

    fig.tight_layout()
    if canvas:
        canvas.draw_idle()

def plot_heat_map():
    global current_chart,df_work
    current_chart = "heatmap"
    clear_figure()
    ax=fig.add_subplot(111)
    #Сводная таблица (матрица)
    pivot = pd.pivot_table(df_work,
                           values="score",
                           index="diff_text",
                           columns="grade_category",
                           aggfunc="count",
                           fill_value=0)
    #Отрисовка тепловой карты
    sns.heatmap(pivot,annot=True,fmt="d",cmap="YlGnBu",ax=ax)

    ax.set_title("Распределение оценок по сложности задач")
    ax.set_xlabel("Категория оценки")
    ax.set_ylabel("Сложность задачи")
    
    fig.tight_layout()
    if canvas:
        canvas.draw_idle()


def refresh_data(*args):
    global df_work
    #Заново прогоняем сырые данные через фильтры и очистку
    df_work = preprocess_data()
    #Смотрим, какой график сейчас открыт, и перерисовываем именно его
    if current_chart == "line": 
        plot_line()
    elif current_chart == "bar": 
        plot_bar()
    elif current_chart == "scatter": 
        plot_scatter()
    elif current_chart == "heatmap": 
        plot_heat_map()

def export_plot():
    #сохранение файла
    filepath = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG","*.png"), ("PDF","*.pdf")])
    if filepath:
        fig.savefig(filepath,dpi=300,bbox_inches="tight")
        messagebox.showinfo("Экспорт",f"График успешно сохранён:\n{filepath}")



#Интерфейс
root = tk.Tk()
root.title("Логи LMS")
root.geometry("1100x700")
#панель управления
ctrl_frame = tk.Frame(root,bg="#e1e4e8",pady=10,padx=10)
ctrl_frame.pack(side=tk.TOP,fill=tk.X)
#Фильтр
tk.Label(ctrl_frame, text="Сложность:", bg="#e1e4e8").pack(side=tk.LEFT, padx=5)
diff_var = tk.StringVar(value="Все") 
diff_combo = ttk.Combobox(ctrl_frame, textvariable=diff_var, values=["Все", "Лёгкие", "Средние", "Сложные"], state="readonly", width=10)
diff_combo.pack(side=tk.LEFT, padx=5)
#Кнопки
tk.Button(ctrl_frame,text="Тренды",command=plot_line,width=12).pack(side=tk.LEFT,padx=5)
tk.Button(ctrl_frame,text="Дни недели",command=plot_bar,width=14).pack(side=tk.LEFT,padx=5)
tk.Button(ctrl_frame,text="Зависимости",command=plot_scatter,width=15).pack(side=tk.LEFT,padx=5)
tk.Button(ctrl_frame,text="Матрица",command=plot_heat_map,width=12).pack(side=tk.LEFT,padx=5)
tk.Button(ctrl_frame,text="Обновить",command=refresh_data,width=12,bg="#cce5ff").pack(side=tk.RIGHT,padx=5)
tk.Button(ctrl_frame,text="Экспорт",command=export_plot,width=12,bg="#d4edda").pack(side=tk.RIGHT,padx=5)
#Рамка графика
plot_frame= tk.Frame(root,bg="white",relief=tk.SUNKEN,bd=1)
plot_frame.pack(fill=tk.BOTH,expand=True,padx=10,pady=10)
# Холст графика с интерфейсом
canvas = FigureCanvasTkAgg(fig,master=plot_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True)
# Панель инструментов
toolbar = NavigationToolbar2Tk(canvas,plot_frame)
toolbar.update()
toolbar.pack(side=tk.TOP,fill=tk.X)
load_data()
refresh_data()
root.mainloop()

