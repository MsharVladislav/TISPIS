import os
import numpy as np
import librosa
from flask import Flask, render_template, request, redirect, url_for

# Инициализация приложения
app = Flask(__name__)

# Настройки
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Создаем папку для загрузок, если её нет
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def allowed_file(filename):
    """Проверка расширения файла"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def hz_to_midi(frequencies):
    """Перевод частоты (Гц) в номер ноты MIDI.
    Формула: 12 * log2(fm / 440) + 69
    """
    # Заменяем 0 на NaN, чтобы не было ошибки логарифма
    frequencies = np.where(frequencies == 0, np.nan, frequencies)
    return 12 * np.log2(frequencies / 440.0) + 69

def midi_to_fret(midi_note, string_tune):
    """
    Определяет лад для ноты на конкретной струне.
    midi_note: нота, которую надо сыграть (число)
    string_tune: открытая нота струны (число)
    Возвращает: номер лада или -1, если нота ниже струны
    """
    fret = int(round(midi_note - string_tune))
    if 0 <= fret <= 24: # Гитара обычно имеет до 24 ладов
        return fret
    return -1

# --- 2. ГЛАВНЫЙ АЛГОРИТМ ГЕНЕРАЦИИ ТАБОВ ---

def generate_tabs(filepath):
    """
    Основная функция: Аудио файл -> Текст табулатуры
    """
    # 1. Загрузка аудио
    # duration=60 ограничивает анализ первой минутой (для скорости курсовой)
    y, sr = librosa.load(filepath, sr=22050, mono=True, duration=60)

    # 2. Извлечение основного тона (Pitch Detection)
    # Используем алгоритм pYIN (Probabilistic YIN) - золотой стандарт для монофонии
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, 
        fmin=librosa.note_to_hz('E2'), # Минимальная нота гитары (6 струна)
        fmax=librosa.note_to_hz('E6')  # Максимальная нота (1 струна, 24 лад)
    )

    # Заменяем "неозвученные" участки (тишину) на NaN
    f0[~voiced_flag] = np.nan

    # 3. Перевод частот в MIDI ноты
    midi_notes = hz_to_midi(f0)

    # Строй гитары (Стандартный E): E2, A2, D3, G3, B3, E4
    # В MIDI номерах это: [40, 45, 50, 55, 59, 64]
    guitar_tuning = {
        'e': 64, # 1-я струна (тонкая)
        'B': 59,
        'G': 55,
        'D': 50,
        'A': 45,
        'E': 40  # 6-я струна (толстая)
    }
    
    # Подготовка строк для вывода
    tab_lines = {k: [] for k in guitar_tuning.keys()}
    
    # Чтобы табы не были километровыми, берем каждое 10-е значение анализа
    # (прореживание данных для читаемости)
    step = 5 
    processed_notes = midi_notes[::step]

    last_fret_pos = 0 # Для оптимизации положения руки (можно усложнить)

    # 4. Проход по всем нотам и распределение по струнам
    for note in processed_notes:
        if np.isnan(note):
            # Если тишина - добавляем прочерк во все струны
            for string in tab_lines:
                tab_lines[string].append("-")
            continue

        best_string = None
        min_fret = 100

        # Эвристика: ищем струну, где лад минимален (ближе к голове грифа)
        # Это простейший алгоритм "новичка"
        for string_name, string_pitch in guitar_tuning.items():
            fret = midi_to_fret(note, string_pitch)
            if fret != -1 and fret < min_fret:
                min_fret = fret
                best_string = string_name

        # Записываем результат
        for string in tab_lines:
            if string == best_string:
                tab_lines[string].append(str(min_fret))
            else:
                tab_lines[string].append("-")

    # 5. Сборка финальной строки
    # Форматируем вывод так, чтобы он выглядел красиво
    # e|--0--2--...
    final_output = []
    chunk_size = 60 # Длина одной строки табов (сколько символов в ширину)
    
    total_length = len(tab_lines['E'])
    
    for i in range(0, total_length, chunk_size):
        block = []
        for string_name in ['e', 'B', 'G', 'D', 'A', 'E']:
            # Берем кусок строки
            segment = tab_lines[string_name][i:i+chunk_size]
            # Превращаем список ['-', '5', '-'] в строку "--5-"
            line_str = "".join(segment) 
            # Добавляем название струны в начало
            block.append(f"{string_name}|{line_str}|")
        
        final_output.append("\n".join(block))
        final_output.append("\n") # Пустая строка между блоками

    return "\n".join(final_output)

# --- 3. МАРШРУТЫ САЙТА (ROUTES) ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/converter')
def converter():
    return render_template('upload.html')

@app.route('/process', methods=['POST'])
def process():
    # Проверка наличия файла
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Сохраняем файл
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # ЗАПУСК АЛГОРИТМА
            tabs_text = generate_tabs(filepath)
            
            # Удаляем файл после обработки (чтобы не забивать место)
            os.remove(filepath)
            
            # Отдаем результат
            return render_template('result.html', tabs_text=tabs_text, filename=filename)
        except Exception as e:
            # Если что-то сломалось (например, битый mp3)
            return f"Ошибка обработки: {str(e)}", 500
    
    return "Недопустимый формат файла", 400

# Запуск сервера
if __name__ == '__main__':
    # debug=True позволяет видеть ошибки прямо в браузере (удобно при разработке)
    app.run(debug=True)