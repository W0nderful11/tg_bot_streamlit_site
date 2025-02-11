```markdown
# AI-Powered Knowledge Hub with OllamaLLM

## Описание проекта

Этот проект представляет собой AI-чатбота, построенного с использованием **OllamaLLM**, **LangChain**, и **ChromaDB**. Чатбот позволяет загружать документы, выполнять запросы, искать информацию в Интернете и взаимодействовать с базой знаний.

---

## Возможности

- 📂 **Загрузка документов**: Поддержка PDF и TXT файлов.
- 🤖 **Обработка запросов**: Генерация ответов с помощью AI на основе загруженных документов.
- 🌐 **Веб-поиск**: Использование DuckDuckGo для получения контекстной информации из Интернета.
- 📊 **Аналитика и визуализация**: Возможность построения PCA-графиков и анализа данных.
- 🔍 **Извлечение ключевых слов**: Анализ текста с помощью TF-IDF.

---

## Технологии

- **Streamlit**: Интерфейс приложения.
- **LangChain**: Управление взаимодействием с LLM.
- **ChromaDB**: Хранение и поиск данных.
- **OllamaLLM**: Генерация текстов.
- **SentenceTransformers**: Модель векторизации текста.
- **DuckDuckGo Search**: Поиск информации в Интернете.

---

## Установка

1. Клонируйте репозиторий:

   ```bash
   git clone https://github.com/your-repo-url.git
   cd your-repo
   ```

2. Установите зависимости:

   ```bash
   pip install -r requirements.txt
   ```

3. Запустите приложение:

   ```bash
   streamlit run main.py
   ```

---

## Использование

1. Запустите приложение, выбрав `main.py` в Streamlit.
2. Используйте боковую панель для навигации:
   - **Home**: Информация о проекте.
   - **View Documents**: Просмотр и удаление документов.
   - **Add Document**: Загрузка файлов или ввод текста.
   - **Ask Ollama**: Вводите свои запросы и получайте ответы.

3. Для загрузки документа:
   - Выберите файл (PDF или TXT).
   - Убедитесь, что документ не был добавлен ранее.

4. Для выполнения запроса:
   - Введите запрос в разделе `Ask Ollama`.
   - При необходимости включите опцию поиска в Интернете.

---

## Функции

### Загрузка документов
- Загружаемые файлы индексируются в **ChromaDB**.
- Для обработки текста используются TF-IDF и SentenceTransformer.

### Обработка запросов
- Локальные документы анализируются для формирования ответа.
- При включении веб-поиска используется DuckDuckGo.

### Визуализация данных
- Построение PCA-графиков для анализа словарных данных.

---

## Примечания

- Для корректной работы требуется подключение к Интернету.
- База данных создается локально в папке `search_db`.

---

## Примеры использования

### 1. Загрузка документа
Загрузите PDF-файл через вкладку **Add Document**. Документ будет проиндексирован и добавлен в базу знаний.

### 2. Выполнение запроса
Введите вопрос в **Ask Ollama**. AI обработает запрос, используя доступные данные.

---