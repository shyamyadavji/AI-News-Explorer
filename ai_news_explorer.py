"""
AI News Explorer

This script creates a desktop application using PyQt5 to fetch, display, and summarize news articles.
It leverages the NewsAPI for fetching news and a local Hugging Face Transformer model
(facebook/bart-large-cnn by default) for generating AI-powered summaries.

Project Features:
- News Aggregation: Fetches current news headlines and articles using the NewsAPI service.
- Keyword Search: Allows users to search for news articles based on keywords.
- AI Summarization: Provides on-demand AI-generated summaries of article content using a local Transformer model. Configurable for longer output.
- User-Friendly Interface: Offers a graphical interface built with PyQt5 for displaying articles and summaries.
- Article Card Display: Presents news items in a clear, card-based format.
- Direct Article Access: Includes buttons to open the full article in a web browser.
- (Potential) Category Filtering: The backend NewsFetcher supports filtering by NewsAPI categories (e.g., Technology, Business), though the current UI primarily uses keyword search.
- (Note) AI Categorization/Sorting: While this application provides AI summaries, it does not currently perform its own AI-based categorization or sorting beyond what NewsAPI provides. This could be a future extension.
- (Note) Responsive Design: As a PyQt5 desktop application, it resizes but is not inherently 'responsive' in the web-design sense for vastly different devices (like mobile).

Potential Applications:
- Personalized News Browsing Tool (Desktop).
- Foundation for a more advanced news aggregation portal.
- Component within an AI-based content filtering or analysis system.
- Starting point for a data-driven news analysis platform (requiring further development).
- Educational tool for demonstrating API usage and local AI model integration.
"""

import sys
import requests
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QLineEdit, QPushButton, QScrollArea, QProgressBar,
                            QMessageBox, QFrame)
from PyQt5.QtGui import QFont, QIcon, QDesktopServices
from PyQt5.QtCore import Qt, QUrl, QThread, pyqtSignal, QMetaObject, Q_ARG, pyqtSlot

# In ai_news_explorer.py

import os
import sys
import requests
from dotenv import load_dotenv # Add this import
# ... other imports

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
# Get the NewsAPI key from an environment variable
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# The rest of your script follows...
# Make sure there are NO hardcoded API keys left anywhere in this file.

# Attempt to import transformers
try:
    from transformers import pipeline
    LOCAL_AI_ENABLED = True
except ImportError:
    print("WARNING: Local AI features disabled. Install dependencies with:")
    print("pip install transformers torch")
    # Or for TensorFlow: pip install transformers tensorflow
    LOCAL_AI_ENABLED = False
    pipeline = None # Define pipeline as None if import fails

# --- Styling ---
COLORS = {
    "primary": "#2E7D32",    # Darker Green
    "secondary": "#4CAF50",  # Standard Green
    "background": "#F5F5F5", # Very Light Grey
    "surface": "#FFFFFF",    # White
    "text": "#212121",       # Almost Black
    "subtext": "#757575",    # Grey
    "border": "#E0E0E0",    # Light Grey Border
    "success": "#388E3C",    # Dark Green for Success
    "error": "#D32F2F",      # Red for Error
    "accent": "#FFC107"      # Amber/Yellow
}

STYLES = f"""
    /* Global Widget Styles */
    QWidget {{
        background-color: {COLORS['background']};
        color: {COLORS['text']};
        font-family: 'Segoe UI', Arial, sans-serif; /* Added fallbacks */
    }}

    QMainWindow {{
        background-color: {COLORS['background']};
    }}

    /* Input Field */
    QLineEdit {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 10px 12px; /* Adjusted padding */
        font-size: 14px;
        color: {COLORS['text']};
    }}
    QLineEdit:focus {{
        border: 1px solid {COLORS['primary']}; /* Highlight on focus */
    }}

    /* Buttons */
    QPushButton {{
        background-color: {COLORS['primary']};
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        font-size: 14px; /* Explicit font size */
        font-weight: 500; /* Medium weight */
        min-height: 20px; /* Ensure minimum height */
        min-width: 60px; /* Ensure minimum width */
    }}
    QPushButton:hover {{
        background-color: {COLORS['secondary']};
    }}
    QPushButton:pressed {{
        background-color: {COLORS['primary']}; /* Keep primary on press */
    }}
    QPushButton#searchButton {{
        padding: 10px 25px;
    }}
    /* Style for AI summary button (kept active visually) */
    QPushButton[objectName^="summaryButton"] {{
        /* You can add subtle style hints here if needed */
    }}

    /* Scroll Area */
    QScrollArea {{
        border: none;
        background-color: transparent;
    }}
    QWidget#scrollAreaWidgetContents {{
        background-color: {COLORS['background']};
    }}

    /* Article Card Styling */
    QFrame.article-card {{
        background-color: {COLORS['surface']};
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 15px;
        border: 1px solid {COLORS['border']};
    }}

    QLabel#articleTitle {{
        font-size: 16px;
        font-weight: 600;
        color: {COLORS['text']};
        margin-bottom: 5px;
    }}

    QLabel#articleDescription {{
        font-size: 14px;
        color: {COLORS['subtext']};
        line-height: 1.5;
        margin-bottom: 10px;
    }}

    /* Summary Box Styling */
    QFrame.summary-box {{
        background-color: #E8F5E9; /* Light green background */
        border-left: 4px solid {COLORS['success']};
        border-radius: 6px;
        padding: 15px;
        margin-top: 10px;
        margin-bottom: 10px;
        border: 1px solid #C8E6C9; /* Lighter green border */
    }}

    QLabel#summaryText {{
        font-size: 14px;
        color: {COLORS['text']};
        line-height: 1.4;
    }}

    /* Progress Bar */
    QProgressBar {{
        border: 1px solid {COLORS['border']};
        border-radius: 5px;
        text-align: center;
        background-color: {COLORS['surface']};
        height: 10px;
    }}
    QProgressBar::chunk {{
        background-color: {COLORS['primary']};
        border-radius: 5px;
        margin: 0.5px;
    }}
"""

# --- Local AI Service (Updated for Longer Summaries) ---
class LocalAIService:
    """Handles local AI model loading and summarization."""
    def __init__(self):
        self.summarizer = None
        if LOCAL_AI_ENABLED:
            self.init_local_ai()
        else:
            print("Local AI Service disabled due to missing dependencies.")

    def init_local_ai(self):
        """Initializes the Hugging Face summarization pipeline."""
        try:
            print("Initializing local AI summarization model (bart-large-cnn)... This may take a moment.")
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            print("Local AI model loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to initialize local AI model: {str(e)}")
            self.summarizer = None
            # Consider showing a non-blocking notification if GUI is ready

    def generate_summary(self, content):
        """
        Generates a potentially very long summary for the given content.
        NOTE: Requesting summaries significantly longer than the input or standard
              summary lengths (e.g., aiming for 1000+ words with bart-large-cnn)
              can lead to decreased quality, repetition, and incoherence.
              The parameters below push the model towards longer output but
              cannot guarantee high quality at extreme lengths.
        """
        if not self.summarizer:
            return "⚠️ Internal Error: AI model not available when generation requested."

        # Increased minimum content length needed for very long summary attempt
        if not content or not isinstance(content, str) or len(content.strip()) < 150:
             return "⚠️ Content too short or invalid for generating a very long summary."

        try:
            # --- PARAMETERS FOR VERY LONG SUMMARIES (Use with Caution) ---
            target_min_length = 400  # Significantly increased minimum tokens
            target_max_length = 800  # Max tokens (pushing limits, ~600 words avg)
            length_penalty_value = 2.5 # Slightly increased penalty
            no_repeat_ngram_size_value = 3 # Keep ngram repetition prevention

            print(f"Attempting VERY LONG summary generation (min_tokens:{target_min_length}, max_tokens:{target_max_length}, penalty:{length_penalty_value})...")
            print("WARNING: Quality may decrease significantly at extreme lengths.")

            summary_result = self.summarizer(
                content,
                min_length=target_min_length,
                max_length=target_max_length,
                length_penalty=length_penalty_value,
                no_repeat_ngram_size=no_repeat_ngram_size_value,
                do_sample=False
            )

            if summary_result and isinstance(summary_result, list):
                summary_text = summary_result[0]['summary_text']
                # Basic text cleaning (optional)
                summary_text = summary_text.replace(" .", ".").replace(" ,", ",")
                word_count = len(summary_text.split())
                # Estimate token count using the model's tokenizer if available
                token_count = "N/A"
                if hasattr(self.summarizer, 'tokenizer'):
                    token_count = len(self.summarizer.tokenizer.encode(summary_text))

                print(f"Very long summary generated (~{word_count} words, ~{token_count} tokens).")
                # Add a note if the result is much shorter than requested min_length
                if isinstance(token_count, int) and token_count < target_min_length * 0.8: # If less than 80% of min target
                    print(f"NOTE: Generated summary ({token_count} tokens) is shorter than the minimum target ({target_min_length} tokens). Model might have hit limitations.")

                return summary_text
            else:
                 raise ValueError("Summarization model did not return expected output format.")

        except Exception as e:
            print(f"ERROR: Local AI summarization failed during long generation: {str(e)}")
            error_message = f"⚠️ AI Error: Could not generate long summary."
            if "out of memory" in str(e).lower():
                error_message += " (GPU Memory Error - Requested length might be too high)"
            elif "maximum sequence length" in str(e).lower():
                 error_message += " (Input text possibly too long for model's internal limits)"
            elif "too long" in str(e).lower(): # Generic "too long" error
                error_message += " (Requested length might exceed model capacity)"
            else:
                 error_message += f" ({type(e).__name__})"
            return error_message


# --- News Fetching Thread ---
class NewsFetcher(QThread):
    """Fetches news articles in a separate thread."""
    fetched = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, category=None, query=None, parent=None):
        super().__init__(parent)
        self.category = category
        self.query = query
        self.api_key_valid = False # Assume invalid initially
        if not NEWSAPI_KEY or NEWSAPI_KEY == "YOUR_NEWSAPI_KEY" or len(NEWSAPI_KEY) < 30:
            print("ERROR: NewsAPI key is not configured or appears invalid.")
            QMetaObject.invokeMethod(self, "_emit_error", Qt.QueuedConnection,
                                     Q_ARG(str, "ERROR: NewsAPI key is not configured or invalid. Please set it in the script."))
        else:
            self.api_key_valid = True

    @pyqtSlot(str)
    def _emit_error(self, msg):
        self.error.emit(msg)

    def run(self):
        """Executes the news fetching request."""
        if not self.api_key_valid:
            print("NewsFetcher.run() stopped: API key was invalid.")
            return

        try:
            params = {'apiKey': NEWSAPI_KEY, 'pageSize': 20, 'language': 'en'}
            base_url = 'https://newsapi.org/v2/'
            request_desc = ""

            if self.category:
                url = base_url + 'top-headlines'
                params['category'] = self.category
                request_desc = f"top headlines for category '{self.category}'"
            elif self.query:
                url = base_url + 'everything'
                params['q'] = self.query
                params['sortBy'] = 'relevancy'
                request_desc = f"everything for query '{self.query}'"
            else:
                url = base_url + 'top-headlines'
                params['country'] = 'us'
                request_desc = "top headlines for US (default)"

            print(f"Fetching {request_desc}...")
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()

            if data.get('status') != 'ok':
                self._emit_error(f"News API Error ({data.get('code')}): {data.get('message', 'Unknown error')}")
                return

            articles_received = data.get('articles', [])
            if not articles_received:
                 self._emit_error(f"No articles found for {request_desc}. Try different keywords or check NewsAPI coverage.")
                 return

            valid_articles = [
                article for article in articles_received
                if article.get('title') and article.get('url') and (article.get('description') or article.get('content'))
                   and article.get('title') != '[Removed]'
            ]

            if not valid_articles:
                 print("Warning: Articles received, but none passed validation (missing info or '[Removed]').")
                 self._emit_error(f"Received news data for {request_desc}, but no valid articles found.")
                 return

            data['articles'] = valid_articles
            self.fetched.emit(data)

        except requests.exceptions.Timeout:
            self._emit_error("Error: The request to NewsAPI timed out. Check connection.")
        except requests.exceptions.HTTPError as http_err:
             status_code = http_err.response.status_code
             if status_code == 401:
                 self._emit_error("Error 401: Invalid NewsAPI Key. Please check the key in the script.")
             elif status_code == 429:
                 self._emit_error("Error 429: NewsAPI request limit reached. Wait and try again later.")
             elif status_code == 426:
                  self._emit_error("Error 426: API usage restriction. Check your NewsAPI plan/request type.")
             elif status_code == 400:
                  self._emit_error(f"Error 400: Bad Request. Check parameters for {request_desc}. (API Message: {http_err.response.text})")
             else:
                 self._emit_error(f"Error: HTTP Error {status_code} fetching news: {http_err}")
        except requests.exceptions.ConnectionError:
             self._emit_error("Error: Could not connect to NewsAPI. Check internet connection/firewall.")
        except requests.exceptions.RequestException as req_err:
            self._emit_error(f"Error: Network or request error: {req_err}")
        except Exception as e:
            import traceback
            print(f"FATAL: Unexpected error in NewsFetcher:\n{traceback.format_exc()}")
            self._emit_error(f"Error: An unexpected error occurred during news fetching: {str(e)}")


# --- Main Application Window ---
class NewsChatbot(QMainWindow):
    """Main application window for the AI News Explorer."""
    def __init__(self):
        super().__init__()
        self.articles = []
        self.summary_widgets = {} # {article_url: summary_widget}
        self.ai_service = LocalAIService()
        self.news_fetch_thread = None
        self.init_ui()
        self.setStyleSheet(STYLES)

    def init_ui(self):
        """Sets up the user interface components."""
        self.setWindowTitle("AI News Explorer")
        self.setGeometry(100, 100, 1200, 750)
        # self.setWindowIcon(QIcon("path/to/icon.png")) # Optional

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Search Bar
        search_container = QHBoxLayout()
        search_container.setSpacing(10)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search for news topics or keywords...")
        self.search_input.returnPressed.connect(self.process_input)
        search_btn = QPushButton("Search")
        search_btn.setObjectName("searchButton")
        search_btn.clicked.connect(self.process_input)
        search_container.addWidget(self.search_input, 1)
        search_container.addWidget(search_btn)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0) # Indeterminate

        # Articles Area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.content_container_widget = QWidget()
        self.content_container_widget.setObjectName("scrollAreaWidgetContents")
        self.content_layout = QVBoxLayout(self.content_container_widget)
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        self.content_layout.setSpacing(0)
        self.content_layout.addStretch(1) # Push content to top
        scroll_area.setWidget(self.content_container_widget)

        # Assemble Layout
        main_layout.addLayout(search_container)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(scroll_area, 1)

        self.display_welcome_message()

    def display_welcome_message(self):
        """Shows a welcome message."""
        self.clear_content_layout()
        welcome_label = QLabel("Enter a topic or keyword above to search for news articles.")
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet("font-size: 16px; color: #757575; margin-top: 50px;")
        self.content_layout.insertWidget(0, welcome_label)

    def process_input(self):
        """Handles user input from the search bar."""
        query = self.search_input.text().strip()
        if not query:
            QMessageBox.information(self, "Input Required", "Please enter a search term.")
            return
        self.fetch_news(query=query)

    def fetch_news(self, category=None, query=None):
        """Starts the news fetching thread."""
        if self.news_fetch_thread and self.news_fetch_thread.isRunning():
            QMessageBox.warning(self, "Busy", "Already fetching news. Please wait.")
            return

        self.progress_bar.setVisible(True)
        self.clear_content_layout()
        loading_label = QLabel(f"Searching for '{query or category}'...")
        loading_label.setObjectName("loadingLabel")
        loading_label.setAlignment(Qt.AlignCenter)
        loading_label.setStyleSheet("font-size: 16px; color: #757575; margin-top: 50px;")
        self.content_layout.insertWidget(0, loading_label)

        self.news_fetch_thread = NewsFetcher(category=category, query=query, parent=self)
        self.news_fetch_thread.fetched.connect(self.handle_news_results)
        self.news_fetch_thread.error.connect(self.handle_fetch_error)
        self.news_fetch_thread.finished.connect(self.on_fetch_finished)
        self.news_fetch_thread.start()

    # --- Thread Signal Handlers (using invokeMethod for safety) ---
    def handle_news_results(self, data):
        QMetaObject.invokeMethod(self, "_update_ui_with_news", Qt.QueuedConnection, Q_ARG(dict, data))

    def handle_fetch_error(self, message):
        QMetaObject.invokeMethod(self, "_show_error_message", Qt.QueuedConnection, Q_ARG(str, message))

    def on_fetch_finished(self):
        QMetaObject.invokeMethod(self, "_finalize_fetch", Qt.QueuedConnection)

    # --- UI Update Slots (executed in main thread) ---
    @pyqtSlot(dict)
    def _update_ui_with_news(self, data):
        print(f"Received {len(data.get('articles', []))} valid articles.")
        self.articles = data.get('articles', [])
        self.summary_widgets.clear()
        self.display_articles()

    @pyqtSlot(str)
    def _show_error_message(self, message):
        print(f"Fetch Error: {message}")
        self.clear_content_layout()
        QMessageBox.critical(self, "Error Fetching News", message)
        self.display_welcome_message()

    @pyqtSlot()
    def _finalize_fetch(self):
        print("News fetch thread finished.")
        self.progress_bar.setVisible(False)
        if self.news_fetch_thread:
             self.news_fetch_thread = None # Release reference

    def clear_content_layout(self):
        """Removes all widgets from the content layout except the bottom stretch."""
        layout = self.content_layout
        while layout.count() > 1: # Keep last item (stretch)
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def display_articles(self):
        """Creates and displays article cards."""
        self.clear_content_layout()
        if not self.articles:
            no_articles_label = QLabel("No articles found matching your criteria.")
            no_articles_label.setAlignment(Qt.AlignCenter)
            no_articles_label.setStyleSheet("font-size: 16px; color: #757575; margin-top: 50px;")
            self.content_layout.insertWidget(0, no_articles_label)
            return

        print(f"Displaying {len(self.articles)} article cards.")
        for idx, article in enumerate(self.articles):
            card = self.create_article_card(article, idx)
            if card:
                self.content_layout.insertWidget(self.content_layout.count() - 1, card)

    def create_article_card(self, article, index):
        """Creates a single QFrame card for an article."""
        title_text = article.get('title', 'No Title Provided')
        description_text = article.get('description', article.get('content', 'No description available.'))
        url = article.get('url')

        if not url or title_text == '[Removed]': return None

        card = QFrame()
        card.setObjectName(f"articleCard_{index}")
        card.setProperty("class", "article-card")

        layout = QVBoxLayout(card)
        layout.setSpacing(10)

        # Card Content
        title_label = QLabel(f"{title_text}")
        title_label.setObjectName("articleTitle")
        title_label.setWordWrap(True)
        title_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        desc_label = QLabel(description_text)
        desc_label.setObjectName("articleDescription")
        desc_label.setWordWrap(True)
        desc_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        source_name = article.get('source', {}).get('name', 'Unknown Source')
        published_at = article.get('publishedAt', '')
        try:
            from datetime import datetime
            if published_at:
                 # Handle potential timezone formats more robustly
                 if published_at.endswith('Z'):
                     published_at = published_at[:-1] + '+00:00'
                 dt_obj = datetime.fromisoformat(published_at)
                 published_display = dt_obj.strftime("%Y-%m-%d %H:%M")
            else:
                 published_display = "N/A"
        except (ValueError, ImportError, TypeError): # Catch potential errors during parsing
            published_display = published_at[:16].replace('T', ' ') if published_at else "N/A"

        source_label = QLabel(f"Source: {source_name} | Published: {published_display}")
        source_label.setStyleSheet("font-size: 11px; color: #9E9E9E;")

        # Buttons
        btn_container = QHBoxLayout()
        btn_container.setSpacing(10)
        btn_container.addStretch(1)

        read_btn = QPushButton("Read Article")
        read_btn.setToolTip(f"Open article in browser:\n{url}")
        read_btn.clicked.connect(lambda _, u=url: self.open_url(u))

        # --- AI Summary Button (Always Active) ---
        summary_btn = QPushButton("AI Summary")
        summary_btn.setObjectName(f"summaryButton_{index}")
        summary_btn.clicked.connect(lambda _, art_url=url, card_w=card: self.handle_summary_button_click(art_url, card_w))
        if LOCAL_AI_ENABLED and self.ai_service.summarizer:
            summary_btn.setToolTip("Generate a very detailed AI summary (local model)\n(Note: May take time & quality varies)")
        else:
            summary_btn.setToolTip("AI Features Disabled: Click for install instructions.")
        # --- End AI Summary Button Setup ---

        btn_container.addWidget(read_btn)
        btn_container.addWidget(summary_btn)

        # Add to Layout
        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        layout.addWidget(source_label)
        layout.addLayout(btn_container)

        return card

    def open_url(self, url):
        """Opens the given URL in the default web browser."""
        try:
            if url and isinstance(url, str) and url.startswith(('http://', 'https://')):
                print(f"Opening URL: {url}")
                QDesktopServices.openUrl(QUrl(url))
            else:
                raise ValueError("Invalid or missing URL.")
        except Exception as e:
            QMessageBox.warning(self, "Error Opening URL", f"Could not open the article URL.\nError: {str(e)}")

    def find_article_by_url(self, url):
        """Helper to find the full article dictionary by its URL."""
        for article in self.articles:
            if article.get('url') == url:
                return article
        return None

    def handle_summary_button_click(self, article_url, card_widget):
        """Handles clicks on the 'AI Summary' button. Checks for dependencies first."""
        if LOCAL_AI_ENABLED and self.ai_service and self.ai_service.summarizer:
            self.toggle_summary_display(article_url, card_widget)
        else:
            QMessageBox.warning(
                self,
                "AI Features Disabled",
                "Local AI summarization requires additional libraries.\n\n"
                "Please install them using pip:\n"
                "<b>pip install transformers torch</b>\n\n"
                "(Or 'pip install transformers tensorflow' if you prefer TensorFlow)\n\n"
                "Restart the application after installation."
            )
            print("AI Summary button clicked, but dependencies/model are missing.")

    def toggle_summary_display(self, article_url, card_widget):
        """Shows or hides the AI summary display area below the card."""
        if not article_url:
            print("Error: toggle_summary_display called with no URL.")
            return

        if article_url in self.summary_widgets:
            summary_widget_to_remove = self.summary_widgets.pop(article_url)
            if summary_widget_to_remove:
                summary_widget_to_remove.deleteLater()
            print(f"Hiding summary for URL: {article_url}")
            return

        article = self.find_article_by_url(article_url)
        if not article:
            QMessageBox.warning(self, "Error", "Could not find article data for summary.")
            return

        print(f"Requesting summary generation for: {article.get('title')}")
        content_to_summarize = article.get('content', '') or article.get('description', '')
        if content_to_summarize and '[+' in content_to_summarize:
             cutoff_index = content_to_summarize.rfind('[+')
             if cutoff_index != -1:
                 content_to_summarize = content_to_summarize[:cutoff_index].strip()

        if not content_to_summarize:
            QMessageBox.information(self, "Cannot Summarize", "Article has no content or description available for summarization.")
            return

        loading_frame = self.create_summary_widget("⏳ Generating very long AI summary, please wait (this may take a while)...")
        loading_frame.setObjectName(f"loadingSummary_{article_url}")
        self._insert_widget_below_card(card_widget, loading_frame)
        self.summary_widgets[article_url] = loading_frame

        # --- Generate summary ---
        # !!! Consider QThread for this if UI freezes are significant !!!
        summary_text = self.ai_service.generate_summary(content_to_summarize)
        # --- Summary generation finished ---

        current_widget_in_dict = self.summary_widgets.get(article_url)
        if current_widget_in_dict == loading_frame:
            summary_label = loading_frame.findChild(QLabel, "summaryText")
            if summary_label:
                 summary_label.setText(summary_text)
                 if "⚠️" in summary_text:
                     loading_frame.setStyleSheet(f"QFrame.summary-box {{ border-left: 4px solid {COLORS['error']}; background-color: #FFEBEE; border-color: #FFCDD2; }}")
                 else:
                     loading_frame.setStyleSheet(f"QFrame.summary-box {{ border-left: 4px solid {COLORS['success']}; background-color: #E8F5E9; border-color: #C8E6C9; }}")
            else:
                 print("Error: Could not find summary label in loading frame.")
                 loading_frame.deleteLater()
                 new_summary_frame = self.create_summary_widget(summary_text)
                 self._insert_widget_below_card(card_widget, new_summary_frame)
                 self.summary_widgets[article_url] = new_summary_frame

        elif current_widget_in_dict is None:
             print("Summary generation finished, but the target widget was already removed.")
             if loading_frame and not loading_frame.parent():
                 loading_frame.deleteLater()
        else:
             print("Warning: Summary widget state mismatch after generation.")
             if loading_frame and not loading_frame.parent():
                 loading_frame.deleteLater()


    def create_summary_widget(self, text):
        """Creates the QFrame container for the summary text."""
        summary_frame = QFrame()
        summary_frame.setProperty("class", "summary-box")
        summary_frame.setStyleSheet(f"QFrame.summary-box {{ border-left: 4px solid {COLORS['accent']}; background-color: #FFFDE7; border-color: #FFF9C4; }}")

        summary_layout = QVBoxLayout(summary_frame)
        summary_layout.setContentsMargins(15, 10, 15, 10)

        summary_label = QLabel(text)
        summary_label.setObjectName("summaryText")
        summary_label.setWordWrap(True)
        summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        summary_layout.addWidget(summary_label)
        return summary_frame

    def _insert_widget_below_card(self, card_widget, widget_to_insert):
        """ Safely inserts a widget directly below a given card widget."""
        try:
             card_index = self.content_layout.indexOf(card_widget)
             if card_index != -1:
                 self.content_layout.insertWidget(card_index + 1, widget_to_insert)
             else:
                 print(f"Error: Could not find card widget {card_widget.objectName()} in layout.")
                 widget_to_insert.deleteLater()
        except Exception as e:
            print(f"Error inserting widget below card: {e}")
            widget_to_insert.deleteLater()

    def closeEvent(self, event):
        """Ensure threads are stopped cleanly on application exit."""
        print("Close event triggered.")
        if self.news_fetch_thread and self.news_fetch_thread.isRunning():
            print("Attempting to stop news fetch thread...")
            self.news_fetch_thread.quit()
            if not self.news_fetch_thread.wait(3000):
                 print("Warning: News fetch thread did not finish gracefully. Terminating.")
                 self.news_fetch_thread.terminate()
                 self.news_fetch_thread.wait()
            else:
                 print("News fetch thread finished.")
        event.accept()


# --- Entry Point ---
if __name__ == "__main__":
    # Critical check for API Key before starting GUI
    if not NEWSAPI_KEY or NEWSAPI_KEY == "YOUR_NEWSAPI_KEY" or len(NEWSAPI_KEY) < 30:
         temp_app = QApplication.instance() or QApplication(sys.argv)
         QMessageBox.critical(None, "API Key Missing",
                              "FATAL ERROR: NEWSAPI_KEY is missing or invalid!\n\n"
                              "Please set a valid key in the script (obtained from newsapi.org) "
                              "and restart the application.")
         sys.exit(1)

    # Initialize and run the main application
    app = QApplication(sys.argv)
    # app.setStyle("Fusion") # Optional: Force style
    window = NewsChatbot()
    window.show()
    sys.exit(app.exec_())