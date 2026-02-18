from __future__ import annotations

import os
import re
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .features import is_image_file
from .indexer import INDEX_FEATURES_FILE, INDEX_HASHES_FILE, INDEX_PATHS_FILE, build_index
from .search import (
    LoadedIndex,
    SearchContext,
    SearchResult,
    find_similar_in_index_with_context,
    load_runtime_index,
)
from .text_search import TextSearchCache

try:
    from PIL import Image, ImageTk
except Exception:  # pragma: no cover - optional dependency at runtime
    Image = None  # type: ignore[assignment]
    ImageTk = None  # type: ignore[assignment]

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    DND_ENABLED = True
except Exception:  # pragma: no cover - optional dependency at runtime
    DND_FILES = ""
    TkinterDnD = None  # type: ignore[assignment]
    DND_ENABLED = False


APP_DIR_NAME = "ImageFinder"
APP_VERSION = "3"
AUTHOR_CREDIT = "\u0637\u0631\u0627\u062d\u06cc \u0634\u062f\u0647 \u062a\u0648\u0633\u0637 \u0627\u062d\u0645\u062f \u0641\u0631\u0632\u0627\u062f \u0647\u0645\u062f\u0631\u062f"
HEADER_ICON = "⌕"
CONTROLS_ICON = "⚙"
RESULTS_ICON = "▣"
BUTTON_SELECT_ICON = "▤"
BUTTON_REBUILD_ICON = "↻"
ROOT_BG = "#0d0d0d"
PANEL_BG = "#141414"
PANEL_SOFT = "#1d1d1d"
PANEL_BORDER = "#3f3f3f"
DROP_BG = "#101010"
DROP_BORDER = "#5b5b5b"
TEXT_MAIN = "#f3f3f3"
TEXT_MUTED = "#c4c4c4"
ACCENT = "#2f2f2f"
ACCENT_DARK = "#1f1f1f"
ACCENT_SOFT = "#2a2a2a"
SUCCESS = "#ffffff"
SUCCESS_SOFT = "#2a2a2a"
WARNING = "#ededed"
WARNING_SOFT = "#262626"
ERROR = "#f6f6f6"
ERROR_SOFT = "#2e2e2e"
RESULTS_BG = "#0b0b0b"
RESULTS_CARD_BG = "#161616"
RESULTS_CARD_BORDER = "#444444"
RESULTS_CARD_TEXT = "#f4f4f4"
RESULTS_CARD_SUBTEXT = "#c8c8c8"
RESULTS_THUMB_BG = "#222222"
LOGO_RELATIVE_PATH = Path("assets") / "app_logo.png"

LANGUAGE_LABELS = {"fa": "فارسی", "en": "English"}
LANGUAGE_BY_LABEL = {label: code for code, label in LANGUAGE_LABELS.items()}
DEFAULT_LANGUAGE = "fa"

UI_TEXTS: dict[str, dict[str, str]] = {
    "fa": {
        "app_title": "ایمیج فایندر v3",
        "language_label": "زبان",
        "language_busy_warning": "هنوز یک عملیات در حال اجراست. بعد از پایان عملیات زبان را عوض کنید.",
        "header_title": "جستجوی تصویر CNC",
        "header_subtitle": "عکس مرجع را رها کنید تا شباهت تصویری و متن OCR در ایندکس دسکتاپ بررسی شود.",
        "controls_title": "کنترل جستجو و ایندکس",
        "controls_subtitle": "عکس را انتخاب یا رها کنید، سپس کارت‌های رتبه‌بندی‌شده را ببینید.",
        "drop_enabled": "عکس را اینجا رها کنید\nیا روی انتخاب عکس بزنید",
        "drop_disabled": "درگ‌ودرآپ در دسترس نیست.\nاز انتخاب عکس استفاده کنید",
        "select_image": "انتخاب عکس",
        "rebuild_index": "بازسازی ایندکس دسکتاپ",
        "status_caption": "وضعیت",
        "index_path": "مسیر ایندکس: {index_dir}",
        "results_title": "نتایج جستجو",
        "results_subtitle": "کارت‌ها بر اساس شباهت چیده می‌شوند.",
        "empty_hint": "کارت‌های مشابه اینجا نمایش داده می‌شود.",
        "empty_start": "برای شروع جستجو، عکس را انتخاب یا رها کنید.",
        "empty_searching": "در حال جستجو، لطفا صبر کنید...",
        "empty_rebuilding": "در حال ساخت ایندکس...",
        "empty_ready": "ایندکس آماده است. یک عکس برای جستجو رها کنید.",
        "empty_failed": "عملیات ناموفق بود.",
        "empty_no_results": "تصویر مشابهی پیدا نشد.",
        "status_index_missing": "ایندکس پیدا نشد. روی بازسازی ایندکس دسکتاپ بزنید.",
        "metrics_index_missing": "ساخت اولیه ممکن است زمان‌بر باشد. دفعات بعدی سریع‌تر است.",
        "status_ready": "آماده",
        "metrics_ready_with_text": "عکس مرجع را رها کنید. متن داخل عکس هم بررسی می‌شود.",
        "metrics_ready_visual_only": "عکس مرجع را برای جستجو رها کنید.",
        "status_searching": "در حال جستجو...",
        "status_search_done": "انجام شد ({count} نتیجه)",
        "status_rebuilding": "در حال ساخت ایندکس دسکتاپ...",
        "status_rebuild_done": "ایندکس آماده است ({indexed} تصویر)",
        "status_failed": "عملیات ناموفق بود",
        "err_index_missing": "ایندکس پیدا نشد. ابتدا ایندکس را بسازید.",
        "err_file_missing": "فایل پیدا نشد:\n{path}",
        "err_desktop_missing": "مسیر دسکتاپ پیدا نشد:\n{desktop}",
        "err_unsupported_image": "فرمت فایل تصویر پشتیبانی نمی‌شود:\n{path}",
        "dialog_pick_image": "انتخاب عکس مرجع",
        "filetype_images": "فایل‌های تصویر",
        "filetype_all": "همه فایل‌ها",
        "metrics_rebuild_progress": "پردازش: {processed}/{total}   استفاده مجدد: {reused}   رد شده: {failed}",
        "metrics_scanning": "در حال اسکن فایل‌ها...",
        "metrics_rebuild_done": "پایان در {elapsed:.1f} ثانیه   ایندکس‌شده: {indexed}   استفاده مجدد: {reused}   رد شده: {failed}",
        "metrics_search_with_rerank": "زمان جستجو: {elapsed:.3f} ثانیه   متن OCR: {text}",
        "metrics_search_with_ocr": "زمان جستجو: {elapsed:.3f} ثانیه   متن OCR شناسایی شد: {text}",
        "metrics_search_visual_only": "زمان جستجو: {elapsed:.3f} ثانیه   کوئری: {query}",
        "dialog_rebuild_done": "ساخت ایندکس کامل شد.\nایندکس‌شده: {indexed}\nاستفاده مجدد: {reused}\nرد شده: {failed}\nزمان: {elapsed:.1f} ثانیه",
        "card_rank": "نتیجه {rank}",
        "card_no_preview": "بدون پیش‌نمایش",
        "card_ocr": "متن OCR: {text}",
        "card_open": "باز کردن فایل",
    },
    "en": {
        "app_title": "Image Finder v3",
        "language_label": "Language",
        "language_busy_warning": "An operation is still running. Change language after it finishes.",
        "header_title": "CNC Image Search",
        "header_subtitle": "Drop a reference image to match visual similarity and OCR text inside your desktop index.",
        "controls_title": "Search And Index Controls",
        "controls_subtitle": "Pick or drop an image, then review ranked result cards.",
        "drop_enabled": "Drop image here\nor click Select Image",
        "drop_disabled": "Drag and drop is unavailable.\nUse Select Image.",
        "select_image": "Select Image",
        "rebuild_index": "Rebuild Desktop Index",
        "status_caption": "Status",
        "index_path": "Index path: {index_dir}",
        "results_title": "Search Results",
        "results_subtitle": "Cards are sorted by similarity.",
        "empty_hint": "Similar cards will appear here.",
        "empty_start": "To start searching, choose or drop an image.",
        "empty_searching": "Searching... please wait.",
        "empty_rebuilding": "Building index...",
        "empty_ready": "Index is ready. Drop an image to search.",
        "empty_failed": "Operation failed.",
        "empty_no_results": "No similar images found.",
        "status_index_missing": "Index not found. Click Rebuild Desktop Index.",
        "metrics_index_missing": "First build may take time. Next runs are faster.",
        "status_ready": "Ready",
        "metrics_ready_with_text": "Drop a reference image. OCR text will also be checked.",
        "metrics_ready_visual_only": "Drop a reference image to start searching.",
        "status_searching": "Searching...",
        "status_search_done": "Done ({count} results)",
        "status_rebuilding": "Building desktop index...",
        "status_rebuild_done": "Index is ready ({indexed} images)",
        "status_failed": "Operation failed",
        "err_index_missing": "Index not found. Build the index first.",
        "err_file_missing": "File not found:\n{path}",
        "err_desktop_missing": "Desktop path not found:\n{desktop}",
        "err_unsupported_image": "Unsupported image format:\n{path}",
        "dialog_pick_image": "Select Reference Image",
        "filetype_images": "Image files",
        "filetype_all": "All files",
        "metrics_rebuild_progress": "Processed: {processed}/{total}   Reused: {reused}   Failed: {failed}",
        "metrics_scanning": "Scanning files...",
        "metrics_rebuild_done": "Finished in {elapsed:.1f}s   Indexed: {indexed}   Reused: {reused}   Failed: {failed}",
        "metrics_search_with_rerank": "Search time: {elapsed:.3f}s   OCR text: {text}",
        "metrics_search_with_ocr": "Search time: {elapsed:.3f}s   OCR detected: {text}",
        "metrics_search_visual_only": "Search time: {elapsed:.3f}s   Query: {query}",
        "dialog_rebuild_done": "Index build completed.\nIndexed: {indexed}\nReused: {reused}\nFailed: {failed}\nTime: {elapsed:.1f}s",
        "card_rank": "Result {rank}",
        "card_no_preview": "No preview",
        "card_ocr": "OCR text: {text}",
        "card_open": "Open File",
    },
}


def _truncate(text: str, max_len: int = 120) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def _resource_path(relative_path: Path) -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")) / relative_path
    return Path(__file__).resolve().parent.parent / relative_path


def _get_default_index_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else Path.home() / "AppData" / "Local"
    return root / APP_DIR_NAME / "index_data"


def _parse_dropped_paths(data: str) -> list[Path]:
    paths: list[Path] = []
    for match in re.finditer(r"{([^}]*)}|(\S+)", data):
        raw = match.group(1) or match.group(2)
        if raw:
            paths.append(Path(raw))
    return paths


def _index_signature(index_dir: Path) -> tuple[int, int, int] | None:
    required = [
        index_dir / INDEX_FEATURES_FILE,
        index_dir / INDEX_HASHES_FILE,
        index_dir / INDEX_PATHS_FILE,
    ]
    if not all(path.exists() for path in required):
        return None
    return tuple(int(path.stat().st_mtime_ns) for path in required)  # type: ignore[return-value]


class ImageFinderApp:
    def __init__(self, root: tk.Tk, index_dir: Path, top_k: int) -> None:
        self.root = root
        self.index_dir = index_dir
        self.top_k = top_k
        self.language_code = DEFAULT_LANGUAGE

        self.is_busy = False
        self.cached_index: LoadedIndex | None = None
        self.cached_signature: tuple[int, int, int] | None = None
        self.cache_lock = threading.Lock()
        self.text_cache = TextSearchCache(index_dir=self.index_dir)

        self.thumbnail_refs: list[object] = []
        self.logo_ref: object | None = None
        self.empty_logo_ref: object | None = None
        self.last_results: list[SearchResult] = []
        self.last_canvas_width = 0

        self._build_ui()
        self._refresh_status()

    def _tr(self, key: str, **kwargs: object) -> str:
        table = UI_TEXTS.get(self.language_code, UI_TEXTS["en"])
        fallback = UI_TEXTS["en"]
        template = table.get(key, fallback.get(key, key))
        try:
            return template.format(**kwargs)
        except Exception:
            return template

    def _app_title(self) -> str:
        return self._tr("app_title")

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("Root.TFrame", background=ROOT_BG)
        style.configure(
            "Title.TLabel",
            background=PANEL_BG,
            foreground=TEXT_MAIN,
            font=("Bahnschrift SemiBold", 19),
        )
        style.configure(
            "Subtitle.TLabel",
            background=PANEL_BG,
            foreground=TEXT_MUTED,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Primary.TButton",
            font=("Bahnschrift SemiBold", 10),
            foreground="#f8f8f8",
            background=ACCENT,
            bordercolor=ACCENT,
            focuscolor=ACCENT,
            padding=(12, 8),
        )
        style.map(
            "Primary.TButton",
            background=[("active", ACCENT_DARK), ("disabled", "#4b4b4b")],
            bordercolor=[("active", ACCENT_DARK), ("disabled", "#4b4b4b")],
            foreground=[("disabled", "#9f9f9f")],
        )
        style.configure(
            "Ghost.TButton",
            font=("Segoe UI Semibold", 10),
            foreground=TEXT_MAIN,
            background=PANEL_SOFT,
            bordercolor=PANEL_BORDER,
            focuscolor=PANEL_SOFT,
            padding=(12, 8),
        )
        style.map(
            "Ghost.TButton",
            background=[("active", "#2a2a2a"), ("disabled", "#1f1f1f")],
            foreground=[("disabled", "#9d9d9d")],
        )
        style.configure(
            "Card.TButton",
            font=("Bahnschrift SemiBold", 9),
            foreground="#f8f8f8",
            background=ACCENT,
            bordercolor=ACCENT,
            focuscolor=ACCENT,
            padding=(10, 6),
        )
        style.map(
            "Card.TButton",
            background=[("active", ACCENT_DARK), ("disabled", "#4b4b4b")],
            bordercolor=[("active", ACCENT_DARK), ("disabled", "#4b4b4b")],
        )
        style.configure(
            "Search.Horizontal.TProgressbar",
            troughcolor="#1b1b1b",
            background=ACCENT,
            lightcolor=ACCENT,
            darkcolor=ACCENT,
            thickness=11,
            bordercolor="#3b3b3b",
        )
        style.configure(
            "TCombobox",
            fieldbackground=PANEL_SOFT,
            background=PANEL_SOFT,
            foreground=TEXT_MAIN,
            arrowcolor=TEXT_MAIN,
            bordercolor=PANEL_BORDER,
            lightcolor=PANEL_BORDER,
            darkcolor=PANEL_BORDER,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", PANEL_SOFT)],
            background=[("readonly", PANEL_SOFT)],
            foreground=[("readonly", TEXT_MAIN)],
        )

    def _make_card(self, parent: tk.Widget) -> tk.Frame:
        return tk.Frame(
            parent,
            bg=PANEL_BG,
            highlightthickness=1,
            highlightbackground=PANEL_BORDER,
            highlightcolor=PANEL_BORDER,
            bd=0,
        )

    def _build_ui(self) -> None:
        self.root.title(self._app_title())
        self.root.geometry("1360x820")
        self.root.minsize(1040, 680)
        self.root.configure(bg=ROOT_BG)
        self._configure_styles()

        shell = ttk.Frame(self.root, padding=18, style="Root.TFrame")
        shell.pack(fill=tk.BOTH, expand=True)

        header = self._make_card(shell)
        header.pack(fill=tk.X)

        header_top = tk.Frame(header, bg=PANEL_BG)
        header_top.pack(fill=tk.X, padx=18, pady=(14, 0))

        title = ttk.Label(header_top, text=f"{HEADER_ICON} {self._tr('header_title')}", style="Title.TLabel")
        title.pack(side=tk.LEFT)

        language_shell = tk.Frame(header_top, bg=PANEL_BG)
        language_shell.pack(side=tk.RIGHT, padx=(0, 10), pady=(4, 0))
        language_label = tk.Label(
            language_shell,
            text=self._tr("language_label"),
            bg=PANEL_BG,
            fg=TEXT_MUTED,
            font=("Segoe UI Semibold", 9),
        )
        language_label.pack(side=tk.LEFT, padx=(0, 6))

        self.language_var = tk.StringVar(value=LANGUAGE_LABELS[self.language_code])
        self.language_combo = ttk.Combobox(
            language_shell,
            width=10,
            state="readonly",
            textvariable=self.language_var,
            values=[LANGUAGE_LABELS["fa"], LANGUAGE_LABELS["en"]],
        )
        self.language_combo.pack(side=tk.LEFT)
        self.language_combo.bind("<<ComboboxSelected>>", self._on_language_selected)

        logo = self._load_logo(width=72, height=72)
        if logo is not None:
            logo_label = tk.Label(header_top, image=logo, bg=PANEL_BG, bd=0)
            logo_label.pack(side=tk.RIGHT, padx=(0, 2), pady=(0, 6))
            self.logo_ref = logo

        subtitle = ttk.Label(header, text=self._tr("header_subtitle"), style="Subtitle.TLabel")
        subtitle.pack(anchor=tk.W, padx=18, pady=(0, 6))

        header_credit = tk.Label(
            header,
            text=AUTHOR_CREDIT,
            bg=PANEL_BG,
            fg=TEXT_MUTED,
            font=("Segoe UI", 9),
        )
        header_credit.pack(anchor=tk.W, padx=18, pady=(0, 14))

        content = tk.Frame(shell, bg=ROOT_BG)
        content.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        controls = self._make_card(content)
        controls.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        controls.configure(width=350)
        controls.pack_propagate(False)

        controls_title = tk.Label(
            controls,
            text=f"{CONTROLS_ICON} {self._tr('controls_title')}",
            bg=PANEL_BG,
            fg=TEXT_MAIN,
            font=("Bahnschrift SemiBold", 12),
        )
        controls_title.pack(anchor=tk.W, padx=16, pady=(14, 4))

        controls_subtitle = tk.Label(
            controls,
            text=self._tr("controls_subtitle"),
            bg=PANEL_BG,
            fg=TEXT_MUTED,
            font=("Segoe UI", 9),
        )
        controls_subtitle.pack(anchor=tk.W, padx=16, pady=(0, 4))

        controls_credit = tk.Label(
            controls,
            text=AUTHOR_CREDIT,
            bg=PANEL_BG,
            fg=TEXT_MUTED,
            font=("Segoe UI", 8),
        )
        controls_credit.pack(anchor=tk.W, padx=16, pady=(0, 10))

        drop_key = "drop_enabled" if DND_ENABLED else "drop_disabled"
        self.drop_zone = tk.Label(
            controls,
            text=self._tr(drop_key),
            relief=tk.FLAT,
            bd=0,
            bg=DROP_BG,
            fg=TEXT_MAIN,
            font=("Bahnschrift SemiBold", 13),
            height=5,
            highlightthickness=2,
            highlightbackground=DROP_BORDER,
            highlightcolor=ACCENT,
            justify=tk.CENTER,
            wraplength=300,
        )
        self.drop_zone.pack(fill=tk.X, padx=16, pady=16)
        if DND_ENABLED:
            try:
                self.drop_zone.drop_target_register(DND_FILES)
                self.drop_zone.dnd_bind("<<Drop>>", self._on_drop)
            except Exception:
                self.drop_zone.configure(text=self._tr("drop_disabled"))

        action_row = tk.Frame(controls, bg=PANEL_BG)
        action_row.pack(fill=tk.X, padx=16, pady=(0, 10))

        self.select_button = ttk.Button(
            action_row,
            text=f"{BUTTON_SELECT_ICON} {self._tr('select_image')}",
            style="Primary.TButton",
            command=self._pick_file,
        )
        self.select_button.pack(fill=tk.X)

        self.rebuild_button = ttk.Button(
            action_row,
            text=f"{BUTTON_REBUILD_ICON} {self._tr('rebuild_index')}",
            style="Ghost.TButton",
            command=self._start_rebuild,
        )
        self.rebuild_button.pack(fill=tk.X, pady=(8, 0))

        status_shell = tk.Frame(
            controls,
            bg=PANEL_SOFT,
            highlightthickness=1,
            highlightbackground=PANEL_BORDER,
            highlightcolor=PANEL_BORDER,
        )
        status_shell.pack(fill=tk.X, padx=16, pady=(6, 8))

        self.status_var = tk.StringVar(value="")
        status_caption = tk.Label(
            status_shell,
            text=self._tr("status_caption"),
            bg=PANEL_SOFT,
            fg=TEXT_MUTED,
            font=("Segoe UI", 9),
        )
        status_caption.pack(anchor=tk.W, padx=10, pady=(10, 4))

        self.status_label = tk.Label(
            status_shell,
            textvariable=self.status_var,
            bg=PANEL_SOFT,
            fg=TEXT_MAIN,
            font=("Bahnschrift SemiBold", 10),
            padx=10,
            pady=5,
            anchor="w",
        )
        self.status_label.pack(fill=tk.X, padx=10, pady=(0, 8))

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress = ttk.Progressbar(
            status_shell,
            style="Search.Horizontal.TProgressbar",
            orient=tk.HORIZONTAL,
            mode="determinate",
            variable=self.progress_var,
            maximum=100.0,
        )
        self.progress.pack(fill=tk.X, padx=10, pady=(0, 8))

        self.metrics_var = tk.StringVar(value="")
        metrics = tk.Label(
            status_shell,
            textvariable=self.metrics_var,
            bg=PANEL_SOFT,
            fg=TEXT_MUTED,
            font=("Segoe UI", 9),
            anchor="w",
            justify=tk.LEFT,
            wraplength=304,
        )
        metrics.pack(fill=tk.X, padx=10, pady=(0, 12))

        self.index_var = tk.StringVar(value=self._tr("index_path", index_dir=self.index_dir))
        index_label = tk.Label(
            controls,
            textvariable=self.index_var,
            bg=PANEL_BG,
            fg=TEXT_MUTED,
            font=("Segoe UI", 9),
            anchor="w",
            justify=tk.LEFT,
            wraplength=314,
        )
        index_label.pack(fill=tk.X, padx=16, pady=(2, 14))

        results_shell = self._make_card(content)
        results_shell.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_shell.configure(
            bg=RESULTS_BG,
            highlightbackground=RESULTS_CARD_BORDER,
            highlightcolor=RESULTS_CARD_BORDER,
        )

        results_head = tk.Frame(results_shell, bg=RESULTS_BG)
        results_head.pack(fill=tk.X, padx=14, pady=(12, 6))
        results_header = tk.Label(
            results_head,
            text=f"{RESULTS_ICON} {self._tr('results_title')}",
            bg=RESULTS_BG,
            fg=RESULTS_CARD_TEXT,
            font=("Bahnschrift SemiBold", 12),
        )
        results_header.pack(anchor=tk.W)

        results_info = tk.Label(
            results_head,
            text=self._tr("results_subtitle"),
            bg=RESULTS_BG,
            fg=RESULTS_CARD_SUBTEXT,
            font=("Segoe UI", 9),
        )
        results_info.pack(anchor=tk.W, pady=(2, 0))

        results_credit = tk.Label(
            results_head,
            text=AUTHOR_CREDIT,
            bg=RESULTS_BG,
            fg=RESULTS_CARD_SUBTEXT,
            font=("Segoe UI", 8),
        )
        results_credit.pack(anchor=tk.W, pady=(2, 0))

        canvas_wrap = tk.Frame(results_shell, bg=RESULTS_BG)
        canvas_wrap.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 10))

        self.results_canvas = tk.Canvas(
            canvas_wrap,
            bg=RESULTS_BG,
            highlightthickness=0,
            borderwidth=0,
            relief=tk.FLAT,
        )
        self.results_scroll = ttk.Scrollbar(canvas_wrap, orient=tk.VERTICAL, command=self.results_canvas.yview)
        self.results_canvas.configure(yscrollcommand=self.results_scroll.set)
        self.results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.results_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.results_frame = tk.Frame(self.results_canvas, bg=RESULTS_BG)
        self.results_window = self.results_canvas.create_window((0, 0), window=self.results_frame, anchor="nw")
        self.results_frame.bind("<Configure>", self._on_results_frame_configure)
        self.results_canvas.bind("<Configure>", self._on_results_canvas_configure)

        self._bind_mousewheel(self.results_canvas)
        self._show_empty_results(self._tr("empty_start"))

        footer = tk.Frame(
            shell,
            bg=PANEL_BG,
            highlightthickness=1,
            highlightbackground=PANEL_BORDER,
            highlightcolor=PANEL_BORDER,
        )
        footer.pack(fill=tk.X, pady=(12, 0))

        footer_credit = tk.Label(
            footer,
            text=AUTHOR_CREDIT,
            bg=PANEL_BG,
            fg=TEXT_MUTED,
            font=("Segoe UI", 9),
        )
        footer_credit.pack(side=tk.LEFT, padx=12, pady=8)

        footer_version = tk.Label(
            footer,
            text=f"Version {APP_VERSION}",
            bg=PANEL_BG,
            fg=TEXT_MAIN,
            font=("Bahnschrift SemiBold", 10),
        )
        footer_version.pack(side=tk.RIGHT, padx=12, pady=8)

    def _on_language_selected(self, _event: tk.Event | None = None) -> None:
        selected = self.language_var.get().strip()
        new_language = LANGUAGE_BY_LABEL.get(selected, self.language_code)
        if new_language == self.language_code:
            return
        if self.is_busy:
            self.language_var.set(LANGUAGE_LABELS[self.language_code])
            messagebox.showwarning(self._app_title(), self._tr("language_busy_warning"))
            return

        previous_results = list(self.last_results)
        self.language_code = new_language
        self._rebuild_ui_for_language_change(previous_results)

    def _rebuild_ui_for_language_change(self, previous_results: list[SearchResult]) -> None:
        for child in self.root.winfo_children():
            child.destroy()

        self._build_ui()
        self._refresh_status()
        if previous_results:
            self._render_result_cards(previous_results)

    def _bind_mousewheel(self, widget: tk.Widget) -> None:
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")

        def _on_mousewheel(event: tk.Event) -> None:
            if hasattr(event, "delta") and event.delta:
                self.results_canvas.yview_scroll(int(-event.delta / 120), "units")
            elif hasattr(event, "num") and event.num == 4:
                self.results_canvas.yview_scroll(-1, "units")
            elif hasattr(event, "num") and event.num == 5:
                self.results_canvas.yview_scroll(1, "units")

        widget.bind_all("<MouseWheel>", _on_mousewheel)
        widget.bind_all("<Button-4>", _on_mousewheel)
        widget.bind_all("<Button-5>", _on_mousewheel)

    def _on_results_frame_configure(self, _event: tk.Event) -> None:
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

    def _on_results_canvas_configure(self, event: tk.Event) -> None:
        self.results_canvas.itemconfigure(self.results_window, width=event.width)
        if self.last_results and abs(event.width - self.last_canvas_width) >= 140:
            self.last_canvas_width = event.width
            self._render_result_cards(self.last_results)

    def _set_busy(self, value: bool) -> None:
        self.is_busy = value
        state = tk.DISABLED if value else tk.NORMAL
        self.select_button.configure(state=state)
        self.rebuild_button.configure(state=state)
        self.language_combo.configure(state="disabled" if value else "readonly")
        self.root.configure(cursor="watch" if value else "")

    def _set_status(self, text: str, color: str = TEXT_MUTED) -> None:
        self.status_var.set(text)
        badge_bg = PANEL_SOFT
        if color == SUCCESS:
            badge_bg = SUCCESS_SOFT
        elif color == ERROR:
            badge_bg = ERROR_SOFT
        elif color == ACCENT:
            badge_bg = ACCENT_SOFT
        elif color == WARNING:
            badge_bg = WARNING_SOFT
        self.status_label.configure(fg=color, bg=badge_bg)

    def _refresh_status(self) -> None:
        signature = _index_signature(self.index_dir)
        if signature is None:
            self._set_status(self._tr("status_index_missing"), ERROR)
            self.metrics_var.set(self._tr("metrics_index_missing"))
            return

        self._set_status(self._tr("status_ready"), SUCCESS)
        if self.text_cache.available:
            self.metrics_var.set(self._tr("metrics_ready_with_text"))
        else:
            self.metrics_var.set(self._tr("metrics_ready_visual_only"))

    def _invalidate_cached_index(self) -> None:
        with self.cache_lock:
            self.cached_index = None
            self.cached_signature = None
        self.text_cache.reset()

    def _get_runtime_index(self) -> LoadedIndex:
        signature = _index_signature(self.index_dir)
        if signature is None:
            raise FileNotFoundError(self._tr("err_index_missing"))

        with self.cache_lock:
            if self.cached_index is not None and self.cached_signature == signature:
                return self.cached_index

            loaded = load_runtime_index(self.index_dir)
            self.cached_index = loaded
            self.cached_signature = signature
            return loaded

    def _start_indeterminate_progress(self) -> None:
        self.progress.configure(mode="indeterminate")
        self.progress.start(13)

    def _stop_progress(self) -> None:
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.progress_var.set(0.0)

    def _pick_file(self) -> None:
        if self.is_busy:
            return
        file_path = filedialog.askopenfilename(
            title=self._tr("dialog_pick_image"),
            filetypes=[
                (self._tr("filetype_images"), "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                (self._tr("filetype_all"), "*.*"),
            ],
        )
        if not file_path:
            return
        self._start_search(Path(file_path))

    def _on_drop(self, event: tk.Event) -> None:
        if self.is_busy:
            return
        dropped = _parse_dropped_paths(str(event.data))
        if not dropped:
            return
        self._start_search(dropped[0])

    def _start_search(self, image_path: Path) -> None:
        if _index_signature(self.index_dir) is None:
            messagebox.showwarning(self._app_title(), self._tr("err_index_missing"))
            return
        if not image_path.exists():
            messagebox.showerror(self._app_title(), self._tr("err_file_missing", path=image_path))
            return
        if not is_image_file(image_path):
            messagebox.showerror(self._app_title(), self._tr("err_unsupported_image", path=image_path))
            return

        self._set_busy(True)
        self._start_indeterminate_progress()
        self.metrics_var.set("")
        self._set_status(self._tr("status_searching"), ACCENT)
        self._show_empty_results(self._tr("empty_searching"))

        worker = threading.Thread(target=self._search_worker, args=(image_path,), daemon=True)
        worker.start()

    def _search_worker(self, image_path: Path) -> None:
        started = time.perf_counter()
        try:
            index = self._get_runtime_index()
            results, context = find_similar_in_index_with_context(
                index=index,
                query_image=image_path,
                top_k=self.top_k,
                text_cache=self.text_cache,
            )
            elapsed = time.perf_counter() - started
        except Exception as exc:  # pragma: no cover - UI error handling
            self.root.after(0, lambda: self._show_error(str(exc)))
            return

        self.root.after(0, lambda: self._render_results(image_path, results, context, elapsed))

    def _on_rebuild_progress(self, processed: int, total: int, reused: int, failed: int) -> None:
        percent = 0.0 if total <= 0 else (processed * 100.0 / float(total))

        def update() -> None:
            self.progress_var.set(percent)
            self.metrics_var.set(
                self._tr(
                    "metrics_rebuild_progress",
                    processed=processed,
                    total=total,
                    reused=reused,
                    failed=failed,
                )
            )

        self.root.after(0, update)

    def _start_rebuild(self) -> None:
        if self.is_busy:
            return
        desktop = Path.home() / "Desktop"
        if not desktop.exists():
            messagebox.showerror(self._app_title(), self._tr("err_desktop_missing", desktop=desktop))
            return

        self._set_busy(True)
        self.progress.configure(mode="determinate")
        self.progress_var.set(0.0)
        self._set_status(self._tr("status_rebuilding"), ACCENT)
        self.metrics_var.set(self._tr("metrics_scanning"))
        self._show_empty_results(self._tr("empty_rebuilding"))

        worker = threading.Thread(target=self._rebuild_worker, args=(desktop,), daemon=True)
        worker.start()

    def _rebuild_worker(self, desktop_path: Path) -> None:
        started = time.perf_counter()
        try:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            stats = build_index(
                folders=[desktop_path],
                output_dir=self.index_dir,
                progress_callback=self._on_rebuild_progress,
            )
            elapsed = time.perf_counter() - started
        except Exception as exc:  # pragma: no cover - UI error handling
            self.root.after(0, lambda: self._show_error(str(exc)))
            return

        self.root.after(0, lambda: self._on_rebuild_done(stats, elapsed))

    def _on_rebuild_done(self, stats: dict[str, int], elapsed: float) -> None:
        self._invalidate_cached_index()
        self._set_busy(False)
        self.progress_var.set(100.0)

        indexed_count = stats.get("indexed", 0)
        reused_count = stats.get("reused", 0)
        skipped_count = stats.get("failed", 0)

        self._set_status(self._tr("status_rebuild_done", indexed=indexed_count), SUCCESS)
        self.metrics_var.set(
            self._tr(
                "metrics_rebuild_done",
                elapsed=elapsed,
                indexed=indexed_count,
                reused=reused_count,
                failed=skipped_count,
            )
        )
        self._show_empty_results(self._tr("empty_ready"))
        messagebox.showinfo(
            self._app_title(),
            self._tr(
                "dialog_rebuild_done",
                indexed=indexed_count,
                reused=reused_count,
                failed=skipped_count,
                elapsed=elapsed,
            ),
        )

    def _show_error(self, message: str) -> None:
        self._set_busy(False)
        self._stop_progress()
        self._set_status(self._tr("status_failed"), ERROR)
        self.metrics_var.set(message)
        self._show_empty_results(self._tr("empty_failed"))
        messagebox.showerror(self._app_title(), message)

    def _load_logo(self, width: int = 84, height: int = 84):
        if Image is None or ImageTk is None:
            return None
        logo_path = _resource_path(LOGO_RELATIVE_PATH)
        if not logo_path.exists():
            return None

        try:
            with Image.open(logo_path) as img:
                source = img.convert("RGBA")
                resample_group = getattr(Image, "Resampling", Image)
                source.thumbnail((width, height), resample_group.LANCZOS)
                canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                x = (width - source.width) // 2
                y = (height - source.height) // 2
                canvas.alpha_composite(source, (x, y))
            return ImageTk.PhotoImage(canvas)
        except Exception:
            return None

    def _create_thumbnail(self, path: Path, width: int = 300, height: int = 220):
        if Image is None or ImageTk is None:
            return None
        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
                resample_group = getattr(Image, "Resampling", Image)
                image.thumbnail((width, height), resample_group.LANCZOS)
                canvas = Image.new("RGB", (width, height), color=(34, 34, 34))
                x = (width - image.width) // 2
                y = (height - image.height) // 2
                canvas.paste(image, (x, y))
            return ImageTk.PhotoImage(canvas)
        except Exception:
            return None

    def _show_empty_results(self, message: str) -> None:
        self.last_results = []
        self.thumbnail_refs = []
        self.empty_logo_ref = None
        for child in self.results_frame.winfo_children():
            child.destroy()

        empty_shell = tk.Frame(
            self.results_frame,
            bg=RESULTS_CARD_BG,
            highlightthickness=1,
            highlightbackground=RESULTS_CARD_BORDER,
            highlightcolor=RESULTS_CARD_BORDER,
            padx=18,
            pady=18,
        )
        empty_shell.pack(fill=tk.X, padx=14, pady=14)

        logo = self._load_logo(width=110, height=110)
        if logo is not None:
            logo_label = tk.Label(empty_shell, image=logo, bg=RESULTS_CARD_BG, bd=0)
            logo_label.pack(anchor=tk.W, pady=(0, 8))
            self.empty_logo_ref = logo

        label = tk.Label(
            empty_shell,
            text=message,
            bg=RESULTS_CARD_BG,
            fg=RESULTS_CARD_TEXT,
            font=("Bahnschrift SemiBold", 13),
            anchor="w",
            justify=tk.LEFT,
            wraplength=760,
        )
        label.pack(anchor=tk.W)

        hint = tk.Label(
            empty_shell,
            text=self._tr("empty_hint"),
            bg=RESULTS_CARD_BG,
            fg=RESULTS_CARD_SUBTEXT,
            font=("Segoe UI", 9),
            anchor="w",
            justify=tk.LEFT,
            wraplength=760,
        )
        hint.pack(anchor=tk.W, pady=(4, 0))
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

    def _card_columns(self) -> int:
        width = max(1, self.results_canvas.winfo_width())
        if width >= 1500:
            return 3
        if width >= 980:
            return 2
        return 1

    def _open_file(self, path: Path) -> None:
        if not path.exists():
            messagebox.showerror(self._app_title(), self._tr("err_file_missing", path=path))
            return
        os.startfile(str(path))  # type: ignore[attr-defined]

    def _render_result_cards(self, results: list[SearchResult]) -> None:
        self.last_results = list(results)
        self.thumbnail_refs = []
        self.empty_logo_ref = None

        for child in self.results_frame.winfo_children():
            child.destroy()

        if not results:
            self._show_empty_results(self._tr("empty_no_results"))
            return

        columns = self._card_columns()
        for col in range(3):
            self.results_frame.grid_columnconfigure(col, weight=0, uniform="")
        for col in range(columns):
            self.results_frame.grid_columnconfigure(col, weight=1, uniform="result-col")

        available_width = max(620, self.results_canvas.winfo_width())
        card_wraplength = max(240, int(available_width / max(columns, 1)) - 88)
        thumb_width = max(250, min(360, int(available_width / max(columns, 1)) - 72))
        thumb_height = int(thumb_width * 0.78)
        self.last_canvas_width = self.results_canvas.winfo_width()

        def _bind_open(widget: tk.Widget, file_path: Path) -> None:
            widget.bind("<Double-Button-1>", lambda _event, p=file_path: self._open_file(p))

        for idx, result in enumerate(results, start=1):
            row = (idx - 1) // columns
            col = (idx - 1) % columns
            result_path = Path(result.path)
            file_name = result_path.name if result_path.name else str(result_path)

            card = tk.Frame(
                self.results_frame,
                bg=RESULTS_CARD_BG,
                highlightthickness=1,
                highlightbackground=RESULTS_CARD_BORDER,
                highlightcolor=RESULTS_CARD_BORDER,
                padx=12,
                pady=12,
            )
            card.grid(row=row, column=col, sticky="nsew", padx=8, pady=8)
            card.grid_columnconfigure(0, weight=1)
            _bind_open(card, result_path)

            rank_label = tk.Label(
                card,
                text=self._tr("card_rank", rank=idx),
                bg="#242424",
                fg=RESULTS_CARD_TEXT,
                font=("Bahnschrift SemiBold", 10),
                padx=9,
                pady=4,
            )
            rank_label.grid(row=0, column=0, sticky="w", pady=(0, 8))
            _bind_open(rank_label, result_path)

            thumb = self._create_thumbnail(result_path, width=thumb_width, height=thumb_height)
            thumb_label = tk.Label(
                card,
                bg=RESULTS_THUMB_BG,
                fg="#bcbcbc",
                text=self._tr("card_no_preview"),
                width=40,
                height=11,
                highlightthickness=1,
                highlightbackground=PANEL_BORDER,
            )
            if thumb is not None:
                thumb_label.configure(image=thumb, text="")
                self.thumbnail_refs.append(thumb)
            thumb_label.grid(row=1, column=0, sticky="ew")
            _bind_open(thumb_label, result_path)

            name_label = tk.Label(
                card,
                text=_truncate(file_name, 80),
                bg=RESULTS_CARD_BG,
                fg=RESULTS_CARD_TEXT,
                font=("Segoe UI Semibold", 10),
                anchor="w",
                justify=tk.LEFT,
                wraplength=card_wraplength,
            )
            name_label.grid(row=2, column=0, sticky="ew", pady=(8, 2))
            _bind_open(name_label, result_path)

            if result.text_excerpt:
                text_label = tk.Label(
                    card,
                    text=self._tr("card_ocr", text=_truncate(result.text_excerpt, 96)),
                    bg="#202020",
                    fg="#f1f1f1",
                    font=("Segoe UI", 9),
                    anchor="w",
                    justify=tk.LEFT,
                    wraplength=card_wraplength,
                    padx=7,
                    pady=5,
                )
                text_label.grid(row=3, column=0, sticky="ew", pady=(3, 6))
                _bind_open(text_label, result_path)

            path_label = tk.Label(
                card,
                text=_truncate(result.path, 140),
                bg=RESULTS_CARD_BG,
                fg=RESULTS_CARD_SUBTEXT,
                font=("Segoe UI", 8),
                anchor="w",
                justify=tk.LEFT,
                wraplength=card_wraplength,
            )
            path_label.grid(row=4, column=0, sticky="ew", pady=(0, 8))
            _bind_open(path_label, result_path)

            open_button = ttk.Button(
                card,
                text=self._tr("card_open"),
                style="Card.TButton",
                command=lambda p=result_path: self._open_file(p),
            )
            open_button.grid(row=5, column=0, sticky="ew")

        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

    def _render_results(
        self,
        image_path: Path,
        results: list[SearchResult],
        context: SearchContext,
        elapsed: float,
    ) -> None:
        self._set_busy(False)
        self._stop_progress()
        self._render_result_cards(results)

        self._set_status(self._tr("status_search_done", count=len(results)), SUCCESS)
        if context.query_text and context.used_text_rerank:
            self.metrics_var.set(
                self._tr("metrics_search_with_rerank", elapsed=elapsed, text=_truncate(context.query_text, 64))
            )
        elif context.query_text:
            self.metrics_var.set(
                self._tr("metrics_search_with_ocr", elapsed=elapsed, text=_truncate(context.query_text, 64))
            )
        else:
            self.metrics_var.set(self._tr("metrics_search_visual_only", elapsed=elapsed, query=image_path))


def run_gui(index_dir: Path | None = None, top_k: int = 12) -> int:
    active_index_dir = index_dir or _get_default_index_dir()
    root = TkinterDnD.Tk() if DND_ENABLED and TkinterDnD is not None else tk.Tk()
    ImageFinderApp(root, index_dir=active_index_dir, top_k=top_k)
    root.mainloop()
    return 0

