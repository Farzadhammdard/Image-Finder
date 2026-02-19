from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .features import is_supported_file, load_preview_gray
from .indexer import (
    INDEX_EMBEDDINGS_FILE,
    INDEX_EMBEDDINGS_META_FILE,
    INDEX_FAISS_FILE,
    INDEX_FEATURES_FILE,
    INDEX_HASHES_FILE,
    INDEX_PATHS_FILE,
    build_index,
)
from .search import (
    LoadedIndex,
    SearchContext,
    SearchResult,
    find_similar_in_index_with_context,
    load_runtime_index,
)
from .text_search import TextSearchCache

try:
    from PySide6.QtCore import QObject, QSize, QThread, Qt, Signal
    from PySide6.QtGui import QAction, QDesktopServices, QIcon, QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFrame,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QSpinBox,
        QStatusBar,
        QVBoxLayout,
        QWidget,
    )

    QT_AVAILABLE = True
except Exception:  # pragma: no cover - optional runtime dependency
    QObject = object  # type: ignore[assignment]
    QThread = object  # type: ignore[assignment]
    Signal = lambda *args, **kwargs: None  # type: ignore[assignment]
    QApplication = None  # type: ignore[assignment]
    QMainWindow = object  # type: ignore[assignment]
    QT_AVAILABLE = False


APP_DIR_NAME = "ImageFinder"
ICON_RELATIVE_PATH = Path("assets") / "app_icon.ico"
LOGO_RELATIVE_PATH = Path("assets") / "app_logo.png"

THEMES: dict[str, dict[str, str]] = {
    "Dark": {
        "bg": "#0b0f15",
        "panel": "#151b24",
        "border": "#2d3746",
        "text": "#ebeff5",
        "muted": "#9ba7ba",
        "accent": "#1f8bff",
        "danger": "#ff7d7d",
        "drop": "#101722",
    },
    "Light": {
        "bg": "#f2f5f9",
        "panel": "#ffffff",
        "border": "#d5dde7",
        "text": "#1c2430",
        "muted": "#526070",
        "accent": "#216bdf",
        "danger": "#d43d3d",
        "drop": "#edf3fb",
    },
}


def _truncate(text: str, max_len: int = 120) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def _resource_path(relative_path: Path) -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")) / relative_path
    return Path(__file__).resolve().parent.parent / relative_path


def _load_app_icon():
    for candidate in (ICON_RELATIVE_PATH, LOGO_RELATIVE_PATH):
        icon_path = _resource_path(candidate)
        if not icon_path.exists():
            continue
        icon = QIcon(str(icon_path))
        if icon.isNull():
            continue
        return icon
    return None


def _build_styles(theme_name: str) -> str:
    palette = THEMES.get(theme_name, THEMES["Dark"])
    return f"""
    QMainWindow {{
        background: {palette['bg']};
        color: {palette['text']};
    }}
    QWidget {{
        color: {palette['text']};
        font-size: 13px;
    }}
    QLabel#muted {{
        color: {palette['muted']};
    }}
    QFrame#dropZone {{
        border: 2px dashed {palette['border']};
        border-radius: 10px;
        background: {palette['drop']};
    }}
    QFrame#optionBar, QFrame#actionBar {{
        border: 1px solid {palette['border']};
        border-radius: 10px;
        background: {palette['panel']};
    }}
    QPushButton {{
        background: {palette['accent']};
        color: white;
        border: 1px solid {palette['accent']};
        border-radius: 8px;
        padding: 8px 12px;
        font-weight: 600;
    }}
    QPushButton:disabled {{
        background: #6d7787;
        border-color: #6d7787;
        color: #d7deea;
    }}
    QListWidget {{
        background: {palette['panel']};
        border: 1px solid {palette['border']};
        border-radius: 8px;
        padding: 10px;
    }}
    QListWidget::item {{
        border: 0;
        margin: 0;
        padding: 0;
    }}
    QFrame#resultCard {{
        border: 1px solid {palette['border']};
        border-radius: 12px;
        background: {palette['panel']};
    }}
    QLabel#previewBox {{
        border: 1px solid {palette['border']};
        border-radius: 10px;
        background: {palette['drop']};
    }}
    QLabel#scoreBadge {{
        border: 1px solid {palette['accent']};
        border-radius: 9px;
        padding: 4px 8px;
        color: {palette['accent']};
        font-weight: 700;
    }}
    QProgressBar {{
        border: 1px solid {palette['border']};
        border-radius: 7px;
        text-align: center;
        background: {palette['panel']};
        min-height: 18px;
    }}
    QProgressBar::chunk {{
        background: {palette['accent']};
        border-radius: 7px;
    }}
    QComboBox, QSpinBox {{
        border: 1px solid {palette['border']};
        border-radius: 6px;
        padding: 5px;
        background: {palette['panel']};
    }}
    """


def _get_default_index_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else Path.home() / "AppData" / "Local"
    return root / APP_DIR_NAME / "index_data"


def _index_signature(index_dir: Path) -> tuple[int, ...] | None:
    required = [
        index_dir / INDEX_FEATURES_FILE,
        index_dir / INDEX_HASHES_FILE,
        index_dir / INDEX_PATHS_FILE,
    ]
    if not all(path.exists() for path in required):
        return None

    optional = [
        index_dir / INDEX_EMBEDDINGS_FILE,
        index_dir / INDEX_EMBEDDINGS_META_FILE,
        index_dir / INDEX_FAISS_FILE,
    ]
    signature_files = required + [path for path in optional if path.exists()]
    return tuple(int(path.stat().st_mtime_ns) for path in signature_files)


def _open_file(path: Path) -> None:
    if not path.exists():
        return
    if hasattr(os, "startfile"):
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    QDesktopServices.openUrl(path.as_uri())  # pragma: no cover - non-windows fallback


class SearchWorker(QObject):
    finished = Signal(object, object, float)
    failed = Signal(str)

    def __init__(
        self,
        index_loader,
        query_image: Path,
        top_k: int,
        text_cache: TextSearchCache,
        enable_text_rerank: bool,
        enable_ai_embedding: bool,
        robust_mode: bool,
    ) -> None:
        super().__init__()
        self._index_loader = index_loader
        self._query_image = query_image
        self._top_k = top_k
        self._text_cache = text_cache
        self._enable_text_rerank = enable_text_rerank
        self._enable_ai_embedding = enable_ai_embedding
        self._robust_mode = robust_mode

    def run(self) -> None:
        started = time.perf_counter()
        try:
            index = self._index_loader()
            results, context = find_similar_in_index_with_context(
                index=index,
                query_image=self._query_image,
                top_k=self._top_k,
                text_cache=self._text_cache,
                text_rerank_pool=180 if self._robust_mode else 120,
                enable_text_rerank=self._enable_text_rerank,
                enable_ai_embedding=self._enable_ai_embedding,
                query_variants=6 if self._robust_mode else 3,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        elapsed = time.perf_counter() - started
        self.finished.emit(results, context, elapsed)


class RebuildWorker(QObject):
    progress = Signal(int, int, int, int)
    finished = Signal(object, float)
    failed = Signal(str)

    def __init__(self, index_dir: Path, folders: list[Path]) -> None:
        super().__init__()
        self._index_dir = index_dir
        self._folders = folders

    def _on_progress(self, processed: int, total: int, reused: int, failed: int) -> None:
        self.progress.emit(processed, total, reused, failed)

    def run(self) -> None:
        started = time.perf_counter()
        try:
            self._index_dir.mkdir(parents=True, exist_ok=True)
            stats = build_index(
                folders=self._folders,
                output_dir=self._index_dir,
                progress_callback=self._on_progress,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        elapsed = time.perf_counter() - started
        self.finished.emit(stats, elapsed)


class ImageFinderWindow(QMainWindow):
    def __init__(self, index_dir: Path, top_k: int) -> None:
        super().__init__()
        self.index_dir = index_dir
        self.default_top_k = top_k
        self.compare_folders: list[Path] = [Path.home() / "Desktop"]

        self.is_busy = False
        self.cached_index: LoadedIndex | None = None
        self.cached_signature: tuple[int, ...] | None = None
        self.cache_lock = threading.Lock()
        self.text_cache = TextSearchCache(index_dir=self.index_dir)

        self._thread: QThread | None = None
        self._worker: QObject | None = None

        self.setWindowTitle("Image Finder v-4.0.0")
        app_icon = _load_app_icon()
        if app_icon is not None:
            self.setWindowIcon(app_icon)

        self.resize(1350, 860)
        self.setMinimumSize(1040, 680)
        self.setAcceptDrops(True)
        self._build_ui()
        self._apply_theme("Dark")
        self._refresh_status()

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self.title_label = QLabel("Image Finder | Smart Line Search")
        self.title_label.setStyleSheet("font-size: 24px; font-weight: 700;")
        layout.addWidget(self.title_label)

        self.subtitle_label = QLabel(
            "Find similar image and DXF shapes, even with color changes, blur, and lower quality."
        )
        self.subtitle_label.setObjectName("muted")
        layout.addWidget(self.subtitle_label)

        self.index_label = QLabel("")
        self.index_label.setObjectName("muted")
        layout.addWidget(self.index_label)

        self.source_label = QLabel("")
        self.source_label.setObjectName("muted")
        layout.addWidget(self.source_label)

        action_bar = QFrame()
        action_bar.setObjectName("actionBar")
        action_layout = QHBoxLayout(action_bar)
        action_layout.setContentsMargins(10, 10, 10, 10)
        action_layout.setSpacing(8)
        layout.addWidget(action_bar)

        self.select_button = QPushButton("Select Query Image")
        self.select_button.clicked.connect(self._pick_file)
        action_layout.addWidget(self.select_button)

        self.rebuild_button = QPushButton("Rebuild Index")
        self.rebuild_button.clicked.connect(self._start_rebuild)
        action_layout.addWidget(self.rebuild_button)

        self.pick_folder_button = QPushButton("Choose Compare Folder")
        self.pick_folder_button.clicked.connect(self._choose_compare_folder)
        action_layout.addWidget(self.pick_folder_button)

        self.use_desktop_button = QPushButton("Use Desktop Folder")
        self.use_desktop_button.clicked.connect(self._use_desktop_folder)
        action_layout.addWidget(self.use_desktop_button)

        option_bar = QFrame()
        option_bar.setObjectName("optionBar")
        option_layout = QHBoxLayout(option_bar)
        option_layout.setContentsMargins(10, 10, 10, 10)
        option_layout.setSpacing(10)
        layout.addWidget(option_bar)

        option_layout.addWidget(QLabel("Theme"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.currentTextChanged.connect(self._apply_theme)
        option_layout.addWidget(self.theme_combo)

        option_layout.addWidget(QLabel("Top K"))
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 100)
        self.top_k_spin.setValue(max(1, int(self.default_top_k)))
        option_layout.addWidget(self.top_k_spin)

        self.ai_check = QCheckBox("AI Embedding")
        self.ai_check.setChecked(True)
        option_layout.addWidget(self.ai_check)

        self.ocr_check = QCheckBox("OCR Rerank")
        self.ocr_check.setChecked(True)
        option_layout.addWidget(self.ocr_check)

        self.robust_check = QCheckBox("Robust Mode")
        self.robust_check.setChecked(True)
        option_layout.addWidget(self.robust_check)

        self.drag_check = QCheckBox("Enable Drag && Drop")
        self.drag_check.setChecked(True)
        option_layout.addWidget(self.drag_check)

        option_layout.addStretch(1)

        self.drop_zone = QFrame()
        self.drop_zone.setObjectName("dropZone")
        drop_layout = QVBoxLayout(self.drop_zone)
        drop_layout.setContentsMargins(14, 14, 14, 14)
        drop_layout.setSpacing(4)
        drop_title = QLabel("Drop Zone")
        drop_title.setStyleSheet("font-weight: 700; font-size: 14px;")
        drop_layout.addWidget(drop_title)
        self.drop_hint = QLabel("Drag an image or DXF file and drop it anywhere in this window to start search.")
        self.drop_hint.setObjectName("muted")
        self.drop_hint.setWordWrap(True)
        drop_layout.addWidget(self.drop_hint)
        layout.addWidget(self.drop_zone)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.status_label)

        self.metrics_label = QLabel("")
        self.metrics_label.setObjectName("muted")
        layout.addWidget(self.metrics_label)

        self.results = QListWidget()
        self.results.setSpacing(10)
        self.results.setUniformItemSizes(False)
        self.results.itemDoubleClicked.connect(self._open_result_item)
        layout.addWidget(self.results, stretch=1)

        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        self.statusBar().showMessage("Ready")

        quit_action = QAction("Exit", self)
        quit_action.triggered.connect(self.close)
        self.menuBar().addAction(quit_action)

    def _apply_theme(self, theme_name: str) -> None:
        self.setStyleSheet(_build_styles(theme_name))

    def _set_busy(self, value: bool) -> None:
        self.is_busy = value
        state = not value
        self.select_button.setEnabled(state)
        self.rebuild_button.setEnabled(state)
        self.pick_folder_button.setEnabled(state)
        self.use_desktop_button.setEnabled(state)
        self.theme_combo.setEnabled(state)
        self.top_k_spin.setEnabled(state)
        self.ai_check.setEnabled(state)
        self.ocr_check.setEnabled(state)
        self.robust_check.setEnabled(state)
        self.drag_check.setEnabled(state)
        self.setCursor(Qt.WaitCursor if value else Qt.ArrowCursor)

    def _set_status(self, text: str, error: bool = False) -> None:
        danger = THEMES.get(self.theme_combo.currentText(), THEMES["Dark"])["danger"]
        color = danger if error else THEMES.get(self.theme_combo.currentText(), THEMES["Dark"])["text"]
        self.status_label.setStyleSheet(f"font-weight: 600; color: {color};")
        self.status_label.setText(f"Status: {text}")
        self.statusBar().showMessage(text)

    def _refresh_status(self) -> None:
        self.index_label.setText(f"Index path: {self.index_dir}")
        self.source_label.setText(
            "Compare folders: " + ", ".join(str(path) for path in self.compare_folders if path.exists())
        )
        if _index_signature(self.index_dir) is None:
            self._set_status("Index not found. Click Rebuild Index.", error=True)
            self.metrics_label.setText("First build can take time. Later searches will be faster.")
        else:
            self._set_status("Ready")
            self.metrics_label.setText("Drop or select an image to start matching.")

    def _invalidate_cached_index(self) -> None:
        with self.cache_lock:
            self.cached_index = None
            self.cached_signature = None
        self.text_cache.reset()

    def _get_runtime_index(self) -> LoadedIndex:
        signature = _index_signature(self.index_dir)
        if signature is None:
            raise FileNotFoundError("Index not found. Build the index first.")

        with self.cache_lock:
            if self.cached_index is not None and self.cached_signature == signature:
                return self.cached_index

            loaded = load_runtime_index(self.index_dir)
            self.cached_index = loaded
            self.cached_signature = signature
            return loaded

    def _choose_compare_folder(self) -> None:
        if self.is_busy:
            return
        chosen = QFileDialog.getExistingDirectory(self, "Choose compare folder", str(Path.home()))
        if not chosen:
            return
        folder = Path(chosen)
        if not folder.exists():
            QMessageBox.critical(self, "Image Finder", f"Folder not found:\n{folder}")
            return
        self.compare_folders = [folder]
        self._refresh_status()

    def _use_desktop_folder(self) -> None:
        if self.is_busy:
            return
        desktop = Path.home() / "Desktop"
        self.compare_folders = [desktop]
        self._refresh_status()

    def _pick_file(self) -> None:
        if self.is_busy:
            return
        file_path, _filter = QFileDialog.getOpenFileName(
            self,
            "Select query file",
            "",
            "Supported files (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp *.dxf);;All files (*.*)",
        )
        if not file_path:
            return
        self._start_search(Path(file_path))

    def _start_search(self, image_path: Path) -> None:
        if self.is_busy:
            return
        if _index_signature(self.index_dir) is None:
            QMessageBox.warning(self, "Image Finder", "Index not found. Build the index first.")
            return
        if not image_path.exists():
            QMessageBox.critical(self, "Image Finder", f"File not found:\n{image_path}")
            return
        if not is_supported_file(image_path):
            QMessageBox.critical(self, "Image Finder", f"Unsupported file (image or DXF):\n{image_path}")
            return

        self.results.clear()
        self.progress.setRange(0, 0)
        self._set_busy(True)
        self._set_status("Searching...")
        self.metrics_label.setText("Running robust line-aware matching...")

        thread = QThread(self)
        worker = SearchWorker(
            index_loader=self._get_runtime_index,
            query_image=image_path,
            top_k=int(self.top_k_spin.value()),
            text_cache=self.text_cache,
            enable_text_rerank=bool(self.ocr_check.isChecked()),
            enable_ai_embedding=bool(self.ai_check.isChecked()),
            robust_mode=bool(self.robust_check.isChecked()),
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._on_search_finished)
        worker.failed.connect(self._show_error)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._thread = thread
        self._worker = worker
        thread.start()

    def _on_search_finished(self, results: object, context: object, elapsed: float) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self._set_busy(False)

        cast_results = list(results) if isinstance(results, list) else []
        cast_context = context if isinstance(context, SearchContext) else SearchContext("", False, False)

        self._render_results(cast_results)
        self._set_status(f"Done ({len(cast_results)} results)")

        detail = (
            f"Search: {elapsed:.3f}s | AI: {cast_context.used_ai_embedding} | OCR rerank: {cast_context.used_text_rerank}"
        )
        if cast_context.query_text:
            detail += f" | OCR text: {_truncate(cast_context.query_text, 68)}"
        self.metrics_label.setText(detail)

    def _render_results(self, results: list[SearchResult]) -> None:
        self.results.clear()
        if not results:
            self.results.addItem("No similar files found.")
            return

        for rank, result in enumerate(results, start=1):
            path = Path(result.path)
            item = QListWidgetItem()
            item.setData(Qt.UserRole, str(path))
            item.setToolTip(str(path))
            item.setSizeHint(QSize(980, 220))
            self.results.addItem(item)
            self.results.setItemWidget(item, self._build_result_card(rank=rank, result=result))

    def _load_preview_pixmap(self, path: Path) -> QPixmap:
        pixmap = QPixmap(str(path))
        if not pixmap.isNull():
            return pixmap

        gray = load_preview_gray(path, max_side=920)
        if gray is None:
            return QPixmap()
        if len(gray.shape) != 2:
            return QPixmap()

        raster = np.ascontiguousarray(gray.astype(np.uint8, copy=False))
        height, width = raster.shape
        image = QImage(raster.data, width, height, width, QImage.Format_Grayscale8).copy()
        return QPixmap.fromImage(image)

    def _build_result_card(self, rank: int, result: SearchResult) -> QWidget:
        path = Path(result.path)
        file_name = path.name or str(path)
        extension = path.suffix.lower()
        file_type = "DXF" if extension == ".dxf" else "Image"

        card = QFrame()
        card.setObjectName("resultCard")
        outer = QHBoxLayout(card)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)

        preview = QLabel("Preview unavailable")
        preview.setObjectName("previewBox")
        preview.setAlignment(Qt.AlignCenter)
        preview.setMinimumSize(300, 188)
        preview.setMaximumSize(300, 188)
        pixmap = self._load_preview_pixmap(path)
        if not pixmap.isNull():
            preview.setPixmap(pixmap.scaled(286, 176, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        outer.addWidget(preview)

        info_col = QVBoxLayout()
        info_col.setSpacing(6)
        outer.addLayout(info_col, stretch=1)

        title = QLabel(f"{rank:02d}. {file_name}")
        title.setStyleSheet("font-size: 16px; font-weight: 700;")
        info_col.addWidget(title)

        score_badge = QLabel(f"Similarity: {result.score * 100.0:.2f}%")
        score_badge.setObjectName("scoreBadge")
        info_col.addWidget(score_badge)

        details = QLabel(
            " | ".join(
                [
                    f"type={file_type}",
                    f"vector={result.vector_score:.4f}",
                    f"hash={result.hash_score:.4f}",
                    f"embed={result.embedding_score:.4f}",
                    f"text={result.text_score:.4f}",
                ]
            )
        )
        details.setObjectName("muted")
        details.setWordWrap(True)
        info_col.addWidget(details)

        path_label = QLabel(_truncate(str(path), 220))
        path_label.setToolTip(str(path))
        path_label.setWordWrap(True)
        info_col.addWidget(path_label)

        if result.text_excerpt:
            ocr_label = QLabel(f"OCR: {_truncate(result.text_excerpt, 150)}")
            ocr_label.setObjectName("muted")
            ocr_label.setWordWrap(True)
            info_col.addWidget(ocr_label)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        open_button = QPushButton("Open File")
        open_button.setMinimumWidth(120)
        open_button.clicked.connect(lambda _checked=False, p=path: self._open_result_path(p))
        action_row.addWidget(open_button)
        action_row.addStretch(1)
        info_col.addLayout(action_row)

        info_col.addStretch(1)
        return card

    def _start_rebuild(self) -> None:
        if self.is_busy:
            return
        if not self.compare_folders:
            QMessageBox.critical(self, "Image Finder", "No compare folder selected.")
            return
        for folder in self.compare_folders:
            if not folder.exists():
                QMessageBox.critical(self, "Image Finder", f"Folder not found:\n{folder}")
                return

        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.results.clear()
        self._set_busy(True)
        self._set_status("Building index...")
        self.metrics_label.setText("Scanning files and extracting robust line features...")

        thread = QThread(self)
        worker = RebuildWorker(index_dir=self.index_dir, folders=self.compare_folders)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_rebuild_progress)
        worker.finished.connect(self._on_rebuild_finished)
        worker.failed.connect(self._show_error)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._thread = thread
        self._worker = worker
        thread.start()

    def _on_rebuild_progress(self, processed: int, total: int, reused: int, failed: int) -> None:
        percent = 0 if total <= 0 else int((processed * 100.0) / float(total))
        self.progress.setValue(max(0, min(percent, 100)))
        self.metrics_label.setText(
            f"Processed: {processed}/{total} | Reused: {reused} | Failed: {failed}"
        )

    def _on_rebuild_finished(self, stats: object, elapsed: float) -> None:
        self._invalidate_cached_index()
        self._set_busy(False)
        self.progress.setValue(100)

        payload = stats if isinstance(stats, dict) else {}
        indexed = int(payload.get("indexed", 0))
        reused = int(payload.get("reused", 0))
        failed = int(payload.get("failed", 0))
        embedding_ready = bool(int(payload.get("embedding_ready", 0)))

        self._set_status(f"Index ready ({indexed} files)")
        self.metrics_label.setText(
            f"Done in {elapsed:.1f}s | Indexed: {indexed} | Reused: {reused} | Failed: {failed} | AI: {embedding_ready}"
        )
        QMessageBox.information(
            self,
            "Image Finder",
            (
                "Index build completed.\n"
                f"Indexed files: {indexed}\n"
                f"Reused: {reused}\n"
                f"Failed: {failed}\n"
                f"AI embedding ready: {embedding_ready}\n"
                f"Time: {elapsed:.1f}s"
            ),
        )

    def _open_result_item(self, item: QListWidgetItem) -> None:
        path_raw = item.data(Qt.UserRole)
        if not path_raw:
            return
        self._open_result_path(Path(str(path_raw)))

    def _open_result_path(self, path: Path) -> None:
        if not path.exists():
            QMessageBox.critical(self, "Image Finder", f"File not found:\n{path}")
            return
        _open_file(path)

    def _show_error(self, message: str) -> None:
        self._set_busy(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self._set_status("Operation failed", error=True)
        self.metrics_label.setText(message)
        QMessageBox.critical(self, "Image Finder", message)

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if self.is_busy:
            event.ignore()
            return
        if not self.drag_check.isChecked():
            event.ignore()
            return
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # type: ignore[override]
        if self.is_busy or not self.drag_check.isChecked():
            event.ignore()
            return
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            dropped = Path(url.toLocalFile())
            self._start_search(dropped)
            event.acceptProposedAction()
            return
        event.ignore()


def run_gui(index_dir: Path | None = None, top_k: int = 12) -> int:
    if not QT_AVAILABLE or QApplication is None:
        raise RuntimeError("PySide6 is not installed. Install with: pip install PySide6")

    active_index_dir = index_dir or _get_default_index_dir()
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app_icon = _load_app_icon()
    if app_icon is not None:
        app.setWindowIcon(app_icon)

    window = ImageFinderWindow(index_dir=active_index_dir, top_k=top_k)
    window.show()
    return app.exec()
