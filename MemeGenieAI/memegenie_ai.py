"""Meme Genie AI Studio: meme editing, local models and image generation.

Copyright (c) 2026 @kaotickj. Bundled model catalog with explicit install and offline-bundle support.
Third-party model weights are installed with the model manager.
No network or API key is used for AI image generation.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont, ImageOps, PngImagePlugin

from PyQt6.QtCore import Qt, QProcess, QSettings, QTimer, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QDesktopServices, QIcon, QImage, QPalette, QPixmap, QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGraphicsScene, QGraphicsView, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QRadioButton, QButtonGroup, QScrollArea,
    QSpinBox, QSplitter, QTabWidget, QTextEdit, QVBoxLayout, QWidget, QListWidget, QListWidgetItem,
)

from sd_backend import GenerationRequest, build_sd_cli_arguments
from model_manager import (
    BASE_DIR, app_data_dir, model_dir, engine_dir, models, resolve_model,
    download_model, verify_model, install_engine, locate_engine, create_offline_bundle,
    retrieve_model_license,
)


class InstallWorker(QThread):
    """QThread for network downloads and multi-gigabyte SHA-256 checks."""
    progress_info = pyqtSignal(int, int, str)
    succeeded = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, task: str, model_id: str = "", engine_flavor: str = "vulkan",
                 bundle_path: str = ""):
        super().__init__()
        self.task, self.model_id, self.engine_flavor = task, model_id, engine_flavor
        self.bundle_path = bundle_path
        self._stopping = False

    def cancel(self):
        self._stopping = True

    def run(self):
        def announce(done, total, description):
            self.progress_info.emit(int(done), int(total or 0), description)
        try:
            if self.task == "model":
                result = download_model(resolve_model(self.model_id), progress=announce,
                                        should_cancel=lambda: self._stopping)
                try:
                    retrieve_model_license()
                except Exception as exc:
                    self.progress_info.emit(0, 0, "Model verified, but license retrieval failed: " + str(exc))
                self.succeeded.emit(str(result))
            elif self.task == "verify":
                info = resolve_model(self.model_id)
                folders = [model_dir(), BASE_DIR / "models",
                           Path(sys.executable).resolve().parent / "models"]
                verified_path = None
                for folder in folders:
                    if info.installed_path(folder).is_file() and verify_model(
                            info, folder, progress=announce,
                            should_cancel=lambda: self._stopping):
                        verified_path = info.installed_path(folder)
                        break
                if verified_path is None:
                    raise ValueError("Model is missing or has an invalid SHA-256")
                self.succeeded.emit(str(verified_path))
            elif self.task == "engine":
                result = install_engine(self.engine_flavor, progress=announce,
                                        should_cancel=lambda: self._stopping)
                self.succeeded.emit(str(result))
            elif self.task == "bundle":
                result = create_offline_bundle(Path(self.bundle_path),
                    selected_ids=(self.model_id,), progress=announce)
                self.succeeded.emit(str(result))
            else:
                raise ValueError("Unknown installation operation")
        except Exception as exc:
            self.failed.emit(str(exc))


class ImageView(QGraphicsView):
    """Scale preview to viewport while retaining the source image resolution."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setBackgroundBrush(QColor("#1b222b"))
        self.setMinimumSize(350, 320)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def show_pil(self, image: Image.Image):
        image = image.convert("RGBA")
        rgba = image.tobytes()
        qt_img = QImage(rgba, image.width, image.height, image.width * 4,
                        QImage.Format.Format_RGBA8888).copy()
        pixmap = QPixmap.fromImage(qt_img)
        self.scene().clear()
        item = self.scene().addPixmap(pixmap)
        self.scene().setSceneRect(item.boundingRect())
        self.fitInView(item, Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        items = self.scene().items()
        if items:
            self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class MemeGeneratorApp(QMainWindow):
    APP_VERSION = "1.2.1"

    def __init__(self):
        super().__init__()
        self.base_image: Image.Image | None = None
        self.base_image_path: str | None = None
        self.selected_font: str | None = None
        self.text_color = (255, 255, 255, 255)
        self.text_shadow = True
        self.ai_process: QProcess | None = None
        self.install_worker: InstallWorker | None = None
        self.ai_canceled = False
        self.ai_output: Path | None = None
        self.ai_log_path: Path | None = None
        self.ai_last_command = ""
        self.ai_generation_metadata: dict[str, str] = {}
        self.settings = QSettings("Kaotick Jay", "Meme Genie AI")

        self.setWindowTitle("Meme Genie | AI Studio 1.2.1")
        app_icon = Path(__file__).with_name("memegenie.ico")
        if app_icon.is_file():
            self.setWindowIcon(QIcon(str(app_icon)))
        self.resize(1250, 830)
        self.build_ui()
        self.restore_settings()
        self.on_model_choice()
        self.refresh_gallery()
        self.statusBar().showMessage("Ready. Load an image or generate one locally.")

    def build_ui(self):
        file_menu = self.menuBar().addMenu("File")
        self.add_action(file_menu, "Open Image…", self.choose_base_image)
        self.add_action(file_menu, "Save Edited Image…", self.generate_meme)
        self.add_action(file_menu, "Save Original / AI Base Image…", self.save_base_image)
        file_menu.addSeparator()
        self.add_action(file_menu, "Exit", self.close)

        edit_menu = self.menuBar().addMenu("Edit")
        self.add_action(edit_menu, "Select Text Color…", self.choose_text_color)
        self.add_action(edit_menu, "Clear Text", lambda: self.text_input.clear())
        self.shadow_action = self.add_action(edit_menu, "Text Shadow", self.toggle_text_shadow)
        self.shadow_action.setCheckable(True)
        self.shadow_action.setChecked(True)

        help_menu = self.menuBar().addMenu("Help")
        self.add_action(help_menu, "Instructions", self.show_instructions)
        self.add_action(help_menu, "About", self.show_about)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(390)
        self.tabs.addTab(self.build_meme_tab(), "Meme Editor")
        self.tabs.addTab(self.build_ai_tab(), "AI Image Generation")
        self.tabs.addTab(self.build_gallery_tab(), "Image Library")
        splitter.addWidget(self.tabs)

        preview_panel = QWidget()
        pv_layout = QVBoxLayout(preview_panel)
        pv_layout.addWidget(QLabel("<b>Image Preview</b>  •  Source resolution is preserved on export"))
        self.image_preview = ImageView()
        pv_layout.addWidget(self.image_preview, 1)
        self.preview_size = QLabel("No image loaded")
        self.preview_size.setAlignment(Qt.AlignmentFlag.AlignRight)
        pv_layout.addWidget(self.preview_size)
        splitter.addWidget(preview_panel)
        splitter.setSizes([450, 790])
        self.setCentralWidget(splitter)

        self.setStyleSheet("""
            QMainWindow { background: #131a22; }
            QWidget { font-size: 13px; }
            QTabWidget::pane { border: 1px solid #364455; }
            QGroupBox { border: 1px solid #364455; border-radius: 6px;
                        margin-top: 10px; padding: 8px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { padding: 7px 9px; }
            QPlainTextEdit, QTextEdit { border-radius: 5px; }
        """)

    @staticmethod
    def add_action(menu, caption, handler):
        action = QAction(caption, menu)
        action.triggered.connect(handler)
        menu.addAction(action)
        return action

    @staticmethod
    def scrolling_tab(inner_layout: QVBoxLayout) -> QScrollArea:
        inner_layout.addStretch(1)
        content = QWidget()
        content.setLayout(inner_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(content)
        return scroll

    def build_meme_tab(self):
        layout = QVBoxLayout()
        open_btn = QPushButton("Choose Base Image…")
        open_btn.clicked.connect(self.choose_base_image)
        layout.addWidget(open_btn)

        layout.addWidget(QLabel("Meme / caption text (leave empty for image-only export):"))
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter meme text or fetch a quote…")
        self.text_input.setFixedHeight(130)
        self.text_input.textChanged.connect(self.update_preview)
        layout.addWidget(self.text_input)

        quote_btn = QPushButton("Fetch Quote (requires internet)")
        quote_btn.clicked.connect(self.fetch_quote)
        layout.addWidget(quote_btn)

        form = QFormLayout()
        self.font_selector = QComboBox()
        self.populate_font_selector()
        self.font_selector.currentTextChanged.connect(self.on_font_select)
        form.addRow("Font:", self.font_selector)
        self.font_size = QSpinBox()
        self.font_size.setRange(10, 250)
        self.font_size.setValue(60)
        self.font_size.valueChanged.connect(self.update_preview)
        form.addRow("Font size:", self.font_size)

        color_btn = QPushButton("Select Color…")
        color_btn.clicked.connect(self.choose_text_color)
        form.addRow("Text color:", color_btn)
        layout.addLayout(form)

        alignment = QHBoxLayout()
        self.align_group = QButtonGroup(self)
        for label in ("Left", "Center", "Right"):
            btn = QRadioButton(label)
            self.align_group.addButton(btn)
            btn.setChecked(label == "Center")
            btn.toggled.connect(self.update_preview)
            alignment.addWidget(btn)
        layout.addLayout(alignment)

        self.text_shadow_checkbox = QCheckBox("Text shadow")
        self.text_shadow_checkbox.setChecked(True)
        self.text_shadow_checkbox.toggled.connect(self.shadow_action.setChecked)
        self.text_shadow_checkbox.toggled.connect(self.update_preview)
        self.shadow_action.toggled.connect(self.text_shadow_checkbox.setChecked)
        layout.addWidget(self.text_shadow_checkbox)

        self.attribution_checkbox = QCheckBox("Add Meme Genie attribution to exported image")
        self.attribution_checkbox.setChecked(False)
        self.attribution_checkbox.toggled.connect(self.update_preview)
        layout.addWidget(self.attribution_checkbox)

        save_btn = QPushButton("Save Edited Image…")
        save_btn.clicked.connect(self.generate_meme)
        layout.addWidget(save_btn)
        export_presets = QHBoxLayout()
        widescreen_export = QPushButton("Export YouTube 1920 × 1080…")
        widescreen_export.clicked.connect(lambda: self.export_sized(1920, 1080))
        vertical_export = QPushButton("Export Short 1080 × 1920…")
        vertical_export.clicked.connect(lambda: self.export_sized(1080, 1920))
        export_presets.addWidget(widescreen_export)
        export_presets.addWidget(vertical_export)
        layout.addLayout(export_presets)
        return self.scrolling_tab(layout)

    def build_ai_tab(self):
        layout = QVBoxLayout()
        intro = QLabel("AI images are generated locally. Use Install Model to download a verified "
                       "Stable Diffusion 1.5 checkpoint. On Windows, Install Engine obtains "
                       "the official Vulkan build. Once installed, generation is fully offline.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        paths_group = QGroupBox("Local engine and installed models")
        paths_form = QFormLayout(paths_group)
        self.sd_executable = QLineEdit()
        paths_form.addRow("sd-cli:", self.with_browse(self.sd_executable, self.browse_engine))
        self.sd_model = QLineEdit()
        paths_form.addRow("Model file:", self.with_browse(self.sd_model, self.browse_model))
        self.model_choice = QComboBox()
        for model in models():
            self.model_choice.addItem(model.display_name, model.id)
        self.model_choice.currentIndexChanged.connect(self.on_model_choice)
        paths_form.addRow("Model catalog:", self.model_choice)
        self.model_license = QLabel("OpenRAIL-M use restrictions apply. Review the model license before installing.")
        self.model_license.setWordWrap(True)
        paths_form.addRow("License:", self.model_license)
        install_line = QHBoxLayout()
        self.model_install_btn = QPushButton("Install Selected Model")
        self.model_install_btn.clicked.connect(self.install_selected_model)
        self.model_verify_btn = QPushButton("Verify SHA-256")
        self.model_verify_btn.clicked.connect(self.verify_selected_model)
        install_line.addWidget(self.model_install_btn)
        install_line.addWidget(self.model_verify_btn)
        paths_form.addRow(install_line)
        engine_line = QHBoxLayout()
        self.engine_install_btn = QPushButton("Install Windows Vulkan Engine")
        self.engine_install_btn.clicked.connect(self.install_vulkan_engine)
        self.engine_install_btn.setEnabled(sys.platform == "win32")
        engine_line.addWidget(self.engine_install_btn)
        paths_form.addRow(engine_line)
        self.install_progress = QProgressBar()
        self.install_progress.setRange(0, 100)
        self.install_progress.setValue(0)
        paths_form.addRow("Install progress:", self.install_progress)
        self.install_status = QLabel("Use model download once; already installed weights work offline.")
        self.install_status.setWordWrap(True)
        paths_form.addRow("Installer:", self.install_status)
        self.install_cancel_btn = QPushButton("Cancel Download (keep partial for resume)")
        self.install_cancel_btn.clicked.connect(self.cancel_install)
        self.install_cancel_btn.setEnabled(False)
        paths_form.addRow(self.install_cancel_btn)
        self.offline_bundle_btn = QPushButton("Create Offline Bundle with Verified Model + Engine…")
        self.offline_bundle_btn.clicked.connect(self.make_offline_bundle)
        paths_form.addRow(self.offline_bundle_btn)
        self.output_dir = QLineEdit()
        paths_form.addRow("Image folder:", self.with_browse(self.output_dir, self.browse_output_dir))
        layout.addWidget(paths_group)

        prompt_group = QGroupBox("Generation prompts")
        prompt_form = QVBoxLayout(prompt_group)
        prompt_form.addWidget(QLabel("Describe the image you want:"))
        self.ai_prompt = QTextEdit()
        self.ai_prompt.setPlaceholderText("A dramatic Fallout-inspired wasteland skyline, moody studio lighting, no lettering…")
        self.ai_prompt.setFixedHeight(115)
        prompt_form.addWidget(self.ai_prompt)
        prompt_form.addWidget(QLabel("Negative prompt (optional):"))
        self.ai_negative = QTextEdit()
        self.ai_negative.setPlaceholderText("text, letters, watermark, blurry, low quality")
        self.ai_negative.setFixedHeight(65)
        prompt_form.addWidget(self.ai_negative)
        layout.addWidget(prompt_group)
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Prompt preset:"))
        self.prompt_preset = QComboBox()
        self.prompt_preset.addItem("Custom prompt", "custom")
        self.prompt_preset.addItem("Fallout-inspired wasteland backdrop", "wasteland")
        self.prompt_preset.addItem("Chicago blues album art", "blues")
        self.prompt_preset.addItem("Cinematic gaming thumbnail", "thumbnail")
        self.prompt_preset.addItem("Character portrait for short", "portrait")
        self.prompt_preset.addItem("Clean meme background", "meme")
        self.prompt_preset.currentIndexChanged.connect(self.apply_prompt_preset)
        preset_row.addWidget(self.prompt_preset, 1)
        layout.addLayout(preset_row)

        options = QGroupBox("Generation settings")
        opt_form = QFormLayout(options)
        self.dim_preset = QComboBox()
        presets = [("Square - 512 × 512", (512, 512)),
                   ("Square - 768 × 768", (768, 768)),
                   ("Landscape - 768 × 512", (768, 512)),
                   ("16:9 - 1024 × 576", (1024, 576)),
                   ("Square - 1024 × 1024", (1024, 1024)),
                   ("Vertical - 576 × 1024", (576, 1024)),
                   ("Landscape - 1152 × 640", (1152, 640))]
        for caption, dimensions in presets:
            self.dim_preset.addItem(caption, dimensions)
        opt_form.addRow("Resolution:", self.dim_preset)

        self.ai_steps = QSpinBox()
        self.ai_steps.setRange(1, 150)
        self.ai_steps.setValue(20)
        opt_form.addRow("Sampling steps:", self.ai_steps)
        self.ai_cfg = QDoubleSpinBox()
        self.ai_cfg.setRange(0, 30)
        self.ai_cfg.setDecimals(1)
        self.ai_cfg.setSingleStep(.5)
        self.ai_cfg.setValue(7.0)
        opt_form.addRow("CFG guidance:", self.ai_cfg)

        self.ai_seed = QSpinBox()
        self.ai_seed.setRange(-1, 2147483647)
        self.ai_seed.setSpecialValueText("Random (-1)")
        self.ai_seed.setValue(-1)
        opt_form.addRow("Seed:", self.ai_seed)
        self.ai_sampler = QComboBox()
        for sampler in ("euler", "euler_a", "heun", "dpm2", "dpm++2m"):
            self.ai_sampler.addItem(sampler)
        opt_form.addRow("Sampler:", self.ai_sampler)
        self.ai_backend = QComboBox()
        self.ai_backend.addItem("Auto detect", "auto")
        self.ai_backend.addItem("Vulkan GPU (AMD)", "vulkan0")
        self.ai_backend.addItem("CPU", "cpu")
        opt_form.addRow("Compute backend:", self.ai_backend)
        self.ai_compat_mode = QCheckBox("Vulkan compatibility: disable Flash Attention and direct convolution")
        self.ai_compat_mode.setToolTip("Use if Vulkan crashes during generation. May run slower. Requires a recent sd-cli build.")
        opt_form.addRow("Compatibility:", self.ai_compat_mode)
        layout.addWidget(options)

        button_line = QHBoxLayout()
        self.ai_generate_btn = QPushButton("Generate Image")
        self.ai_generate_btn.clicked.connect(self.generate_ai_image)
        button_line.addWidget(self.ai_generate_btn)
        self.ai_cancel_btn = QPushButton("Cancel")
        self.ai_cancel_btn.setEnabled(False)
        self.ai_cancel_btn.clicked.connect(self.cancel_ai)
        button_line.addWidget(self.ai_cancel_btn)
        layout.addLayout(button_line)
        self.ai_progress = QProgressBar()
        self.ai_progress.setRange(0, 100)
        self.ai_progress.setValue(0)
        layout.addWidget(self.ai_progress)
        self.ai_status = QLabel("Download a model and the engine above, or select pre-existing installations.")
        self.ai_status.setWordWrap(True)
        layout.addWidget(self.ai_status)
        log_buttons = QHBoxLayout()
        copy_cmd = QPushButton("Copy engine command")
        copy_cmd.clicked.connect(self.copy_engine_command)
        log_buttons.addWidget(copy_cmd)
        open_logs = QPushButton("Open diagnostic log folder")
        open_logs.clicked.connect(self.open_ai_log_folder)
        log_buttons.addWidget(open_logs)
        layout.addLayout(log_buttons)
        self.ai_log = QPlainTextEdit()
        self.ai_log.setReadOnly(True)
        self.ai_log.setMaximumBlockCount(2000)
        self.ai_log.setMinimumHeight(105)
        layout.addWidget(self.ai_log)
        return self.scrolling_tab(layout)

    @staticmethod
    def with_browse(line_edit, callback):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit, 1)
        button = QPushButton("Browse…")
        button.clicked.connect(callback)
        layout.addWidget(button)
        return row

    def restore_settings(self):
        engine_candidate = self.find_engine()
        model_candidate = self.find_model()
        saved_engine = self.settings.value("ai/executable", "")
        saved_model = self.settings.value("ai/model", "")
        self.sd_executable.setText(str(engine_candidate) if engine_candidate else saved_engine)
        self.sd_model.setText(str(model_candidate) if model_candidate else saved_model)
        selected_name = Path(self.sd_model.text()).name
        for index in range(self.model_choice.count()):
            entry = resolve_model(self.model_choice.itemData(index))
            if entry.filename == selected_name:
                self.model_choice.setCurrentIndex(index)
                break
        self.output_dir.setText(self.settings.value("ai/output_dir", str(Path.home() / "Pictures")))
        self.ai_backend.setCurrentIndex(int(self.settings.value("ai/backend_index", 0)))

    def closeEvent(self, event):
        if self.install_worker is not None and self.install_worker.isRunning():
            self.install_worker.cancel()
            self.install_worker.wait(6000)
            if self.install_worker.isRunning():
                QMessageBox.warning(self, "Installer active", "Please cancel the active download before closing.")
                event.ignore()
                return
        if self.ai_process is not None and self.ai_process.state() != QProcess.ProcessState.NotRunning:
            response = QMessageBox.question(self, "Generation active", "Cancel generation and exit?")
            if response != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.ai_process.kill()
            self.ai_process.waitForFinished(3000)
        self.settings.setValue("ai/executable", self.sd_executable.text().strip())
        self.settings.setValue("ai/model", self.sd_model.text().strip())
        self.settings.setValue("ai/output_dir", self.output_dir.text().strip())
        self.settings.setValue("ai/backend_index", self.ai_backend.currentIndex())
        super().closeEvent(event)

    def apply_prompt_preset(self):
        presets = {
            "wasteland": ("Wide cinematic post-apocalyptic American wasteland, lonely ruined highway, "
                         "dramatic distant storm clouds, atmospheric perspective, painterly digital concept art, "
                         "no recognizable brands, no lettering", 3),
            "blues": ("Authentic 1950s Chicago blues club interior at midnight, upright bass, saxophone, "
                       "smoky golden stage lights, moody vintage record cover photograph, atmospheric, no text", 0),
            "thumbnail": ("Cinematic high-contrast video game scene, dramatic lighting and powerful focal point, "
                           "clean empty space on the left for a video title, professional digital key art, no text", 3),
            "portrait": ("Expressive dramatic character portrait, detailed mature face, shallow depth of field, "
                          "professional digital painting, vertical composition, no lettering", 5),
            "meme": ("Expressive humorous cinematic still of an absurd everyday situation, "
                      "strong central subject, open negative space for overlay text, no lettering", 0),
        }
        choice = self.prompt_preset.currentData()
        if choice in presets:
            prompt, index = presets[choice]
            self.ai_prompt.setPlainText(prompt)
            self.ai_negative.setPlainText("text, writing, logo, watermark, blurred, distorted, duplicated subjects")
            self.dim_preset.setCurrentIndex(index)

    def find_engine(self):
        candidates = [engine_dir(), BASE_DIR / "engine",
                      Path(sys.executable).resolve().parent / "engine"]
        for folder in candidates:
            path = locate_engine(folder)
            if path:
                return path
        return None

    def find_model(self):
        candidates = [model_dir(), BASE_DIR / "models",
                      Path(sys.executable).resolve().parent / "models"]
        for info in models():
            for folder in candidates:
                path = info.installed_path(folder)
                if path.is_file() and path.stat().st_size >= info.minimum_size_bytes:
                    return path
        return None

    def on_model_choice(self, *args):
        info = resolve_model(self.model_choice.currentData())
        model_paths = [info.installed_path(model_dir()), info.installed_path(BASE_DIR / "models"),
                       info.installed_path(Path(sys.executable).resolve().parent / "models")]
        installed = next((p for p in model_paths if p.is_file() and
                         p.stat().st_size >= info.minimum_size_bytes), None)
        if installed:
            self.sd_model.setText(str(installed))
            self.install_status.setText("Found model (verify SHA-256 to confirm integrity): " + str(installed))
        else:
            self.install_status.setText("Not installed: " + info.display_name)
        self.model_license.setText(f"{info.license_name}. Review: {info.license_url}")

    def _begin_install(self, task, model_id="", engine_flavor="vulkan", bundle_path=""):
        if self.install_worker is not None and self.install_worker.isRunning():
            QMessageBox.information(self, "Installation busy", "Another install or verification is in progress.")
            return
        worker = InstallWorker(task, model_id, engine_flavor, bundle_path)
        self.install_worker = worker
        worker.progress_info.connect(self.update_install_progress)
        worker.succeeded.connect(self.install_finished)
        worker.failed.connect(self.install_failed)
        worker.finished.connect(self.install_idle)
        self.install_cancel_btn.setEnabled(task != "bundle")
        self.model_install_btn.setEnabled(False)
        self.model_verify_btn.setEnabled(False)
        self.engine_install_btn.setEnabled(False)
        self.install_progress.setRange(0, 0)
        self.install_status.setText("Working on local setup...")
        worker.start()

    def install_selected_model(self):
        info = resolve_model(self.model_choice.currentData())
        answer = QMessageBox.question(self, "Install image model",
                        "Download " + info.display_name + "?\n\n"
                        "This downloads multi-GB third-party weights from Hugging Face. "
                        "CreativeML OpenRAIL-M terms and use restrictions apply. "
                        "The installed model subsequently operates offline.\n\n"
                        "License: " + info.license_url,
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            self._begin_install("model", model_id=info.id)

    def verify_selected_model(self):
        self._begin_install("verify", model_id=self.model_choice.currentData())

    def install_vulkan_engine(self):
        answer = QMessageBox.question(self, "Install generation engine",
                       "Download the official stable-diffusion.cpp Windows Vulkan release "
                       "from GitHub? A third-party executable will be saved to your user data folder.")
        if answer == QMessageBox.StandardButton.Yes:
            self._begin_install("engine")

    def update_install_progress(self, current, total, status):
        if total:
            self.install_progress.setRange(0, 100)
            self.install_progress.setValue(min(100, int(current * 100 / total)))
            self.install_status.setText(f"{status}: {current / (1024**2):,.1f} / {total / (1024**2):,.1f} MiB")
        else:
            self.install_progress.setRange(0, 0)
            self.install_status.setText(f"{status}: {current / (1024**2):,.1f} MiB")

    def install_finished(self, path):
        if Path(path).suffix.lower() in (".gguf", ".safetensors"):
            self.sd_model.setText(path)
            self.install_status.setText("Model installed and SHA-256 verified: " + path)
        elif path.lower().endswith(".zip"):
            self.install_status.setText("Offline ZIP created: " + path)
        else:
            self.sd_executable.setText(path)
            self.install_status.setText("Engine installed: " + path)
        self.install_progress.setRange(0, 100)
        self.install_progress.setValue(100)

    def install_failed(self, message):
        self.install_status.setText("Setup failed: " + message)
        self.install_progress.setRange(0, 100)
        QMessageBox.warning(self, "Setup failed", message)

    def install_idle(self):
        self.install_cancel_btn.setEnabled(False)
        self.model_install_btn.setEnabled(True)
        self.model_verify_btn.setEnabled(True)
        self.engine_install_btn.setEnabled(sys.platform == "win32")

    def cancel_install(self):
        if self.install_worker and self.install_worker.isRunning():
            self.install_worker.cancel()
            self.install_status.setText("Cancel requested. Partial model download is retained for resume.")

    def make_offline_bundle(self):
        info = resolve_model(self.model_choice.currentData())
        target, _ = QFileDialog.getSaveFileName(self, "Offline archive destination",
                     str(Path.home() / "Downloads" / "MemeGenieAI-Offline.zip"),
                     "ZIP archive (*.zip)")
        if not target:
            return
        self._begin_install("bundle", model_id=info.id, bundle_path=target)

    def build_gallery_tab(self):
        root = QVBoxLayout()
        root.addWidget(QLabel("Choose any image from your exports. Double-click to load it into the editor."))
        self.gallery = QListWidget()
        self.gallery.itemDoubleClicked.connect(self.gallery_open)
        root.addWidget(self.gallery)
        row = QHBoxLayout()
        refresh = QPushButton("Refresh Image Library")
        refresh.clicked.connect(self.refresh_gallery)
        row.addWidget(refresh)
        open_any = QPushButton("Open image from disk...")
        open_any.clicked.connect(self.choose_base_image)
        row.addWidget(open_any)
        root.addLayout(row)
        return self.scrolling_tab(root)

    def refresh_gallery(self):
        folder = Path(self.output_dir.text().strip()).expanduser()
        self.gallery.clear()
        if not folder.is_dir():
            return
        for file in sorted(folder.iterdir(), key=lambda f: f.stat().st_mtime,
                           reverse=True):
            if file.is_file() and file.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                item = QListWidgetItem(file.name)
                item.setData(Qt.ItemDataRole.UserRole, str(file))
                self.gallery.addItem(item)
            if self.gallery.count() >= 500:
                break

    def gallery_open(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        try:
            self.load_base_image(path, from_ai=False)
            self.tabs.setCurrentIndex(0)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot open image", str(exc))

    def browse_engine(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select stable-diffusion.cpp sd-cli", "",
                                              "Executable (sd-cli* *.exe);;All files (*)")
        if path:
            self.sd_executable.setText(path)

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select single-file model", "",
                                              "Model (*.safetensors *.ckpt *.gguf *.pth);;All files (*)")
        if path:
            self.sd_model.setText(path)

    def browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Save generated images to", self.output_dir.text())
        if path:
            self.output_dir.setText(path)

    def generate_ai_image(self):
        if self.ai_process is not None and self.ai_process.state() != QProcess.ProcessState.NotRunning:
            return
        output_dir = Path(self.output_dir.text().strip()).expanduser()
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(self, "Output folder", str(exc))
            return
        dimensions = self.dim_preset.currentData()
        seed = self.ai_seed.value()
        if seed < 0:
            seed = secrets.randbelow(2147483647)
        self.ai_output = output_dir / ("meme_genie_ai_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".png")
        request = GenerationRequest(
            executable=self.sd_executable.text().strip(),
            model=self.sd_model.text().strip(),
            output=str(self.ai_output),
            prompt=self.ai_prompt.toPlainText(),
            negative_prompt=self.ai_negative.toPlainText(),
            width=dimensions[0], height=dimensions[1], steps=self.ai_steps.value(),
            cfg_scale=self.ai_cfg.value(), seed=seed,
            sampler=self.ai_sampler.currentText(), backend=self.ai_backend.currentData(),
            compatibility_mode=self.ai_compat_mode.isChecked(),
        )
        try:
            args = build_sd_cli_arguments(request)
        except ValueError as exc:
            QMessageBox.warning(self, "Check AI settings", str(exc))
            return

        self.ai_generation_metadata = {
            "Meme Genie AI prompt": request.prompt,
            "Meme Genie AI negative prompt": request.negative_prompt,
            "Meme Genie AI seed": str(request.seed),
            "Meme Genie AI model": Path(request.model).name,
            "Meme Genie AI steps": str(request.steps),
            "Meme Genie AI CFG": str(request.cfg_scale),
            "Meme Genie AI sampler": request.sampler,
        }
        self.ai_canceled = False
        self.ai_generate_btn.setEnabled(False)
        self.ai_cancel_btn.setEnabled(True)
        self.ai_progress.setRange(0, 0)
        self.ai_status.setText(f"Generating locally. Seed: {seed} (output: {self.ai_output.name})")
        self.ai_log.clear()
        # Keep the complete diagnostic trace on disk. Qt's visible log is deliberately bounded.
        self.ai_log_path = output_dir / (self.ai_output.stem + ".log")
        self.ai_last_command = subprocess.list2cmdline([request.executable, *args])
        self.append_ai_log("Engine: " + request.executable + "\nModel: " + request.model
                           + "\nCommand: " + self.ai_last_command + "\n")
        self.ai_process = QProcess(self)
        self.ai_process.setWorkingDirectory(str(Path(request.executable).resolve().parent))
        self.ai_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.ai_process.readyReadStandardOutput.connect(self.read_ai_log)
        self.ai_process.errorOccurred.connect(self.on_ai_process_error)
        self.ai_process.finished.connect(self.ai_finished)
        self.ai_process.start(request.executable, args)

    def append_ai_log(self, chunk: str):
        if not chunk:
            return
        self.ai_log.moveCursor(QTextCursor.MoveOperation.End)
        self.ai_log.insertPlainText(chunk)
        self.ai_log.moveCursor(QTextCursor.MoveOperation.End)
        if self.ai_log_path is not None:
            try:
                with self.ai_log_path.open("a", encoding="utf-8", errors="replace") as log:
                    log.write(chunk)
            except OSError as exc:
                self.ai_status.setText("Cannot save diagnostic log: " + str(exc))

    def copy_engine_command(self):
        if self.ai_last_command:
            QApplication.clipboard().setText(self.ai_last_command)
            self.statusBar().showMessage("Last engine command copied to clipboard.")
        else:
            QMessageBox.information(self, "Engine command", "Run a generation attempt first.")

    def open_ai_log_folder(self):
        folder = self.ai_log_path.parent if self.ai_log_path else Path(self.output_dir.text()).expanduser()
        if folder.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder.resolve())))
        else:
            QMessageBox.information(self, "Diagnostic logs", "No generated image folder exists yet.")

    def read_ai_log(self):
        if self.ai_process is None:
            return
        chunk = bytes(self.ai_process.readAllStandardOutput()).decode("utf-8", "replace")
        self.append_ai_log(chunk)

    def on_ai_process_error(self, error):
        if self.ai_process is not None:
            self.append_ai_log("\n[QProcess error] " + self.ai_process.errorString() + "\n")
        if error == QProcess.ProcessError.FailedToStart:
            self.ai_status.setText("Failed to start sd-cli. Check the executable and its runtime dependencies.")
            self.ai_generate_btn.setEnabled(True)
            self.ai_cancel_btn.setEnabled(False)
            self.ai_progress.setRange(0, 100)
            self.ai_progress.setValue(0)
            QMessageBox.warning(self, "Engine error", self.ai_process.errorString())

    def cancel_ai(self):
        proc = self.ai_process
        if proc is None or proc.state() == QProcess.ProcessState.NotRunning:
            return
        self.ai_canceled = True
        self.ai_status.setText("Canceling local generation…")
        proc.terminate()
        QTimer.singleShot(2000, lambda: proc.kill() if proc.state() != QProcess.ProcessState.NotRunning else None)

    def ai_finished(self, exit_code, exit_status):
        self.read_ai_log()
        self.append_ai_log("\n[Process finished] exit code: " + str(exit_code)
                           + "; status: " + str(exit_status) + "\n")
        self.ai_generate_btn.setEnabled(True)
        self.ai_cancel_btn.setEnabled(False)
        self.ai_progress.setRange(0, 100)
        self.ai_progress.setValue(0)
        if self.ai_canceled:
            self.ai_status.setText("Generation canceled. Diagnostic log saved.")
            return
        if exit_status != QProcess.ExitStatus.NormalExit or exit_code != 0:
            self.ai_status.setText(f"Generation failed (exit code {exit_code}). See saved diagnostic log.")
            tail = self.ai_log.toPlainText().splitlines()[-16:]
            location = str(self.ai_log_path) if self.ai_log_path else "Not available"
            QMessageBox.warning(self, "Image generation failed",
                                f"sd-cli exit code: {exit_code}\n"
                                f"Exit status: {exit_status}\n\n"
                                f"Last log lines:\n{chr(10).join(tail)}\n\n"
                                f"Full diagnostic log: {location}\n\n"
                                "Use Copy engine command to reproduce this outside the GUI.")
            return
        if self.ai_output is None or not self.ai_output.is_file():
            self.ai_status.setText("Engine finished without producing the expected PNG file.")
            return
        try:
            self.load_base_image(str(self.ai_output), from_ai=True)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid image", str(exc))
            return
        self.ai_progress.setValue(100)
        self.refresh_gallery()
        self.ai_status.setText("Image generated and opened in editor. Source saved at: " + str(self.ai_output))
        self.tabs.setCurrentIndex(0)
        self.statusBar().showMessage("AI image generated successfully.")

    def populate_font_selector(self):
        font_dirs = [
            "/usr/share/fonts", "/usr/local/share/fonts", str(Path.home() / ".fonts"),
            "C:/Windows/Fonts", "/Library/Fonts", "/System/Library/Fonts",
        ]
        self.font_map: dict[str, str] = {}
        for folder in font_dirs:
            if not Path(folder).is_dir():
                continue
            for root, _, files in os.walk(folder):
                for name in files:
                    if name.lower().endswith((".ttf", ".otf")):
                        self.font_map[name] = str(Path(root) / name)
        self.font_selector.addItems(sorted(self.font_map))
        default_name = "DejaVuSans-Bold.ttf"
        if default_name in self.font_map:
            self.font_selector.setCurrentText(default_name)
        elif self.font_selector.count():
            self.font_selector.setCurrentIndex(0)
        else:
            self.selected_font = None

    def on_font_select(self, name):
        self.selected_font = self.font_map.get(name)
        self.update_preview()

    def choose_text_color(self):
        color = QColorDialog.getColor(QColor(*self.text_color[:3]), self)
        if color.isValid():
            self.text_color = (color.red(), color.green(), color.blue(), 255)
            self.update_preview()

    def toggle_text_shadow(self, checked):
        self.text_shadow = checked
        self.update_preview()

    def choose_base_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "",
                                               "Images (*.png *.jpg *.jpeg *.bmp *.webp *.gif)")
        if path:
            try:
                self.load_base_image(path, from_ai=False)
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "Unable to load image", str(exc))

    def load_base_image(self, path: str, from_ai: bool):
        with Image.open(path) as original:
            original.load()
            self.base_image = original.convert("RGBA")
        self.base_image_path = path
        if not from_ai:
            self.ai_generation_metadata = {}
        self.update_preview()
        self.statusBar().showMessage("Loaded " + path)

    def selected_alignment(self):
        for btn in self.align_group.buttons():
            if btn.isChecked():
                return btn.text().lower()
        return "center"

    @staticmethod
    def font_for_path(font_path: str | None, size: int):
        try:
            if font_path:
                return ImageFont.truetype(font_path, size)
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
        except OSError:
            try:
                return ImageFont.load_default(size=size)
            except TypeError:
                return ImageFont.load_default()

    @staticmethod
    def wrapped_text(draw, raw_text: str, font, max_width: int):
        lines: list[str] = []
        for paragraph in raw_text.splitlines():
            if not paragraph.strip():
                lines.append("")
                continue
            line = ""
            for word in paragraph.split():
                candidate = (line + " " + word).strip()
                if not line or draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
                    line = candidate
                else:
                    lines.append(line)
                    line = word
            if line:
                lines.append(line)
        return lines

    def render_meme(self):
        if self.base_image is None:
            return None
        result = self.base_image.copy()
        text = self.text_input.toPlainText()
        draw = ImageDraw.Draw(result)
        width, height = result.size
        if text.strip():
            # Scale fonts down on smaller source images while preserving large-image clarity.
            font_px = min(self.font_size.value(), max(12, width // 7))
            font = self.font_for_path(self.selected_font, font_px)
            lines = self.wrapped_text(draw, text, font, int(width * .9))
            line_spacing = max(4, font_px // 6)
            metrics = [draw.textbbox((0, 0), line or "Ag", font=font) for line in lines]
            heights = [max(12, box[3] - box[1]) for box in metrics]
            total_height = sum(heights) + line_spacing * max(0, len(lines) - 1)
            y = max(8, (height - total_height) // 2)
            shadow_color = (0, 0, 0, 255) if sum(self.text_color[:3]) >= 384 else (255, 255, 255, 255)
            for line, box, line_height in zip(lines, metrics, heights):
                if line:
                    span = box[2] - box[0]
                    align = self.selected_alignment()
                    x = 10 if align == "left" else (width - span - 10 if align == "right" else (width - span) // 2)
                    x = max(5, x)
                    if self.text_shadow:
                        offset = max(2, font_px // 20)
                        draw.text((x + offset, y + offset), line, font=font, fill=shadow_color)
                    draw.text((x, y), line, font=font, fill=self.text_color)
                y += line_height + line_spacing
        if self.attribution_checkbox.isChecked():
            credit = "Meme Genie, by @kaotickj"
            size = max(10, min(20, width // 70))
            font = self.font_for_path(self.selected_font, size)
            box = draw.textbbox((0, 0), credit, font=font)
            x = max(5, width - (box[2] - box[0]) - 12)
            y = max(5, height - (box[3] - box[1]) - 12)
            draw.text((x, y), credit, font=font, fill=(255, 255, 255, 220),
                      stroke_width=1, stroke_fill=(0, 0, 0, 180))
        return result

    def update_preview(self, *args):
        rendered = self.render_meme()
        if rendered is None:
            return
        self.image_preview.show_pil(rendered)
        self.preview_size.setText(f"{rendered.width} × {rendered.height} px")

    def fetch_quote(self):
        try:
            response = requests.get("https://kdgwebsolutions.com/quote.php", timeout=12)
            response.raise_for_status()
            element = BeautifulSoup(response.text, "html.parser").find("p")
            if element is None:
                raise ValueError("Response had no quote paragraph.")
            self.text_input.setPlainText(element.get_text(strip=True))
        except (requests.RequestException, ValueError) as exc:
            QMessageBox.warning(self, "Quote retrieval failed", str(exc))

    def generate_meme(self):
        result = self.render_meme()
        if result is None:
            QMessageBox.warning(self, "No image", "Load or generate an image first.")
            return
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Edited Image", "meme.png", "PNG (*.png);;JPEG (*.jpg *.jpeg)")
        if not path:
            return
        target = Path(path)
        if target.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            target = target.with_suffix(".jpg" if selected_filter.startswith("JPEG") else ".png")
        try:
            if target.suffix.lower() == ".png":
                metadata = PngImagePlugin.PngInfo()
                for k, v in self.ai_generation_metadata.items():
                    metadata.add_text(k, v)
                result.save(target, format="PNG", pnginfo=metadata)
            else:
                result.convert("RGB").save(target, format="JPEG", quality=95)
        except OSError as exc:
            QMessageBox.warning(self, "Save error", str(exc))
            return
        self.refresh_gallery()
        self.statusBar().showMessage("Saved " + str(target))
        QMessageBox.information(self, "Saved", "Image saved to:\n" + str(target))

    def export_sized(self, width: int, height: int):
        """Precisely size images for YouTube; resize doesn't add AI detail."""
        rendered = self.render_meme()
        if rendered is None:
            QMessageBox.warning(self, "No image", "Load or generate an image first.")
            return
        # Scale and crop proportionally to fill the target canvas.
        sized = ImageOps.fit(rendered.convert("RGB"), (width, height),
                             method=Image.Resampling.LANCZOS,
                             centering=(.5, .5))
        save_path, kind = QFileDialog.getSaveFileName(self,
            f"Export {width} × {height}",
            str(Path(self.output_dir.text().strip()).expanduser() / f"meme_{width}x{height}.png"),
            "PNG (*.png);;JPEG (*.jpg)")
        if not save_path:
            return
        target = Path(save_path)
        if target.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            target = target.with_suffix(".jpg" if kind.startswith("JPEG") else ".png")
        try:
            if target.suffix.lower() == ".png":
                info = PngImagePlugin.PngInfo()
                for key, value in self.ai_generation_metadata.items():
                    info.add_text(key, value)
                sized.save(target, pnginfo=info)
            else:
                sized.save(target, quality=95)
        except OSError as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        self.refresh_gallery()
        self.statusBar().showMessage("Saved " + str(target))

    def save_base_image(self):
        if self.base_image is None:
            QMessageBox.warning(self, "No image", "Load or generate an image first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Original Base Image", "base_image.png", "PNG (*.png)")
        if path:
            target = Path(path)
            if target.suffix.lower() != ".png":
                target = target.with_suffix(".png")
            self.base_image.save(target, format="PNG")
            self.statusBar().showMessage("Base image saved to " + str(target))

    def show_instructions(self):
        QMessageBox.information(self, "Instructions", "Meme Editor: Open an image, enter text, choose font and alignment, "
                                "and save the finished image. The text is optional.\n\n"
                                "AI Image Generation: Choose a model and click Install Selected Model. "
                                "On Windows, click Install Windows Vulkan Engine. Then describe your artwork, "
                                "choose an optional preset and click Generate Image. Downloaded weights and "
                                "the engine operate entirely offline after setup.\n\n"
                                "Image Library: Open any generated PNG in the selected exports folder. "
                                "Editor export can resize to YouTube or Shorts dimensions.\n\n"
                                "For fully offline redistribution, install and verify the weight files, "
                                "then choose Create Offline Bundle. The original source archive "
                                "does not contain several-gigabyte model weights.")

    def show_about(self):
        QMessageBox.information(self, "About Meme Genie", f"Meme Genie {self.APP_VERSION}\n\n"
                                "Created by @kaotickj / Kaotick Jay\n"
                                "https://github.com/kaotickj\n\n"
                                "Local image generation through stable-diffusion.cpp. Model installation uses "
                                "Hugging Face, and Windows engine installation uses GitHub. "
                                "No cloud inference API, account, or telemetry is required.\n"
                                "All third-party engines and models retain their respective licenses.")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#19232e"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#e5edf6"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#121a24"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#273342"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#e5edf6"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#26364a"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#e5edf6"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#4679aa"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#26364a"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#ffffff"))
    app.setPalette(palette)
    window = MemeGeneratorApp()
    window.show()
    if "--smoke-test" in sys.argv:
        # Exercise real Qt startup and packaged resources without downloading models.
        QTimer.singleShot(250, app.quit)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
