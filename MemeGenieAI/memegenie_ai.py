"""Meme Genie AI Studio: meme editing, local models and image generation.

Copyright (c) 2026 @kaotickj. Bundled model catalog with explicit install and offline-bundle support.
Third-party model weights are installed with the model manager.
Stable Diffusion image generation stays local; Luna caption suggestions are opt-in.
"""

from __future__ import annotations

import os
import copy
import json
import secrets
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageStat, PngImagePlugin

from PyQt6.QtCore import Qt, QProcess, QSettings, QTimer, QThread, QUrl, QRectF, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QDesktopServices, QIcon, QImage, QPalette, QPixmap, QTextCursor, QPen, QBrush
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QDialog, QDialogButtonBox,
    QFileDialog, QFormLayout, QGraphicsScene, QGraphicsView, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QRadioButton, QButtonGroup, QScrollArea,
    QSpinBox, QSplitter, QTabWidget, QTextEdit, QVBoxLayout, QWidget, QListWidget, QListWidgetItem, QSizePolicy,
)

from meme_project import TextLayer, render_layers, save_project, load_project, validate_image
from ai_meme_layout import suggest_layout, fit_caption_layers, place_around_subjects, subject_clearance_regions, NoCaptionSpace
from sd_backend import GenerationRequest, build_sd_cli_arguments
from model_manager import (
    BASE_DIR, app_data_dir, model_dir, engine_dir, models, resolve_model,
    download_model, verify_model, install_engine, locate_engine, create_offline_bundle,
    retrieve_model_license,
)


class LayoutWorker(QThread):
    succeeded = pyqtSignal(str, object)
    failed = pyqtSignal(str)
    def __init__(self, image, instruction, model, *, force_analysis=False, provider="local"):
        super().__init__()
        self.image, self.instruction, self.model = image.copy(), instruction, model
        self.force_analysis = force_analysis
        self.provider = provider
    def run(self):
        try:
            description, layers = suggest_layout(self.image, self.instruction, self.model, alternatives=True, force_analysis=self.force_analysis, provider=self.provider)
            self.succeeded.emit(description, layers)
        except Exception as exc:
            self.failed.emit(str(exc))


class OpenAIImageWorker(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, prompt, negative_prompt, dimensions, output):
        super().__init__()
        self.prompt, self.negative_prompt = prompt, negative_prompt
        self.dimensions, self.output = dimensions, output

    def run(self):
        from openai_images import generate_image
        try:
            image = generate_image(self.prompt, self.negative_prompt, self.dimensions,
                                   canceled=self.isInterruptionRequested)
            self.succeeded.emit(image)
        except Exception as exc:
            self.failed.emit(str(exc))


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


class ProtectedAreaView(ImageView):
    """Optional drag-to-protect interaction, confined to the suggestion preview."""
    area_selected = pyqtSignal(object)
    area_clicked = pyqtSignal(int)

    def __init__(self,parent=None):
        super().__init__(parent)
        self.selecting=False
        self.reviewing=False
        self._areas=[]
        self._drag_start=None
        self._drag_item=None

    def show_pil(self,image):
        self._drag_start=None
        self._drag_item=None
        super().show_pil(image)

    def set_selecting(self,active):
        self.selecting=active
        self.setCursor(Qt.CursorShape.CrossCursor if active else Qt.CursorShape.ArrowCursor)

    def show_areas(self,areas,selected=-1):
        self._areas=copy.deepcopy(areas)
        bounds=self.scene().sceneRect()
        clearances=subject_clearance_regions(areas)
        for index,(area,clearance) in enumerate(zip(areas,clearances)):
            color=QColor('#66ffcc' if index==selected else '#ffcc66')
            pen=QPen(color);pen.setWidth(2);pen.setCosmetic(True)
            self.scene().addRect(area['x']*bounds.width(),area['y']*bounds.height(),
                                area['width']*bounds.width(),area['height']*bounds.height(),
                                pen,QBrush(QColor(color.red(),color.green(),color.blue(),35)))
            margin=QPen(color);margin.setCosmetic(True);margin.setStyle(Qt.PenStyle.DashLine)
            self.scene().addRect(clearance['x']*bounds.width(),clearance['y']*bounds.height(),
                                clearance['width']*bounds.width(),clearance['height']*bounds.height(),margin)
            label=self.scene().addText(str(index+1))
            label.setDefaultTextColor(color)
            label.setScale(max(1,bounds.width()/350))
            label.setPos(area['x']*bounds.width(),area['y']*bounds.height())

    def mousePressEvent(self,event):
        if self.selecting and event.button()==Qt.MouseButton.LeftButton:
            point=self.mapToScene(event.pos())
            if self.scene().sceneRect().contains(point):
                self._drag_start=point
            event.accept();return
        if self.reviewing and event.button()==Qt.MouseButton.LeftButton:
            point=self.mapToScene(event.pos());bounds=self.scene().sceneRect()
            candidates=[(area['width']*area['height'],index) for index,area in enumerate(self._areas)
                        if QRectF(area['x']*bounds.width(),area['y']*bounds.height(),
                                  area['width']*bounds.width(),area['height']*bounds.height()).contains(point)]
            if candidates:
                self.area_clicked.emit(min(candidates)[1]);event.accept();return
        super().mousePressEvent(event)

    def mouseMoveEvent(self,event):
        if self.selecting and self._drag_start is not None:
            bounds=self.scene().sceneRect()
            rect=QRectF(self._drag_start,self.mapToScene(event.pos())).normalized().intersected(bounds)
            if self._drag_item is not None:self.scene().removeItem(self._drag_item)
            pen=QPen(QColor('#ffcc66'));pen.setWidth(2);pen.setCosmetic(True)
            self._drag_item=self.scene().addRect(rect,pen,QBrush(QColor(255,204,102,45)))
            event.accept();return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self,event):
        if self.selecting and event.button()==Qt.MouseButton.LeftButton:
            start=self._drag_start;self._drag_start=None
            if start is not None:
                bounds=self.scene().sceneRect()
                rect=QRectF(start,self.mapToScene(event.pos())).normalized().intersected(bounds)
                if rect.width()>=bounds.width()*.01 and rect.height()>=bounds.height()*.01:
                    self.area_selected.emit(dict(label='User protected area',x=rect.x()/bounds.width(),
                        y=rect.y()/bounds.height(),width=rect.width()/bounds.width(),height=rect.height()/bounds.height()))
            event.accept();return
        super().mouseReleaseEvent(event)


class MemeGeneratorApp(QMainWindow):
    APP_VERSION = "1.2.1"

    def __init__(self):
        super().__init__()
        self.text_layers = [TextLayer()]
        self._active_layer = 0
        self._loading_layers = True
        self.base_image: Image.Image | None = None
        self.base_image_path: str | None = None
        self.selected_font: str | None = None
        self.text_color = (255, 255, 255, 255)
        self.text_shadow = True
        self.ai_process: QProcess | None = None
        self.install_worker: InstallWorker | None = None
        self.layout_worker = None
        self.image_worker = None
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
        self._loading_layers = False
        self.restore_settings()
        self.on_model_choice()
        self.refresh_gallery()
        self.statusBar().showMessage("Ready. Choose a background, type your caption, and export your meme.")

    def build_ui(self):
        file_menu = self.menuBar().addMenu("File")
        self.add_action(file_menu, "Open Image…", self.choose_base_image)
        self.add_action(file_menu, "Open Editable Project…", self.open_editable_project)
        self.add_action(file_menu, "Save Editable Project…", self.save_editable_project)
        self.add_action(file_menu, "Save Edited Image…", self.generate_meme)
        self.add_action(file_menu, "Save Base Image with Credit…", self.save_base_image)
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
        self.tabs.addTab(self.build_ai_tab(), "Experimental Image Generation")
        self.tabs.setTabVisible(1, False)
        tools_menu = self.menuBar().addMenu("Tools")
        generation_action = self.add_action(tools_menu, "Experimental Image Generation", lambda: None)
        generation_action.setCheckable(True)
        generation_action.toggled.connect(lambda visible: self.tabs.setTabVisible(1, visible))
        self.tabs.addTab(self.build_gallery_tab(), "Image Library")
        splitter.addWidget(self.tabs)

        preview_panel = QWidget()
        pv_layout = QVBoxLayout(preview_panel)
        pv_layout.addWidget(QLabel("<b>Image Preview</b>"))
        self.preview_format = QComboBox()
        self.preview_format.addItem("Original image", None)
        self.preview_format.addItem("Widescreen · 1920 × 1080", (1920,1080))
        self.preview_format.addItem("Shorts · 1080 × 1920", (1080,1920))
        self.preview_format.currentIndexChanged.connect(self.update_preview)
        pv_layout.addWidget(self.preview_format)
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
        open_btn = QPushButton("1. Choose Background…")
        open_btn.clicked.connect(self.choose_base_image)
        layout.addWidget(open_btn)

        layer_row = QHBoxLayout()
        self.layer_choice = QComboBox()
        self.layer_choice.addItem("Text layer 1")
        self.layer_choice.currentIndexChanged.connect(self.select_text_layer)
        layer_row.addWidget(self.layer_choice, 1)
        add_layer = QPushButton("Add text")
        add_layer.clicked.connect(self.add_text_layer)
        remove_layer = QPushButton("Remove text")
        remove_layer.clicked.connect(self.remove_text_layer)
        layer_row.addWidget(add_layer)
        layer_row.addWidget(remove_layer)
        layout.addLayout(layer_row)
        layout.addWidget(QLabel("2. Type your caption:"))
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Type a caption, or ask the AI to write and style one.")
        self.text_input.setFixedHeight(130)
        self.text_input.textChanged.connect(self.on_caption_changed)
        layout.addWidget(self.text_input)

        self.caption_brief = QLineEdit()
        self.caption_brief.setPlaceholderText("AI idea or tone, e.g. a joke about Monday mornings")
        self.caption_brief.setMaxLength(2000)
        layout.addWidget(self.caption_brief)
        self.caption_model = QComboBox()
        self.caption_model.addItems(["gemma3:4b", "qwen3.5:9b-q8_0", "gemma4:12b-it-q8_0"])
        self.caption_model.addItem("OpenAI · gpt-6-luna (online)", "openai")
        layout.addWidget(self.caption_model)
        self.caption_provider_note = QLabel()
        self.caption_provider_note.setWordWrap(True)
        layout.addWidget(self.caption_provider_note)
        self.caption_model.currentIndexChanged.connect(self.update_caption_provider_note)
        self.update_caption_provider_note()
        self.caption_ai_button = QPushButton("AI: Suggest memes")
        self.caption_ai_button.clicked.connect(self.start_ai_layout)
        self.caption_recheck_button = QPushButton("Recheck image")
        self.caption_recheck_button.setToolTip("Re-examine the image and create new suggestions. Use this when the AI misunderstood the picture.")
        self.caption_recheck_button.clicked.connect(lambda: self.start_ai_layout(force_analysis=True))
        ai_actions = QHBoxLayout()
        ai_actions.addWidget(self.caption_ai_button,1)
        ai_actions.addWidget(self.caption_recheck_button)
        layout.addLayout(ai_actions)
        self.caption_ai_status = QLabel("Ready to suggest a caption")
        self.caption_ai_status.setWordWrap(True)
        layout.addWidget(self.caption_ai_status)

        self.auto_layout_checkbox = QCheckBox("Automatically arrange captions")
        self.auto_layout_checkbox.setChecked(False)
        self.auto_layout_checkbox.toggled.connect(lambda enabled: self.auto_arrange_text() if enabled else None)
        layout.addWidget(self.auto_layout_checkbox)
        arrange = QPushButton("Arrange text now")
        arrange.clicked.connect(self.auto_arrange_text)
        layout.addWidget(arrange)
        details_toggle = QPushButton("Fine-tune text settings…")
        details_toggle.setCheckable(True)
        details_toggle.setChecked(True)
        layout.addWidget(details_toggle)
        self.text_details = QWidget()
        details = QVBoxLayout(self.text_details)
        self.text_details.setVisible(True)
        details_toggle.toggled.connect(self.text_details.setVisible)
        details_toggle.toggled.connect(lambda opened: self.auto_layout_checkbox.setChecked(False) if opened else None)
        layout.addWidget(self.text_details)

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
        self.layer_x = QSpinBox()
        self.layer_y = QSpinBox()
        self.layer_width = QSpinBox()
        for control, maximum, value in ((self.layer_x, 95, 5), (self.layer_y, 95, 40), (self.layer_width, 95, 90)):
            control.setRange(5 if control is self.layer_width else 0, maximum)
            control.setValue(value)
            control.setSuffix(" %")
            control.valueChanged.connect(self.update_preview)
        self.layer_x.valueChanged.connect(lambda x: self.layer_width.setMaximum(100-x))
        form.addRow("Left position:", self.layer_x)
        form.addRow("Top position:", self.layer_y)
        form.addRow("Text box width:", self.layer_width)

        color_btn = QPushButton("Select Color…")
        color_btn.clicked.connect(self.choose_text_color)
        form.addRow("Text color:", color_btn)
        details.addLayout(form)

        alignment = QHBoxLayout()
        self.align_group = QButtonGroup(self)
        for label in ("Left", "Center", "Right"):
            btn = QRadioButton(label)
            self.align_group.addButton(btn)
            btn.setChecked(label == "Center")
            btn.toggled.connect(self.update_preview)
            alignment.addWidget(btn)
        details.addLayout(alignment)

        self.text_shadow_checkbox = QCheckBox("Text shadow")
        self.text_shadow_checkbox.setChecked(True)
        self.text_shadow_checkbox.toggled.connect(self.shadow_action.setChecked)
        self.text_shadow_checkbox.toggled.connect(self.update_preview)
        self.shadow_action.toggled.connect(self.text_shadow_checkbox.setChecked)
        details.addWidget(self.text_shadow_checkbox)


        save_btn = QPushButton("3. Export Meme…")
        save_btn.clicked.connect(self.generate_meme)
        layout.insertWidget(layout.indexOf(self.text_details)-1, save_btn)
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
        intro = QLabel("Local AI images use Stable Diffusion. Use Install Model to download a verified "
                       "Stable Diffusion 1.5 checkpoint. On Windows, Install Engine obtains "
                       "the official Vulkan build. Once installed, generation is fully offline.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.image_provider = QComboBox()
        self.image_provider.addItem("Local · Stable Diffusion", "local")
        self.image_provider.addItem("OpenAI · gpt-image-2.6-flare (online)", "openai")
        layout.addWidget(self.image_provider)
        self.image_provider_note = QLabel()
        self.image_provider_note.setWordWrap(True)
        self.image_provider_note.setMaximumWidth(350)
        self.image_provider_note.setOpenExternalLinks(True)
        layout.addWidget(self.image_provider_note)
        self.image_provider.currentIndexChanged.connect(self.update_image_provider)

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
        self.prompt_preset.addItem("Bass instrument detail photograph", "instrument")
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
        self.update_image_provider()
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
        if self.image_worker is not None and self.image_worker.isRunning():
            QMessageBox.information(self, "OpenAI image request running",
                "Wait for the request to finish or time out before closing. Cancel discards its result; API charges may still apply.")
            event.ignore()
            return
        if self.layout_worker is not None and self.layout_worker.isRunning():
            QMessageBox.information(self, "Caption suggestion running", "The caption request is still running. Please wait for its result or timeout before closing.")
            event.ignore()
            return
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
            "instrument": ("Professional product photograph of one upright double bass, standing alone "
                           "on a wooden floor against a dark brown wall, warm spotlight, detailed "
                           "polished wood, sharp focus.", 0),
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
            negative = "text, writing, logo, watermark, blurred, distorted, duplicated subjects"
            if choice == "instrument":
                negative = "people, person, musician, man, woman, face, hands, " + negative
            self.ai_negative.setPlainText(negative)
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
        # User-browsed checkpoints can live outside the application model folders.
        # Switching the catalog must also find sibling checkpoints in that location.
        current_model = self.sd_model.text().strip()
        folders = ([Path(current_model).expanduser().parent] if current_model else [])
        folders += [model_dir(), BASE_DIR / "models",
                    Path(sys.executable).resolve().parent / "models"]
        model_paths = [info.installed_path(folder) for folder in folders]
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
        clear = QPushButton("Clear Image Library…")
        clear.clicked.connect(lambda: self.clear_gallery())
        root.addWidget(clear)
        restore = QPushButton("Restore cleared library entries")
        restore.clicked.connect(self.restore_gallery)
        root.addWidget(restore)
        return self.scrolling_tab(root)

    @staticmethod
    def library_key(file):
        stat = file.stat()
        return json.dumps([str(file.resolve()), stat.st_size, stat.st_mtime_ns])

    def library_files(self):
        folder = Path(self.output_dir.text().strip()).expanduser()
        if not folder.is_dir():
            return []
        return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}), key=lambda p:p.stat().st_mtime, reverse=True)

    def hidden_library_entries(self):
        try:
            entries=json.loads(self.settings.value("library/hidden_entries", "[]"))
            return set(entries) if isinstance(entries,list) and all(isinstance(x,str) for x in entries) else set()
        except (ValueError,TypeError):
            return set()

    def clear_gallery(self, *, confirm=True):
        files=self.library_files()
        hidden=self.hidden_library_entries()
        new={self.library_key(p) for p in files}-hidden
        if not new:
            self.statusBar().showMessage("Image Library is already clear.")
            return 0
        if confirm and QMessageBox.question(self,"Clear Image Library",f"Hide {len(new)} image entries from this library? No image files, saved projects or current canvas will be deleted. Restore cleared library entries brings them back.") != QMessageBox.StandardButton.Yes:
            return 0
        self.settings.setValue("library/hidden_entries",json.dumps(sorted(hidden|new)))
        self.settings.sync()
        self.refresh_gallery()
        self.statusBar().showMessage(f"Cleared {len(new)} library entries. All image files remain on disk.")
        return len(new)

    def restore_gallery(self):
        self.settings.remove("library/hidden_entries")
        self.settings.sync()
        self.refresh_gallery()
        self.statusBar().showMessage("Cleared library entries restored. No files changed.")

    def refresh_gallery(self):
        self.gallery.clear()
        hidden=self.hidden_library_entries()
        for file in self.library_files():
            if self.library_key(file) in hidden:
                continue
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

    def update_image_provider(self):
        online = self.image_provider.currentData() == "openai"
        self.image_provider_note.setText(
            'OpenAI: one paid image per click, medium quality/complexity. Sends prompts only. '
            'Negative prompt is added as an avoidance instruction. Standard API canvas is cropped/resized '
            'to your resolution. Steps, CFG, seed, sampler and backend apply only to local generation. '
            '<a href="https://developers.openai.com/api/docs/models/gpt-image-2.6-flare">Current pricing</a>. '
            'Cancel discards the result; charges may still apply.' if online else
            "Local generation stays on this computer and uses the selected engine/model and settings.")
        for control in (self.ai_steps, self.ai_cfg, self.ai_seed, self.ai_sampler, self.ai_backend, self.ai_compat_mode):
            control.setEnabled(not online)

    def start_openai_image(self):
        from openai_images import generation_body
        from openai_layout import api_key
        prompt, negative = self.ai_prompt.toPlainText(), self.ai_negative.toPlainText()
        dimensions = self.dim_preset.currentData()
        try:
            generation_body(prompt, negative, dimensions)
            api_key()
            output_dir = Path(self.output_dir.text().strip()).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "OpenAI image settings", str(exc))
            return
        output = output_dir / ("meme_genie_openai_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".png")
        self.ai_canceled = False
        self.ai_log_path = None
        self.ai_last_command = ""
        self.ai_log.clear()
        self.ai_generate_btn.setEnabled(False)
        self.ai_cancel_btn.setEnabled(True)
        self.ai_progress.setRange(0, 0)
        self.ai_status.setText("Generating one image with OpenAI (API charges apply)…")
        self.image_worker = OpenAIImageWorker(prompt, negative, dimensions, output)
        self.image_worker.succeeded.connect(self.openai_image_ready)
        self.image_worker.failed.connect(self.openai_image_failed)
        self.image_worker.finished.connect(self.openai_image_finished)
        self.image_worker.start()

    def openai_image_finished(self):
        self.ai_generate_btn.setEnabled(True)
        self.ai_cancel_btn.setEnabled(False)
        self.ai_progress.setRange(0, 100)

    def openai_image_failed(self, reason):
        self.ai_progress.setValue(0)
        self.ai_status.setText("OpenAI image unavailable: " + reason + ". Current editor image is unchanged.")

    def openai_image_ready(self, image):
        if self.ai_canceled:
            self.openai_image_failed("Result discarded; API charges may still apply")
            return
        from openai_images import MODEL, native_size
        worker = self.image_worker
        metadata = {"Meme Genie AI provider": "OpenAI", "Meme Genie AI model": MODEL,
                    "Meme Genie AI prompt": worker.prompt,
                    "Meme Genie AI negative prompt": worker.negative_prompt,
                    "Meme Genie AI quality": "medium", "Meme Genie AI complexity": "medium",
                    "Meme Genie AI API size": native_size(worker.dimensions)}
        temp_path = None
        try:
            pnginfo = PngImagePlugin.PngInfo()
            for key, value in metadata.items():
                pnginfo.add_text(key, value)
            with tempfile.NamedTemporaryFile(dir=worker.output.parent, prefix=".memegenie-", suffix=".png", delete=False) as handle:
                temp_path = Path(handle.name)
            self.add_attribution(image).save(temp_path, format="PNG", pnginfo=pnginfo)
            os.replace(temp_path, worker.output)
        except (OSError, ValueError) as exc:
            self.openai_image_failed("Image received but could not be saved: " + str(exc))
            return
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        # Keep the clean source in memory for editing; every saved image gets credit.
        self.base_image = image.copy()
        self.base_image_path = str(worker.output)
        self.ai_output = worker.output
        self.ai_generation_metadata = metadata
        if self.auto_layout_checkbox.isChecked():
            self.auto_arrange_text()
        else:
            self.update_preview()
        self.refresh_gallery()
        self.ai_progress.setRange(0, 100)
        self.ai_progress.setValue(100)
        self.ai_status.setText("OpenAI image generated and opened in editor. Credited image saved at: " + str(worker.output))
        self.tabs.setCurrentIndex(0)
        self.statusBar().showMessage("OpenAI image generated successfully.")

    def generate_ai_image(self):
        if self.image_worker is not None and self.image_worker.isRunning():
            return
        if self.ai_process is not None and self.ai_process.state() != QProcess.ProcessState.NotRunning:
            return
        if self.image_provider.currentData() == "openai":
            self.start_openai_image()
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
        if self.image_worker is not None and self.image_worker.isRunning():
            self.ai_canceled = True
            self.image_worker.requestInterruption()
            self.ai_cancel_btn.setEnabled(False)
            self.ai_status.setText("Discarding OpenAI result when the request returns; API charges may still apply.")
            return
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
            validate_image(original)
            original.load()
            self.base_image = original.convert("RGBA")
        self.base_image_path = path
        if not from_ai:
            self.ai_generation_metadata = {}
        if self.auto_layout_checkbox.isChecked():
            self.auto_arrange_text()
        else:
            self.update_preview()
        self.statusBar().showMessage("Loaded " + path)

    def update_caption_provider_note(self):
        online = self.caption_model.currentData() == "openai"
        self.caption_provider_note.setText(
            "Luna sends this image crop and brief to OpenAI when you click Suggest or Recheck (API charges apply)."
            if online else "Local model: image and brief stay on this computer.")

    def start_ai_layout(self, *, force_analysis=False):
        if self.base_image is None:
            QMessageBox.information(self, "Choose a background", "Choose the image the AI should look at first.")
            return
        if self.layout_worker is not None and self.layout_worker.isRunning():
            return
        self.capture_text_layer()
        self._layout_background = self.base_image.copy()
        self._layout_dimensions = self.preview_format.currentData()
        model_image = self.base_image
        if self._layout_dimensions:
            model_image = ImageOps.fit(model_image,self._layout_dimensions,method=Image.Resampling.LANCZOS,centering=(.5,.5))
        provider = self.caption_model.currentData() or "local"
        model = "gpt-6-luna" if provider == "openai" else self.caption_model.currentText()
        self._layout_provider = provider
        self.layout_worker = LayoutWorker(model_image, self.caption_brief.text(), model, force_analysis=force_analysis, provider=provider)
        self.layout_worker.succeeded.connect(self.preview_ai_layout)
        self.layout_worker.failed.connect(self.layout_failed)
        self.layout_worker.finished.connect(lambda: (self.caption_ai_button.setEnabled(True),self.caption_recheck_button.setEnabled(True)))
        self.caption_ai_button.setEnabled(False)
        self.caption_recheck_button.setEnabled(False)
        self.caption_ai_status.setText(("OpenAI Luna" if provider == "openai" else "Local AI") + " is preparing caption ideas. You can keep editing; nothing is replaced without Apply.")
        self.layout_worker.start()

    def layout_failed(self, reason):
        self.caption_ai_status.setText("AI suggestion unavailable: " + reason + ". Your image and text are unchanged; manual editing still works.")

    def preview_ai_layout(self, description, layers):
        # Accept legacy single-proposal calls as well as the new alternatives response.
        options = layers if layers and isinstance(layers[0],dict) else [{"name":"Suggestion", "layers":layers}]
        orientation_changed = hasattr(self,'_layout_dimensions') and self.preview_format.currentData()!=self._layout_dimensions
        if orientation_changed or self.base_image is None or self.base_image.size != self._layout_background.size or self.base_image.tobytes() != self._layout_background.tobytes():
            self.layout_failed("Background or orientation changed while the suggestion was running. Try again for the current image")
            return
        design_background=self.base_image
        dimensions=self.preview_format.currentData()
        if dimensions:
            design_background=ImageOps.fit(design_background,dimensions,method=Image.Resampling.LANCZOS,centering=(.5,.5))
        # Subject-checked proposals already have collision-free rendered bounds.
        # Re-running the generic fitter could move them onto a protected subject.
        options=[option if option.get('subject_checked') else
                 {**option,'layers':fit_caption_layers(design_background,option['layers'],self.font_map)} for option in options]
        dialog = QDialog(self)
        dialog.setWindowTitle("Preview AI suggestion")
        dialog.resize(850,700)
        layout = QVBoxLayout(dialog)
        summary = QLabel("AI image description: " + description)
        summary.setTextFormat(Qt.TextFormat.PlainText)
        summary.setWordWrap(True)
        layout.addWidget(summary)
        choices = QComboBox()
        for option in options:
            choices.addItem(option['name'])
        layout.addWidget(choices)
        original_options=copy.deepcopy(options)
        model_regions=[copy.deepcopy(option.get('protected_regions',[])) for option in options]
        user_areas=[]
        history=[]
        draw_target=None
        placement_errors={i:o['placement_error'] for i,o in enumerate(options) if o.get('placement_error')}
        preview = ProtectedAreaView()
        protect = QCheckBox("Keep an area clear")
        protect.setToolTip("Mark up to four additional areas to keep clear across these suggestions.")
        review = QCheckBox("Review subject areas")
        review.setToolTip("Inspect automatic boxes, redraw inaccurate ones, or remove false detections.")
        undo_area = QPushButton("Undo area")
        undo_area.setEnabled(False)
        area_row=QHBoxLayout();area_row.addWidget(protect);area_row.addWidget(review);area_row.addWidget(undo_area)
        layout.addLayout(area_row)
        region_controls=QWidget();region_row=QHBoxLayout(region_controls);region_row.setContentsMargins(0,0,0,0)
        region_choices=QComboBox();region_choices.setObjectName('protectedRegionChoice')
        redraw=QPushButton("Redraw area");remove=QPushButton("Remove area")
        region_row.addWidget(region_choices,1);region_row.addWidget(redraw);region_row.addWidget(remove)
        layout.addWidget(region_controls)
        def current_areas():
            return model_regions[choices.currentIndex()]+user_areas
        def show_choice():
            index=choices.currentIndex();areas=current_areas()
            selected=min(max(0,region_choices.currentIndex()),len(areas)-1)
            region_choices.blockSignals(True);region_choices.clear()
            count=len(model_regions[index])
            for i,area in enumerate(areas):
                origin='Marked' if i>=count else ('Reviewed' if area.get('reviewed') else 'AI')
                region_choices.addItem(f"{i+1}. {origin}: {area['label']}")
            region_choices.setCurrentIndex(selected);region_choices.blockSignals(False)
            region_controls.setVisible(review.isChecked())
            redraw.setEnabled(bool(areas) and not protect.isChecked())
            remove.setEnabled(bool(areas) and not protect.isChecked())
            if protect.isChecked() or review.isChecked():
                background=design_background
                if not protect.isChecked() and index not in placement_errors:
                    background=render_layers(background,options[index]['layers'],self.font_map)
                preview.show_pil(background);preview.show_areas(areas,selected)
                if protect.isChecked():
                    instruction="Drag the replacement box around the whole subject." if draw_target else "Drag over the face or object you want to keep clear."
                else:
                    instruction="Select a box or its name. Dashed edges show the full clearance margin. Redraw or remove inaccurate AI boxes."
                note.setText(placement_errors.get(index,'')+' '+instruction)
            else:
                preview.show_pil(self.add_attribution(render_layers(design_background,options[index]['layers'],self.font_map)))
                note.setText(placement_errors.get(index,"Apply uses these editable captions. Cancel keeps your current text."))
            preview.reviewing=review.isChecked()
            preview.set_selecting(protect.isChecked())
            buttons.button(QDialogButtonBox.StandardButton.Apply).setEnabled(index not in placement_errors and not protect.isChecked())
        def recompute():
            placement_errors.clear()
            for index,option in enumerate(original_options):
                regions=model_regions[index]+user_areas
                if regions==option.get('protected_regions',[]) and not option.get('placement_error'):
                    options[index]=copy.deepcopy(option)
                    continue
                try:
                    placed=place_around_subjects(design_background,option['layers'],regions)
                    options[index]={**option,'layers':placed,'subject_checked':True,'protected_regions':copy.deepcopy(regions)}
                except NoCaptionSpace:
                    placement_errors[index]="No readable caption space remains. Review the boxes, undo a change, or cancel to edit the caption manually."
            undo_area.setEnabled(bool(history))
            protect.setEnabled(len(user_areas)<4 or draw_target is not None)
            show_choice()
        def remember():
            history.append(copy.deepcopy((model_regions,user_areas)))
        def change_region(target,replacement=None):
            kind,record=target
            if kind=='user':
                position=user_areas.index(record)
                if replacement is None:user_areas.pop(position)
                else:user_areas[position]=replacement
            else:
                # Scene observations normally match across alternatives. Apply
                # the correction to matching boxes, preserving distinct ones.
                for regions in model_regions:
                    for position in range(len(regions)-1,-1,-1):
                        if regions[position]==record:
                            if replacement is None:regions.pop(position)
                            else:regions[position]=copy.deepcopy(replacement)
        def selected_target():
            index=region_choices.currentIndex();count=len(model_regions[choices.currentIndex()])
            if index<0:return None
            return ('model' if index<count else 'user',copy.deepcopy(current_areas()[index]))
        def add_area(area):
            nonlocal draw_target
            if draw_target is None and len(user_areas)>=4:return
            remember();area={**area,'reviewed':True}
            if draw_target:
                area['label']=draw_target[1]['label'];change_region(draw_target,area)
            else:user_areas.append(area)
            draw_target=None;protect.setChecked(False);recompute()
        def start_redraw():
            nonlocal draw_target
            draw_target=selected_target()
            if draw_target:
                protect.setEnabled(True);protect.setChecked(True);show_choice()
        def remove_area():
            target=selected_target()
            if target:remember();change_region(target);recompute()
        def undo():
            nonlocal draw_target
            draw_target=None;protect.setChecked(False)
            if history:
                saved_regions,saved_user=history.pop()
                model_regions[:]=saved_regions;user_areas[:]=saved_user
            recompute()
        def drawing_changed(active):
            nonlocal draw_target
            if not active:
                draw_target=None
                protect.setEnabled(len(user_areas)<4)
            show_choice()
        def choice_changed():
            nonlocal draw_target
            draw_target=None;protect.setChecked(False);region_choices.setCurrentIndex(-1);show_choice()
        preview.area_selected.connect(add_area)
        preview.area_clicked.connect(region_choices.setCurrentIndex)
        protect.toggled.connect(drawing_changed)
        review.toggled.connect(show_choice)
        undo_area.clicked.connect(undo)
        redraw.clicked.connect(start_redraw);remove.clicked.connect(remove_area)
        region_choices.currentIndexChanged.connect(show_choice)
        choices.currentIndexChanged.connect(choice_changed)
        layout.addWidget(preview,1)
        note = QLabel("Check that faces and key objects remain clear. Apply replaces your current captions with these editable suggestions. Cancel keeps your current text.")
        note.setWordWrap(True)
        note.setSizePolicy(QSizePolicy.Policy.Preferred,QSizePolicy.Policy.Minimum)
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        review.setChecked(bool(placement_errors))
        show_choice()
        if dialog.exec() == QDialog.DialogCode.Accepted and choices.currentIndex() not in placement_errors and not protect.isChecked():
            self.auto_layout_checkbox.setChecked(False)
            self.text_layers = options[choices.currentIndex()]['layers']
            self._active_layer = 0
            self.refresh_layer_choices()
            self.caption_ai_status.setText("AI caption and style applied. Font, size, position and text remain editable.")
        else:
            self.caption_ai_status.setText("Suggestion discarded. Your current captions are unchanged.")

    def on_caption_changed(self):
        if self._loading_layers:
            return
        if self.auto_layout_checkbox.isChecked():
            self.auto_arrange_text()
        else:
            self.update_preview()

    def auto_arrange_text(self):
        """Local deterministic layout assistance, not model inference."""
        if self._loading_layers or self.base_image is None:
            return
        from dataclasses import replace
        cursor = self.text_input.textCursor()
        cursor_position, cursor_anchor = cursor.position(), cursor.anchor()
        self.capture_text_layer()
        count = len(self.text_layers)
        for index, layer in enumerate(self.text_layers):
            if count == 1:
                # Prefer the quieter end of the supplied image, reserving credit space.
                gray = self.base_image.convert("L").resize((128,128))
                top = ImageStat.Stat(gray.crop((6,6,122,45))).var[0]
                bottom = ImageStat.Stat(gray.crop((6,72,122,108))).var[0]
                y = .05 if top <= bottom else .54
                available = .35
            else:
                available = .76 / count
                y = .04 + index * available
            # Fit a multiline caption into its allocated band using actual font metrics.
            size = 48
            draw = ImageDraw.Draw(self.base_image)
            while size > 10:
                font = self.font_for_path(None, max(10, round(size*self.base_image.width/512)))
                lines = self.wrapped_text(draw, layer.text, font, self.base_image.width*.9)
                if len(lines)*font.size*1.25 <= self.base_image.height*available:
                    break
                size -= 1
            self.text_layers[index] = replace(layer, x=.05, y=y, width=.9, size=size,
                align="center", color=(255,255,255,255), shadow=True, font="DejaVuSans-Bold.ttf")
        self.load_layer_controls()
        cursor = self.text_input.textCursor()
        cursor.setPosition(cursor_anchor)
        cursor.setPosition(cursor_position, QTextCursor.MoveMode.KeepAnchor)
        self.text_input.setTextCursor(cursor)
        self.statusBar().showMessage("Captions arranged.")

    def capture_text_layer(self):
        if self._loading_layers:
            return
        from dataclasses import asdict
        values = self.layer_control_values()
        previous = getattr(self, '_loaded_control_values', {})
        old = asdict(self.text_layers[self._active_layer])
        if getattr(self, '_loaded_control_index', None) == self._active_layer:
            values = {key:old[key] if previous.get(key)==value else value for key,value in values.items()}
        self.text_layers[self._active_layer] = TextLayer(**values)
        self._loaded_control_values = self.layer_control_values()
        self._loaded_control_index = self._active_layer

    def layer_control_values(self):
        return dict(text=self.text_input.toPlainText()[:20000], x=self.layer_x.value()/100,
            y=self.layer_y.value()/100, width=min(self.layer_width.value()/100,1-self.layer_x.value()/100),
            size=self.font_size.value(), font=self.font_selector.currentText() or "DejaVuSans-Bold.ttf",
            align=self.selected_alignment(), color=self.text_color, shadow=self.text_shadow)

    def select_text_layer(self, index):
        if self._loading_layers or index < 0 or index >= len(self.text_layers):
            return
        self.capture_text_layer()
        self._active_layer = index
        self.load_layer_controls()

    def load_layer_controls(self):
        self._loading_layers = True
        layer = self.text_layers[self._active_layer]
        self.text_input.setPlainText(layer.text)
        if self.font_selector.findText(layer.font) < 0:
            self.font_selector.addItem(layer.font)
        self.font_selector.setCurrentText(layer.font)
        self.font_size.setValue(layer.size)
        self.layer_x.setValue(round(layer.x*100))
        self.layer_y.setValue(round(layer.y*100))
        self.layer_width.setValue(round(layer.width*100))
        self.text_color = layer.color
        self.text_shadow = layer.shadow
        self.shadow_action.setChecked(layer.shadow)
        for button in self.align_group.buttons():
            button.setChecked(button.text().lower() == layer.align)
        self._loaded_control_values = self.layer_control_values()
        self._loaded_control_index = self._active_layer
        self._loading_layers = False
        self.update_preview()

    def refresh_layer_choices(self):
        self._loading_layers = True
        self.layer_choice.clear()
        self.layer_choice.addItems([f"Text layer {i+1}" for i in range(len(self.text_layers))])
        self.layer_choice.setCurrentIndex(self._active_layer)
        self._loading_layers = False
        self.load_layer_controls()

    def add_text_layer(self):
        if len(self.text_layers) >= 20:
            QMessageBox.warning(self, "Layer limit", "This project supports up to 20 text layers.")
            return
        self.capture_text_layer()
        self.text_layers.append(TextLayer(y=.1 if len(self.text_layers) % 2 else .7))
        self._active_layer = len(self.text_layers)-1
        self.refresh_layer_choices()

    def remove_text_layer(self):
        if self.text_input.toPlainText().strip() and QMessageBox.question(
                self, "Remove text layer", "Remove the selected text layer?") != QMessageBox.StandardButton.Yes:
            return
        del self.text_layers[self._active_layer]
        if not self.text_layers:
            self.text_layers = [TextLayer()]
        self._active_layer = min(self._active_layer, len(self.text_layers)-1)
        self.refresh_layer_choices()

    def save_editable_project(self):
        if self.base_image is None:
            QMessageBox.warning(self, "No image", "Load a background first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Editable Project", "meme.mgproject", "Meme project (*.mgproject)")
        if not path:
            return
        if not path.lower().endswith('.mgproject'):
            path += '.mgproject'
        self.capture_text_layer()
        try:
            save_project(path, self.base_image, self.text_layers, True)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Project save failed", str(exc))
            return
        self.statusBar().showMessage("Editable project saved: " + path)

    def open_editable_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Editable Project", "", "Meme project (*.mgproject)")
        if not path:
            return
        try:
            background, layers, attribution = load_project(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Project open failed", str(exc))
            return
        if self.base_image is not None and QMessageBox.question(
                self, "Replace current project", "Replace the current editor contents? Save your editable project first if needed.") != QMessageBox.StandardButton.Yes:
            return
        self._loading_layers = True
        self.base_image = background
        self.base_image_path = None
        self.ai_generation_metadata = {}
        self.text_layers = layers
        self.auto_layout_checkbox.setChecked(False)
        self._active_layer = 0
        self._loading_layers = False
        self.refresh_layer_choices()
        self.tabs.setCurrentIndex(0)
        self.statusBar().showMessage("Opened editable project: " + path)

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

    def render_meme(self, background=None):
        if self.base_image is None:
            return None
        self.capture_text_layer()
        result = render_layers(self.base_image if background is None else background, self.text_layers, self.font_map)
        return self.add_attribution(result)

    @staticmethod
    def add_attribution(image):
        """Mandatory visible original credit; never mutate the editable background."""
        validate_image(image)
        source = image.convert("RGBA")
        credit = "Generated by Meme Genie, by @kaotickj: https://github.com/kaotickj"
        width = max(240, source.width)
        size = max(10, min(20, width // 40))
        font = MemeGeneratorApp.font_for_path(None, size)
        probe = ImageDraw.Draw(Image.new("RGB",(1,1)))
        lines = MemeGeneratorApp.wrapped_text(probe, credit, font, width-16)
        line_height = size + 5
        footer_height = len(lines)*line_height + 12
        height = max(source.height, footer_height + 32)
        # Fit the entire rendered meme above the credit instead of painting over captions.
        result = Image.new("RGBA",(width,height),(15,15,15,255))
        fitted = ImageOps.contain(source,(width,height-footer_height),Image.Resampling.LANCZOS)
        result.paste(fitted,((width-fitted.width)//2,0))
        draw=ImageDraw.Draw(result)
        y=height-footer_height
        for line in lines:
            draw.text((8,y+6),line,font=font,fill="white")
            y+=line_height
        return result

    def update_preview(self, *args):
        if self._loading_layers:
            return
        rendered = self.selected_preview_image()
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

    def selected_preview_image(self):
        dimensions = self.preview_format.currentData()
        return self.sized_meme(*dimensions) if dimensions else self.render_meme()

    def sized_meme(self, width, height):
        if self.base_image is None:
            return None
        background = ImageOps.fit(self.base_image,(width,height),method=Image.Resampling.LANCZOS,centering=(.5,.5))
        return self.render_meme(background).convert("RGB")

    def generate_meme(self):
        result = self.selected_preview_image()
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
        if self.base_image is None:
            QMessageBox.warning(self, "No image", "Load or generate an image first.")
            return
        # Crop the background first, then lay out captions in output coordinates.
        # This avoids cropping captions when changing the output aspect ratio.
        sized = self.sized_meme(width,height)
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
        path, _ = QFileDialog.getSaveFileName(self, "Save Base Image with Credit", "base_image.png", "PNG (*.png)")
        if path:
            target = Path(path)
            if target.suffix.lower() != ".png":
                target = target.with_suffix(".png")
            self.add_attribution(self.base_image).save(target, format="PNG")
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
