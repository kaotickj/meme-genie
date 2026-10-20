import sys
import os
import requests
from bs4 import BeautifulSoup
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QTextEdit, QPushButton, QVBoxLayout,
    QFileDialog, QComboBox, QColorDialog, QRadioButton, QButtonGroup,
    QHBoxLayout, QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QMenuBar, QMainWindow
)
from PyQt6.QtGui import QPixmap, QImage, QIcon, QAction
from PyQt6.QtCore import Qt
from PIL import Image, ImageDraw, ImageFont


class MemeGeneratorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Meme Genie")
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon("memegenie.ico"))

        self.base_image = None
        self.base_image_path = None
        self.selected_font = "DejaVuSans-Bold.ttf"
        self.text_color = (255, 255, 255, 255)
        self.text_alignment = "center"
        self.text_shadow = True  # default ON

        self.setup_ui()
        self.showMaximized()

    def setup_ui(self):
        layout = QVBoxLayout()
        menubar = QMenuBar(self)

        # --- File Menu ---
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open Image", self)
        open_action.triggered.connect(self.choose_base_image)
        file_menu.addAction(open_action)

        save_action = QAction("Save Meme", self)
        save_action.triggered.connect(self.generate_meme)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # --- Edit Menu ---
        edit_menu = menubar.addMenu("Edit")

        font_action = QAction("Select Font", self)
        font_action.triggered.connect(lambda: self.font_selector.showPopup())
        edit_menu.addAction(font_action)

        color_action = QAction("Select Text Color", self)
        color_action.triggered.connect(self.choose_text_color)
        edit_menu.addAction(color_action)

        clear_text_action = QAction("Clear Text", self)
        clear_text_action.triggered.connect(lambda: self.text_input.clear())
        edit_menu.addAction(clear_text_action)

        toggle_shadow_action = QAction("Toggle Text Shadow", self)
        toggle_shadow_action.setCheckable(True)
        toggle_shadow_action.setChecked(self.text_shadow)
        toggle_shadow_action.triggered.connect(self.toggle_text_shadow)
        edit_menu.addAction(toggle_shadow_action)

        # --- Help Menu ---
        help_menu = menubar.addMenu("Help")

        instructions_action = QAction("Instructions", self)
        instructions_action.triggered.connect(self.show_instructions)
        help_menu.addAction(instructions_action)

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        layout.setMenuBar(menubar)

        self.image_preview = QGraphicsView()
        self.image_preview.setMinimumSize(800, 600)  # 4:3 ratio (width:height)
        self.image_preview.setMaximumHeight(600)
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scene = QGraphicsScene()
        self.image_preview.setScene(self.scene)
        layout.addWidget(self.image_preview)

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter meme text or fetch quote...")
        self.text_input.setFixedHeight(100)
        self.text_input.textChanged.connect(self.update_preview)
        layout.addWidget(self.text_input)

        self.choose_image_btn = QPushButton("Choose Base Image")
        self.choose_image_btn.clicked.connect(self.choose_base_image)
        layout.addWidget(self.choose_image_btn)

        self.fetch_quote_btn = QPushButton("Fetch Quote")
        self.fetch_quote_btn.clicked.connect(self.fetch_quote)
        layout.addWidget(self.fetch_quote_btn)

        font_label = QLabel("Select Font:")
        layout.addWidget(font_label)

        self.font_selector = QComboBox()
        self.populate_font_selector()
        self.font_selector.currentTextChanged.connect(self.on_font_select)
        self.font_selector.currentTextChanged.connect(self.update_preview)
        layout.addWidget(self.font_selector)

        self.color_btn = QPushButton("Select Text Color")
        self.color_btn.clicked.connect(self.choose_text_color)
        layout.addWidget(self.color_btn)

        alignment_layout = QHBoxLayout()
        self.align_group = QButtonGroup()
        for text, value in [("Left", "left"), ("Center", "center"), ("Right", "right")]:
            btn = QRadioButton(text)
            self.align_group.addButton(btn)
            self.align_group.setId(btn, len(self.align_group.buttons()) - 1)
            if value == "center":
                btn.setChecked(True)
            btn.toggled.connect(self.update_preview)
            alignment_layout.addWidget(btn)
        layout.addLayout(alignment_layout)

        self.generate_btn = QPushButton("Generate Meme")
        self.generate_btn.clicked.connect(self.generate_meme)
        layout.addWidget(self.generate_btn)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def toggle_text_shadow(self, checked):
        self.text_shadow = checked
        self.update_preview()

    def luminance(self, r, g, b):
        # Calculate perceived brightness
        return 0.299 * r + 0.587 * g + 0.114 * b

    def show_instructions(self):
        QMessageBox.information(self, "Instructions",
                                "1. Click 'Choose Base Image' to load an image.\n"
                                "2. Enter or paste text in the box.\n"
                                "3. Optionally fetch a quote.\n"
                                "4. Adjust font, color, and alignment.\n"
                                "5. Click 'Generate Meme' to preview and save.\n"
                                "\nYou can also access these features from the menu bar."
                                )

    def show_about(self):
        QMessageBox.information(self, "About Meme Genie",
                                "Meme Genie v1.0\n\n"
                                "Created by @kaotickj\n"
                                "GitHub: https://github.com/kaotickj\n"
                                "\nA simple yet powerful meme generation tool."
                                )

    def populate_font_selector(self):
        font_dirs = ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
                     "C:/Windows/Fonts", "/Library/Fonts", "/System/Library/Fonts"]
        self.font_map = {}
        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                for root, dirs, files in os.walk(font_dir):
                    for file in files:
                        if file.lower().endswith(".ttf"):
                            path = os.path.join(root, file)
                            self.font_map[file] = path
        for name in sorted(self.font_map):
            self.font_selector.addItem(name)
        default = "DejaVuSans-Bold.ttf"
        if default in self.font_map:
            self.font_selector.setCurrentText(default)
            self.selected_font = self.font_map[default]

    def on_font_select(self, font_name):
        self.selected_font = self.font_map.get(font_name)

    def choose_text_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.text_color = (color.red(), color.green(), color.blue(), 255)
            self.update_preview()

    def display_image_only(self):
        qt_image = QImage(self.base_image.tobytes(), self.base_image.width, self.base_image.height,
                          QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qt_image).scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio,
                                                    Qt.TransformationMode.SmoothTransformation)
        self.scene.clear()
        self.scene.addItem(QGraphicsPixmapItem(pixmap))

    def choose_base_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", os.getcwd(),
                                                   "Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        if file_path:
            self.base_image_path = file_path
            self.base_image = Image.open(file_path).convert("RGBA")
            self.display_image_only()
            self.update_preview()

    def fetch_quote(self):
        try:
            r = requests.get("https://kdgwebsolutions.com/quote.php")
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            quote = soup.find("p").text.strip()
            self.text_input.setPlainText(quote)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to retrieve quote: {e}")

    def wrap_text_line(self, text_line, font, max_width):
        """Wrap a single text line if it exceeds max_width, returning a list of wrapped lines."""
        if not text_line.strip():
            return [""]  # preserve empty line

        words = text_line.split()
        lines = []
        current_line = ""
        draw = ImageDraw.Draw(self.base_image)
        for word in words:
            test_line = f"{current_line} {word}".strip() if current_line else word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] <= max_width * 0.9:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines

    def update_preview(self):
        if not self.base_image:
            return

        text = self.text_input.toPlainText()
        if not text.strip():
            return

        meme_img = self.base_image.copy()
        draw = ImageDraw.Draw(meme_img)
        font_size = 60
        try:
            font = ImageFont.truetype(self.selected_font, font_size)
        except Exception:
            font = ImageFont.load_default()

        width, height = meme_img.size

        raw_lines = text.splitlines()
        lines = []
        for raw_line in raw_lines:
            wrapped_lines = self.wrap_text_line(raw_line, font, width)
            lines.extend(wrapped_lines)

        total_text_height = 0
        for line in lines:
            if line.strip() == "":
                bbox = draw.textbbox((0, 0), "A", font=font)
                line_height = bbox[3] - bbox[1]
                total_text_height += line_height
            else:
                bbox = draw.textbbox((0, 0), line, font=font)
                total_text_height += bbox[3] - bbox[1]

        y_offset = (height - total_text_height) // 2

        # Determine shadow color based on text brightness
        shadow_color = (0, 0, 0, 255)
        if self.text_shadow:
            if self.luminance(*self.text_color[:3]) < 128:
                shadow_color = (255, 255, 255, 255)  # White shadow for dark text
            else:
                shadow_color = (0, 0, 0, 255)  # Black shadow for light text

        for line in lines:
            if line.strip() == "":
                bbox = draw.textbbox((0, 0), "A", font=font)
                line_height = bbox[3] - bbox[1]
                y_offset += line_height
            else:
                bbox = draw.textbbox((0, 0), line, font=font)
                alignment = self.get_text_alignment()
                if alignment == "left":
                    x = 10
                elif alignment == "right":
                    x = width - bbox[2] - 10
                else:  # center
                    x = (width - bbox[2]) // 2

                # Draw shadow first if enabled
                if self.text_shadow:
                    # Draw shadow offset by 8 pixels right and down
                    shadow_pos = (x + 8, y_offset + 8)
                    draw.text(shadow_pos, line, font=font, fill=shadow_color)

                # Draw main text
                draw.text((x, y_offset), line, font=font, fill=self.text_color)
                y_offset += bbox[3] - bbox[1]

        # Add attribution text (semi-transparent white)
        attribution_text = "Generated by Meme Genie, by @kaotickj: https://github.com/kaotickj/meme-genie"
        attribution_font_size = 18
        try:
            attribution_font = ImageFont.truetype("DejaVuSans-Bold.ttf", attribution_font_size)
        except IOError:
            attribution_font = ImageFont.load_default()

        attribution_bbox = draw.textbbox((0, 0), attribution_text, font=attribution_font)
        attribution_width = attribution_bbox[2] - attribution_bbox[0]
        attribution_height = attribution_bbox[3] - attribution_bbox[1]
        attribution_x = width - attribution_width - 10
        attribution_y = height - attribution_height - 10

        draw.text((attribution_x, attribution_y), attribution_text, font=attribution_font, fill=(255, 255, 255, 128))

        qt_image = QImage(meme_img.tobytes(), meme_img.width, meme_img.height, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qt_image).scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio,
                                                    Qt.TransformationMode.SmoothTransformation)
        self.scene.clear()
        self.scene.addItem(QGraphicsPixmapItem(pixmap))

    def get_text_alignment(self):
        for btn in self.align_group.buttons():
            if btn.isChecked():
                return btn.text().lower()
        return "center"

    def generate_meme(self):
        if not self.base_image:
            QMessageBox.warning(self, "Error", "No base image selected!")
            return

        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Error", "Text cannot be empty!")
            return

        meme_img = self.base_image.copy()
        draw = ImageDraw.Draw(meme_img)
        font_size = 60
        try:
            font = ImageFont.truetype(self.selected_font, font_size)
        except Exception:
            font = ImageFont.load_default()

        width, height = meme_img.size

        raw_lines = text.splitlines()
        lines = []
        for raw_line in raw_lines:
            wrapped_lines = self.wrap_text_line(raw_line, font, width)
            lines.extend(wrapped_lines)

        total_text_height = 0
        for line in lines:
            if line.strip() == "":
                bbox = draw.textbbox((0, 0), "A", font=font)
                line_height = bbox[3] - bbox[1]
                total_text_height += line_height
            else:
                bbox = draw.textbbox((0, 0), line, font=font)
                total_text_height += bbox[3] - bbox[1]

        y_offset = (height - total_text_height) // 2

        shadow_color = (0, 0, 0, 255)
        if self.text_shadow:
            if self.luminance(*self.text_color[:3]) < 128:
                shadow_color = (255, 255, 255, 255)
            else:
                shadow_color = (0, 0, 0, 255)

        for line in lines:
            if line.strip() == "":
                bbox = draw.textbbox((0, 0), "A", font=font)
                line_height = bbox[3] - bbox[1]
                y_offset += line_height
            else:
                bbox = draw.textbbox((0, 0), line, font=font)
                alignment = self.get_text_alignment()
                if alignment == "left":
                    x = 10
                elif alignment == "right":
                    x = width - bbox[2] - 10
                else:
                    x = (width - bbox[2]) // 2

                if self.text_shadow:
                    shadow_pos = (x + 2, y_offset + 2)
                    draw.text(shadow_pos, line, font=font, fill=shadow_color)

                draw.text((x, y_offset), line, font=font, fill=self.text_color)
                y_offset += bbox[3] - bbox[1]

        attribution_text = "Generated by Meme Genie, by @kaotickj: https://github.com/kaotickj"
        attribution_font_size = 16
        try:
            attribution_font = ImageFont.truetype("DejaVuSans-Bold.ttf", attribution_font_size)
        except IOError:
            attribution_font = ImageFont.load_default()

        attribution_bbox = draw.textbbox((0, 0), attribution_text, font=attribution_font)
        attribution_width = attribution_bbox[2] - attribution_bbox[0]
        attribution_height = attribution_bbox[3] - attribution_bbox[1]
        attribution_x = width - attribution_width - 10
        attribution_y = height - attribution_height - 10

        draw.text((attribution_x, attribution_y), attribution_text, font=attribution_font, fill=(255, 255, 255, 128))

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Meme", "meme.jpg",
                                                   "JPEG Files (*.jpg);;PNG Files (*.png)")
        if save_path:
            meme_img.convert("RGB").save(save_path)
            QMessageBox.information(self, "Saved", f"Meme saved to {save_path}")

    def wrap_text(self, text, font, max_width):
        draw = ImageDraw.Draw(self.base_image)
        words = text.split()
        lines = []
        current = ""
        for word in words:
            test = f"{current} {word}".strip()
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] <= max_width * 0.9:
                current = test
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MemeGeneratorApp()
    window.show()
    sys.exit(app.exec())
