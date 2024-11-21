import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import requests
from bs4 import BeautifulSoup
import os

class MemeGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Meme Genie")

        self.base_image = None
        self.selected_font = "DejaVuSans-Bold.ttf"
        self.text_color = (255, 255, 255, 255)  # Default white color
        self.text_alignment = "center"  # Default text alignment
        self.text_shadow = True  # Default text shadow is enabled

        # Label for the base image selection
        self.image_label = tk.Label(root, text="No image selected", width=40, height=5)
        self.image_label.grid(row=0, column=0, padx=10, pady=10)

        # Label for the text input
        self.text_label = tk.Label(root, text="Enter text for the meme or fetch a quote:")
        self.text_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        # Text area widget for the meme text
        self.text_entry = tk.Text(root, width=40, height=5, wrap="word")
        self.text_entry.grid(row=2, column=0, padx=10, pady=10)

        # Button to choose the base image
        self.choose_image_button = tk.Button(root, text="Choose Base Image", command=self.choose_base_image)
        self.choose_image_button.grid(row=3, column=0, padx=10, pady=10)

        # Button to fetch a quote
        self.fetch_quote_button = tk.Button(root, text="Fetch Quote", command=self.fetch_quote)
        self.fetch_quote_button.grid(row=4, column=0, padx=10, pady=10)

        # Button to generate the meme
        self.generate_button = tk.Button(root, text="Generate Meme", command=self.generate_meme)
        self.generate_button.grid(row=5, column=0, padx=10, pady=10)

        # Font selection dropdown
        self.font_selector_label = tk.Label(root, text="Select Font:")
        self.font_selector_label.grid(row=6, column=0, padx=10, pady=5, sticky="w")
        self.font_selector = ttk.Combobox(root, values=["DejaVuSans-Bold.ttf", "Arial", "Comic Sans MS"], width=40)
        self.font_selector.set(self.selected_font)
        self.font_selector.grid(row=7, column=0, padx=10, pady=10)
        self.font_selector.bind("<<ComboboxSelected>>", self.on_font_select)

        # Text color picker button
        self.color_button = tk.Button(root, text="Select Text Color", command=self.choose_text_color)
        self.color_button.grid(row=8, column=0, padx=10, pady=10)

        # Text alignment options
        self.alignment_label = tk.Label(root, text="Select Text Alignment:")
        self.alignment_label.grid(row=9, column=0, padx=10, pady=5, sticky="w")
        self.alignment_var = tk.StringVar(value="center")
        self.alignment_left = tk.Radiobutton(root, text="Left", variable=self.alignment_var, value="left")
        self.alignment_left.grid(row=10, column=0, padx=10, pady=5, sticky="w")
        self.alignment_center = tk.Radiobutton(root, text="Center", variable=self.alignment_var, value="center")
        self.alignment_center.grid(row=11, column=0, padx=10, pady=5, sticky="w")
        self.alignment_right = tk.Radiobutton(root, text="Right", variable=self.alignment_var, value="right")
        self.alignment_right.grid(row=12, column=0, padx=10, pady=5, sticky="w")

        # Text shadow toggle
        # self.shadow_checkbox = tk.Checkbutton(root, text="Enable Text Shadow", variable=self.text_shadow, onvalue=True, offvalue=False)
        # self.shadow_checkbox.grid(row=13, column=0, padx=10, pady=5)

    def choose_base_image(self):
        file_path = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select an Image",
            filetypes=[
                ("JPEG files", "*.jpg"),
                ("JPEG files", "*.jpeg"),
                ("PNG files", "*.png"),
                ("BMP files", "*.bmp"),
                ("GIF files", "*.gif"),
                ("All Files", "*.*")  # Option for All Files
            ]

        )
        if file_path:
            try:
                self.base_image = Image.open(file_path)
                self.base_image = self.base_image.convert("RGBA")
                self.display_image(self.base_image)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open image: {e}")

    def display_image(self, img):
        img = img.resize((200, 400))  # Resize the image for display in the GUI
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.config(image=img_tk, text="")
        self.image_label.image = img_tk

    def fetch_quote(self):
        url = "https://kdgwebsolutions.com/quote.php"
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            quote = soup.find("p").get_text(strip=True)
            self.text_entry.delete(1.0, tk.END)
            self.text_entry.insert("1.0", quote)
        except requests.RequestException as e:
            messagebox.showerror("Error", f"Failed to retrieve quote: {e}")

    def choose_text_color(self):
        color = colorchooser.askcolor()[1]
        if color:
            self.text_color = self.hex_to_rgb(color)

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)) + (255,)

    def on_font_select(self, event):
        self.selected_font = self.font_selector.get()

    def get_max_font_size(self, meme_image, text):
        width, height = meme_image.size
        max_font_size = 100
        draw = ImageDraw.Draw(meme_image)
        
        while True:
            try:
                font = ImageFont.truetype(self.selected_font, max_font_size)
            except IOError:
                return 40

            bbox = draw.textbbox((0, 0), text, font=font)
            if bbox[2] > width * 0.9 or bbox[3] > height * 0.9:
                break
            max_font_size += 10
        return max_font_size - 10

    def wrap_text(self, text, font, image_width):
        draw = ImageDraw.Draw(self.base_image)
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] <= image_width * 0.9:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines

    def generate_meme(self):
        if not self.base_image:
            messagebox.showerror("Error", "No base image selected!")
            return

        text = self.text_entry.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Error", "Text cannot be empty!")
            return

        meme_image = self.base_image.copy()
        draw = ImageDraw.Draw(meme_image)
        
        max_font_size = self.get_max_font_size(meme_image, text)
        try:
            font = ImageFont.truetype(self.selected_font, max_font_size)
        except IOError:
            font = ImageFont.load_default()

        width, height = meme_image.size
        lines = self.wrap_text(text, font, width)
        total_text_height = sum([draw.textbbox((0, 0), line, font=font)[3] for line in lines])
        y_offset = (height - total_text_height) // 2

        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            if self.text_shadow:
                # Draw shadow with slight offset
                shadow_offset = 5
                shadow_color = (0, 0, 0, 128)  # Semi-transparent black for shadow
                if self.text_alignment == "left":
                    x_offset_shadow = 10 + shadow_offset
                elif self.text_alignment == "right":
                    x_offset_shadow = width - bbox[2] - 10 + shadow_offset
                else:  # Default to center
                    x_offset_shadow = (width - bbox[2]) // 2 + shadow_offset
                draw.text((x_offset_shadow, y_offset + shadow_offset), line, font=font, fill=shadow_color)

            if self.text_alignment == "left":
                x_offset = 10
            elif self.text_alignment == "right":
                x_offset = width - bbox[2] - 10
            else:  # Default to center
                x_offset = (width - bbox[2]) // 2
            draw.text((x_offset, y_offset), line, font=font, fill=self.text_color)

            y_offset += bbox[3]

        # Add attribution text
        attribution_text = "Generated by Meme Gen, by @kaotickj: https://github.com/kaotickj"
        attribution_font_size = 16  # Smaller font size for attribution
        try:
            attribution_font = ImageFont.truetype("DejaVuSans-Bold.ttf", attribution_font_size)
        except IOError:
            attribution_font = ImageFont.load_default()  # Fallback if the font isn't found

        # Calculate attribution position (bottom-right corner)
        text_width, text_height = draw.textbbox((0, 0), attribution_text, font=attribution_font)[2:4]
        attribution_x = width - text_width - 10  # 10px padding from the right
        attribution_y = height - text_height - 10  # 10px padding from the bottom

        # Draw the attribution text
        draw.text((attribution_x, attribution_y), attribution_text, font=attribution_font, fill=(255, 255, 255, 128))  # Semi-transparent white

        meme_image = meme_image.convert("RGB")
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if save_path:
            meme_image.save(save_path)
            messagebox.showinfo("Success", f"Meme saved to {save_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MemeGeneratorApp(root)
    root.mainloop()
