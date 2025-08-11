import os
import tensorflow as tf
import tensorflow_datasets as tfds


def _preprocess_example(image: tf.Tensor, label: tf.Tensor, num_classes: int, label_offset: int,
                        target_size=(28, 28)):
    """Resize, normalize image and one-hot encode label.

    Args:
        image: Input image tensor.
        label: Scalar label tensor.
        num_classes: Total number of classes for one-hot encoding.
        label_offset: Value to subtract from labels to make them zero-based.
        target_size: Desired image size (height, width).
    Returns:
        (image, label) tuple with image float32 in [0,1], shape (H,W,1), and one-hot label.
    """
    image = tf.image.resize(image, target_size)
    image = tf.cast(image, tf.float32) / 255.0

    # Ensure single channel
    if image.shape.rank == 2:
        image = tf.expand_dims(image, -1)
    elif image.shape.rank == 3 and image.shape[-1] != 1:
        image = tf.image.rgb_to_grayscale(image)

    # Zero-base labels if needed and one-hot encode
    label = tf.cast(label, tf.int32) - tf.cast(label_offset, tf.int32)
    label = tf.one_hot(label, depth=num_classes)
    return image, label


def get_datasets(dataset_name: str, batch_size: int = 128):
    """Load TFDS dataset (MNIST or EMNIST letters), build train/test pipelines.

    Args:
        dataset_name: 'mnist' or 'emnist/letters'.
        batch_size: Batch size for training and evaluation.
    Returns:
        train_ds, test_ds, input_shape, num_classes
    """
    builder = tfds.builder(dataset_name)
    builder.download_and_prepare()
    info = builder.info

    num_classes = info.features["label"].num_classes

    # EMNIST/letters labels are 1..26; shift to 0..25
    label_offset = 1 if dataset_name.lower().startswith("emnist/letters") else 0

    def make_split(split: str, shuffle: bool):
        ds = tfds.load(dataset_name, split=split, as_supervised=True)
        if shuffle:
            ds = ds.shuffle(buffer_size=10_000, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda x, y: _preprocess_example(x, y, num_classes, label_offset),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_split("train", shuffle=True)
    test_ds = make_split("test", shuffle=False)

    input_shape = (28, 28, 1)
    return train_ds, test_ds, input_shape, num_classes


def build_cnn_model(input_shape, num_classes: int) -> tf.keras.Model:
    """CNN with light augmentation, batch norm, and dropout for better generalization."""
    from tensorflow.keras import layers, models

    data_augmentation = models.Sequential([
        layers.RandomTranslation(0.1, 0.1, fill_mode="constant", fill_value=0.0),
        layers.RandomRotation(0.06, fill_mode="constant", fill_value=0.0),
        layers.RandomZoom(0.1, fill_mode="constant", fill_value=0.0),
        layers.RandomContrast(0.1),
    ])

    model = models.Sequential([
        layers.Input(shape=input_shape),
        data_augmentation,
        layers.Conv2D(32, 3, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_and_evaluate(dataset_name: str = "mnist", batch_size: int = 128, epochs: int = 10,
                       save_dir: str = "saved_models"):
    train_ds, test_ds, input_shape, num_classes = get_datasets(dataset_name, batch_size=batch_size)
    model = build_cnn_model(input_shape, num_classes)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, verbose=1),
    ]

    model.fit(train_ds, epochs=epochs, validation_data=test_ds, callbacks=callbacks, verbose=2)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    os.makedirs(save_dir, exist_ok=True)
    safe_name = dataset_name.replace("/", "_")
    save_path = os.path.join(save_dir, f"cnn_{safe_name}.keras")
    model.save(save_path)
    print(f"Model saved to: {save_path}")


def run_training_cli():
    # Fixed to MNIST for digits
    dataset_name = "mnist"
    batch_size = 128
    epochs = 5

    train_and_evaluate(dataset_name=dataset_name, batch_size=batch_size, epochs=epochs)


# --------------------------
# Enhanced Modern Tkinter GUI
# --------------------------
import sys
from typing import Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageOps, ImageDraw, ImageTk, ImageFilter
import numpy as np
import threading


class ModernHCRApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🎨 AI Handwritten Character Recognition")
        self.root.geometry("900x700")
        self.root.configure(bg="#1a1a2e")
        
        # Configure modern styling
        self.setup_styles()
        
        # State
        self.dataset_name = tk.StringVar(value="mnist")
        self.model: Optional[tf.keras.Model] = None
        self.current_image: Optional[Image.Image] = None
        self.is_drawing = False
        
        # Color scheme
        self.colors = {
            "primary": "#6c5ce7",
            "secondary": "#fd79a8",
            "success": "#00b894",
            "warning": "#fdcb6e",
            "danger": "#e17055",
            "dark": "#1a1a2e",
            "light": "#ffffff",
            "gray": "#74b9ff"
        }
        
        self.setup_ui()

        # Track which dataset the currently loaded model corresponds to (if known)
        self.loaded_dataset_name: Optional[str] = None
        self.training_thread: Optional[threading.Thread] = None

        # Try to auto-load a model at startup (MNIST)
        self.auto_load_model(initial=True)

    def setup_styles(self):
        """Configure modern ttk styles"""
        style = ttk.Style()
        style.theme_use("clam")
        
        # Configure button styles
        style.configure("Primary.TButton",
                       background="#6c5ce7",
                       foreground="white",
                       borderwidth=0,
                       focuscolor="none",
                       padding=(20, 10))
        style.map("Primary.TButton",
                 background=[("active", "#5f3dc4")])
        
        style.configure("Success.TButton",
                       background="#00b894",
                       foreground="white",
                       borderwidth=0,
                       focuscolor="none",
                       padding=(15, 8))
        style.map("Success.TButton",
                 background=[("active", "#00a085")])
        
        style.configure("Warning.TButton",
                       background="#fdcb6e",
                       foreground="#2d3436",
                       borderwidth=0,
                       focuscolor="none",
                       padding=(15, 8))
        style.map("Warning.TButton",
                 background=[("active", "#e17055")])

    def setup_ui(self):
        """Setup the modern UI layout"""
        # Main container
        main_frame = tk.Frame(self.root, bg=self.colors["dark"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        self.create_header(main_frame)
        
        # Content area
        content_frame = tk.Frame(main_frame, bg=self.colors["dark"])
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        # Left panel (controls)
        self.create_left_panel(content_frame)
        
        # Right panel (canvas and results)
        self.create_right_panel(content_frame)

    def create_header(self, parent):
        """Create modern header with title and dataset selection"""
        header_frame = tk.Frame(parent, bg=self.colors["primary"], height=80)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        header_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(header_frame, 
                              text="🤖 AI Handwriting Recognition",
                              font=("Helvetica", 24, "bold"),
                              fg="white",
                              bg=self.colors["primary"])
        title_label.pack(side=tk.LEFT, padx=20, pady=20)
        
        # Dataset indicator (fixed to MNIST)
        dataset_frame = tk.Frame(header_frame, bg=self.colors["primary"])
        dataset_frame.pack(side=tk.RIGHT, padx=20, pady=20)
        tk.Label(dataset_frame,
                 text="Dataset: MNIST (Digits)",
                 font=("Helvetica", 12, "bold"),
                 fg="white",
                 bg=self.colors["primary"]).pack(side=tk.LEFT)

    def create_modern_radio(self, parent, text, value):
        """Create modern-looking radio button"""
        radio_frame = tk.Frame(parent, bg=self.colors["primary"])
        radio_frame.pack(side=tk.LEFT, padx=5)
        
        radio = tk.Radiobutton(radio_frame,
                              text=text,
                              variable=self.dataset_name,
                              value=value,
                              font=("Helvetica", 10, "bold"),
                              fg="white",
                              bg=self.colors["primary"],
                              activebackground=self.colors["primary"],
                              activeforeground="white",
                              selectcolor=self.colors["secondary"],
                              borderwidth=0,
                              highlightthickness=0)
        radio.pack()

    def create_left_panel(self, parent):
        """Create left control panel"""
        left_frame = tk.Frame(parent, bg="#16213e", width=250)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        left_frame.pack_propagate(False)
        
        # Panel title
        tk.Label(left_frame,
                text="🎛️ Controls",
                font=("Helvetica", 16, "bold"),
                fg="white",
                bg="#16213e").pack(pady=20)
        
        # Model controls
        model_frame = tk.LabelFrame(left_frame,
                                   text="Model Management",
                                   font=("Helvetica", 10, "bold"),
                                   fg=self.colors["gray"],
                                   bg="#16213e")
        model_frame.pack(fill=tk.X, padx=15, pady=10)
        
        ttk.Button(model_frame,
                  text="📁 Load Model",
                  style="Primary.TButton",
                  command=self.load_model).pack(fill=tk.X, padx=10, pady=10)
        
        self.model_status = tk.Label(model_frame,
                                    text="No model loaded",
                                    font=("Helvetica", 9),
                                    fg=self.colors["warning"],
                                    bg="#16213e")
        self.model_status.pack(pady=5)
        
        # Drawing controls
        draw_frame = tk.LabelFrame(left_frame,
                                  text="Drawing Tools",
                                  font=("Helvetica", 10, "bold"),
                                  fg=self.colors["gray"],
                                  bg="#16213e")
        draw_frame.pack(fill=tk.X, padx=15, pady=10)
        
        # Brush size
        tk.Label(draw_frame,
                text="Brush Size:",
                font=("Helvetica", 9, "bold"),
                fg="white",
                bg="#16213e").pack(pady=(10, 5))
        
        self.brush_size = tk.IntVar(value=18)
        brush_scale = tk.Scale(draw_frame,
                              from_=8, to=30,
                              orient=tk.HORIZONTAL,
                              variable=self.brush_size,
                              bg="#16213e",
                              fg="white",
                              activebackground=self.colors["primary"],
                              troughcolor=self.colors["dark"],
                              highlightthickness=0)
        brush_scale.pack(fill=tk.X, padx=10, pady=5)
        
        # Action buttons
        ttk.Button(draw_frame,
                  text="🖼️ Open Image",
                  style="Success.TButton",
                  command=self.open_image).pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(draw_frame,
                  text="⚙️ Re-Train MNIST (Slow)",
                  style="Warning.TButton",
                  command=self.auto_train_if_missing).pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(draw_frame,
                  text="🗑️ Clear Canvas",
                  style="Warning.TButton",
                  command=self.clear_canvas).pack(fill=tk.X, padx=10, pady=5)

    def create_right_panel(self, parent):
        """Create right panel with canvas and results"""
        right_frame = tk.Frame(parent, bg="#16213e")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Canvas section
        canvas_frame = tk.LabelFrame(right_frame,
                                    text="✏️ Drawing Canvas",
                                    font=("Helvetica", 12, "bold"),
                                    fg=self.colors["gray"],
                                    bg="#16213e")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Canvas with border effect
        canvas_container = tk.Frame(canvas_frame, bg=self.colors["primary"], bd=3, relief=tk.RAISED)
        canvas_container.pack(pady=20)
        
        self.canvas_size = 320
        self.canvas = tk.Canvas(canvas_container,
                               width=self.canvas_size,
                               height=self.canvas_size,
                               bg="white",
                               bd=0,
                               highlightthickness=0,
                               cursor="pencil")
        self.canvas.pack(padx=5, pady=5)
        
        # PIL drawing surface
        self.pil_canvas = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.pil_draw = ImageDraw.Draw(self.pil_canvas)
        
        # Bind drawing events
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
        # Prediction section
        self.create_prediction_section(right_frame)

    def create_prediction_section(self, parent):
        """Create prediction results section"""
        pred_frame = tk.LabelFrame(parent,
                                  text="🔮 Prediction Results",
                                  font=("Helvetica", 12, "bold"),
                                  fg=self.colors["gray"],
                                  bg="#16213e")
        pred_frame.pack(fill=tk.X, padx=15, pady=(0, 10))
        
        # Predict button
        predict_btn = ttk.Button(pred_frame,
                                text="🚀 Predict Character",
                                style="Primary.TButton",
                                command=self.predict)
        predict_btn.pack(pady=15)
        
        # Results display
        results_frame = tk.Frame(pred_frame, bg="#16213e")
        results_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        # Main prediction
        self.prediction_label = tk.Label(results_frame,
                                        text="Draw something and click Predict!",
                                        font=("Helvetica", 18, "bold"),
                                        fg=self.colors["gray"],
                                        bg="#16213e")
        self.prediction_label.pack(pady=10)
        
        # Confidence
        self.confidence_label = tk.Label(results_frame,
                                        text="",
                                        font=("Helvetica", 12),
                                        fg=self.colors["secondary"],
                                        bg="#16213e")
        self.confidence_label.pack()
        
        # Progress bar for visual feedback
        self.progress = ttk.Progressbar(results_frame,
                                       mode='indeterminate',
                                       length=200)
        self.progress.pack(pady=10)

    def start_draw(self, event):
        """Start drawing"""
        self.is_drawing = True
        self.draw(event)

    def draw(self, event):
        """Draw on canvas"""
        if not self.is_drawing:
            return
            
        x, y = event.x, event.y
        r = self.brush_size.get() // 2
        
        # Draw on Tk canvas with smooth circles
        self.canvas.create_oval(x - r, y - r, x + r, y + r,
                               fill="black", outline="black", width=0)
        
        # Draw on PIL surface
        self.pil_draw.ellipse((x - r, y - r, x + r, y + r), fill=0)

    def stop_draw(self, event):
        """Stop drawing"""
        self.is_drawing = False

    def clear_canvas(self):
        """Clear the drawing canvas with animation effect"""
        self.canvas.delete("all")
        self.pil_canvas.paste(255)
        self.current_image = None
        
        # Reset prediction display
        self.prediction_label.config(text="Canvas cleared! Draw something new.",
                                    fg=self.colors["gray"])
        self.confidence_label.config(text="")
        
        # Brief visual feedback
        self.canvas.config(bg="#f0f0f0")
        self.root.after(200, lambda: self.canvas.config(bg="white"))

    def load_model(self):
        """Load model with improved feedback"""
        safe_name = self.dataset_name.get().replace("/", "_")
        default_path = os.path.join("saved_models", f"cnn_{safe_name}.keras")
        
        if os.path.exists(default_path):
            model_path = default_path
        else:
            model_path = filedialog.askopenfilename(
                title="Select trained model file",
                filetypes=[("Keras Model", "*.keras"), ("All Files", "*.*")]
            )
            if not model_path:
                return
        
        try:
            # Show loading progress
            self.progress.start()
            self.model_status.config(text="Loading model...", fg=self.colors["warning"])
            self.root.update()
            
            self._load_model_silent(model_path)
            
            # Success feedback
            self.progress.stop()
            self.model_status.config(text="✅ Model loaded successfully!", 
                                   fg=self.colors["success"])
            self.flash_success()
        except Exception as e:
            self.progress.stop()
            self.model_status.config(text="❌ Failed to load model", 
                                   fg=self.colors["danger"])
            messagebox.showerror("Load Error", f"Failed to load model:\n{str(e)}")

    def open_image(self):
        """Open image file with preview"""
        file_path = filedialog.askopenfilename(
            title="Open image file",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("All Files", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        try:
            img = Image.open(file_path).convert("L")
            self.current_image = img
            
            # Clear canvas and show preview
            self.clear_canvas()
            self.prediction_label.config(text="Image loaded! Click Predict to analyze.",
                                       fg=self.colors["success"])
            
        except Exception as e:
            messagebox.showerror("Open Error", f"Failed to open image:\n{str(e)}")

    def preprocess_image(self, img: Image.Image, is_canvas: bool = False) -> np.ndarray:
        """Preprocess image to mimic MNIST formatting.

        Steps:
        - Convert to grayscale array
        - Decide inversion so foreground (digit) is bright on dark background
        - Threshold to find bounding box
        - Resize longest side to 20 and keep aspect ratio
        - Paste into 28x28 black canvas centered; shift by center of mass
        - Normalize to [0,1]
        """
        # Grayscale array in [0,1]
        img_gray = img.convert("L")
        arr_uint8 = np.array(img_gray, dtype=np.uint8)
        arr_norm = arr_uint8.astype(np.float32) / 255.0

        # Orientation handling
        if is_canvas:
            # Our canvas has white background and black ink → always invert to get bright digit on dark
            work = 1.0 - arr_norm
        else:
            # External images: decide by mean
            work = 1.0 - arr_norm if arr_norm.mean() > 0.5 else arr_norm

        # Binarize for bbox
        # Binary mask using a stable threshold for strokes
        bw = work > 0.2
        coords = np.argwhere(bw)
        if coords.size == 0:
            # No strokes detected; return zeros
            return np.zeros((1, 28, 28, 1), dtype=np.float32)

        y0, x0 = coords.min(0)
        y1, x1 = coords.max(0) + 1
        crop = work[y0:y1, x0:x1]

        # Resize keeping aspect: longest side -> 20
        h, w = crop.shape
        scale = 20.0 / float(max(h, w))
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        crop_img = Image.fromarray((crop * 255).astype(np.uint8))
        # Light blur to reduce aliasing on thin strokes
        crop_img = crop_img.filter(ImageFilter.GaussianBlur(radius=0.5))
        crop_img = crop_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        crop_arr = np.array(crop_img, dtype=np.float32) / 255.0

        # Paste into 28x28 black canvas
        canvas = np.zeros((28, 28), dtype=np.float32)
        y_offset = (28 - new_h) // 2
        x_offset = (28 - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = crop_arr

        # Center of mass shift (clip to canvas to avoid wrap-around artifacts)
        mass = canvas
        total = float(mass.sum())
        if total > 0:
            y_indices = np.arange(28, dtype=np.float32)[:, None]
            x_indices = np.arange(28, dtype=np.float32)[None, :]
            cy = float((mass * y_indices).sum() / total)
            cx = float((mass * x_indices).sum() / total)
            dy = int(round(14 - cy))
            dx = int(round(14 - cx))
            if dy != 0 or dx != 0:
                shifted = np.zeros_like(canvas)
                y_src0 = max(0, -dy)
                y_src1 = min(28, 28 - dy)
                x_src0 = max(0, -dx)
                x_src1 = min(28, 28 - dx)
                y_dst0 = max(0, dy)
                y_dst1 = y_dst0 + (y_src1 - y_src0)
                x_dst0 = max(0, dx)
                x_dst1 = x_dst0 + (x_src1 - x_src0)
                shifted[y_dst0:y_dst1, x_dst0:x_dst1] = canvas[y_src0:y_src1, x_src0:x_src1]
                canvas = shifted

        return canvas.reshape((1, 28, 28, 1))

    def predict(self):
        """Make prediction with enhanced feedback"""
        if self.model is None:
            messagebox.showwarning("No Model", 
                                 "Please load a trained model first using the 'Load Model' button.")
            return
        
        # Determine source image
        source_img = self.current_image if self.current_image else self.pil_canvas
        
        try:
            # Show progress
            self.progress.start()
            self.prediction_label.config(text="🤔 Thinking...", fg=self.colors["warning"])
            self.confidence_label.config(text="Processing your input...")
            self.root.update()
            
            # Preprocess and predict
            batch = self.preprocess_image(source_img, is_canvas=(source_img is self.pil_canvas))
            predictions = self.model.predict(batch, verbose=0)[0]
            
            # Get results
            pred_idx = int(np.argmax(predictions))
            confidence = float(predictions[pred_idx])

            # Anti-collapse heuristic: if model collapses to 0 with high confidence on obviously thin/large strokes,
            # slightly penalize confidence to surface alternatives in UI
            sample = batch[0, :, :, 0]
            active = float((sample > 0.2).mean())
            if pred_idx == 0 and confidence > 0.9 and (0.03 < active < 0.55):
                confidence *= 0.8
            
            # Format prediction
            if self.dataset_name.get() == "mnist":
                predicted_char = str(pred_idx)
                char_type = "Digit"
            else:
                predicted_char = chr(ord('A') + pred_idx)
                char_type = "Letter"
            
            # Stop progress and show results
            self.progress.stop()
            
            # Color-coded confidence
            if confidence > 0.95:
                conf_color = self.colors["success"]
                emoji = "🎯"
            elif confidence > 0.75:
                conf_color = self.colors["warning"]
                emoji = "🤔"
            else:
                conf_color = self.colors["danger"]
                emoji = "😅"
            
            self.prediction_label.config(
                text=f"{emoji} {char_type}: {predicted_char}",
                fg="white",
                font=("Helvetica", 24, "bold")
            )
            
            self.confidence_label.config(
                text=f"Confidence: {confidence:.1%}",
                fg=conf_color,
                font=("Helvetica", 14, "bold")
            )
            
            # Flash effect for high confidence, only if not 0 to avoid reinforcing bad bias
            if confidence > 0.95 and pred_idx != 0:
                self.flash_success()
                
        except Exception as e:
            self.progress.stop()
            self.prediction_label.config(text="❌ Prediction failed", fg=self.colors["danger"])
            self.confidence_label.config(text="")
            messagebox.showerror("Prediction Error", f"Failed to run prediction:\n{str(e)}")

    def flash_success(self):
        """Visual success feedback"""
        original_bg = self.root.cget("bg")
        self.root.configure(bg=self.colors["success"])
        self.root.after(150, lambda: self.root.configure(bg=original_bg))

    # --------------------------
    # Auto-load helpers
    # --------------------------
    def _default_model_path_for(self, dataset_name: str) -> str:
        safe = dataset_name.replace("/", "_")
        return os.path.join("saved_models", f"cnn_{safe}.keras")

    def _scan_saved_models(self) -> list[str]:
        models_dir = "saved_models"
        if not os.path.isdir(models_dir):
            return []
        candidates = []
        for name in os.listdir(models_dir):
            if name.lower().endswith(".keras"):
                full = os.path.join(models_dir, name)
                candidates.append(full)
        # Sort by modified time, newest first
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates

    def _infer_dataset_from_filename(self, path: str) -> Optional[str]:
        lower = os.path.basename(path).lower()
        if "emnist_letters" in lower:
            return "emnist/letters"
        if "mnist" in lower:
            return "mnist"
        return None

    def auto_load_model(self, initial: bool = False):
        """Attempt to automatically load a model for the selected dataset.

        Strategy:
        1) If default path for current dataset exists, load it.
        2) Else, pick the newest .keras under saved_models matching the dataset in filename.
        3) Else, pick the newest .keras under saved_models (any), and keep current dataset toggle.
        """
        # Avoid reloading if already have a model that matches selected dataset
        selected = self.dataset_name.get()
        if self.model is not None and self.loaded_dataset_name == selected:
            return

        default_path = self._default_model_path_for(selected)
        try:
            if os.path.exists(default_path):
                # Silent load
                self.model_status.config(text="Loading model...", fg=self.colors["warning"])
                self.root.update()
                self._load_model_silent(default_path)
                self.model_status.config(text="✅ Model auto-loaded", fg=self.colors["success"])
                return

            # Fallback: scan directory
            candidates = self._scan_saved_models()
            if not candidates:
                # Nothing to load -> auto-train MNIST model
                if initial:
                    self.auto_train_if_missing()
                return

            # Prefer candidate matching dataset in filename
            preferred = None
            for c in candidates:
                inferred = self._infer_dataset_from_filename(c)
                if inferred == selected:
                    preferred = c
                    break
            chosen = preferred or candidates[0]

            # Load chosen
            self.model_status.config(text="Loading model...", fg=self.colors["warning"])
            self.root.update()
            self._load_model_silent(chosen)
            self.model_status.config(text=f"✅ Auto-loaded: {os.path.basename(chosen)}", fg=self.colors["success"])
        except Exception:
            # Stay quiet on startup errors
            self.model_status.config(text="Auto-load failed. Use 'Load Model'", fg=self.colors["danger"])

    def _load_model_silent(self, model_path: str):
        model = tf.keras.models.load_model(model_path)
        inferred = self._infer_dataset_from_filename(model_path) or self.dataset_name.get()
        self.model = model
        self.loaded_dataset_name = inferred

    def auto_train_if_missing(self):
        if self.training_thread and self.training_thread.is_alive():
            return
        self.model_status.config(text="No model found. Training MNIST model...", fg=self.colors["warning"])
        self.progress.start()

        def worker():
            try:
                # Train and save MNIST model with stronger model & longer epochs
                train_and_evaluate(dataset_name="mnist", batch_size=128, epochs=10, save_dir="saved_models")
                path = self._default_model_path_for("mnist")
                # After training, load model on UI thread
                def finalize():
                    try:
                        self._load_model_silent(path)
                        self.progress.stop()
                        self.model_status.config(text="✅ Model trained and loaded", fg=self.colors["success"])
                        self.flash_success()
                    except Exception:
                        self.progress.stop()
                        self.model_status.config(text="❌ Auto-train load failed", fg=self.colors["danger"])
                self.root.after(0, finalize)
            except Exception:
                def fail():
                    self.progress.stop()
                    self.model_status.config(text="❌ Auto-train failed", fg=self.colors["danger"])
                self.root.after(0, fail)

        self.training_thread = threading.Thread(target=worker, daemon=True)
        self.training_thread.start()


def run_ui():
    """Run the enhanced UI"""
    root = tk.Tk()
    app = ModernHCRApp(root)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (900 // 2)
    y = (root.winfo_screenheight() // 2) - (700 // 2)
    root.geometry(f"900x700+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel("ERROR")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        run_training_cli()
    else:
        run_ui()
