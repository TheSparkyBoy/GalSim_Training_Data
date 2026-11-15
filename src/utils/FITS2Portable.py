import sys
import os
import glob
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, 
    QPushButton, QFileDialog, QComboBox, QProgressBar, 
    QVBoxLayout, QHBoxLayout, QGridLayout, QTextEdit, QMessageBox
)
from PyQt6.QtCore import QThread, pyqtSignal
from astropy.io.fits import open as fits_open
import numpy as np
from PIL import Image

class ConversionWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(int, int) # success_count, fail_count

    def __init__(self, src_dir, out_dir, fmt, p_low, p_high):
        super().__init__()
        self.src_dir = src_dir
        self.out_dir = out_dir
        self.fmt = fmt.lower()
        self.p_low = p_low
        self.p_high = p_high

    def run(self):
        # Match common FITS file extensions (.fits, .fit, .FITS)
        search_pattern = os.path.join(self.src_dir, "*.[fF][iI][tT]*")
        files = glob.glob(search_pattern)
        total = len(files)
        success = 0
        fail = 0

        if total == 0:
            self.progress.emit(100, "No FITS files found in the selected directory.")
            self.finished.emit(0, 0)
            return

        os.makedirs(self.out_dir, exist_ok=True)

        for i, filepath in enumerate(files):
            filename = os.path.basename(filepath)
            base_name = os.path.splitext(filename)[0]
            out_path = os.path.join(self.out_dir, f"{base_name}.{self.fmt}")

            try:
                with fits_open(filepath) as hdul:
                    # Find the first HDU with 2D or 3D image data
                    data = None
                    for hdu in hdul:
                        if hdu.data is not None and hdu.data.ndim >= 2:
                            data = hdu.data
                            break
                    
                    if data is None:
                        raise ValueError("No valid 2D/3D image data found in FITS HDU.")

                    # Handle 3D data (e.g., RGB FITS) by taking the first slice or collapsing
                    if data.ndim == 3:
                        data = data[0]

                    data = np.nan_to_num(data.astype(np.float32))
                    
                    # Apply percentile-based clipping & normalization (stretch)
                    vmin, vmax = np.percentile(data, (self.p_low, self.p_high))
                    clipped = np.clip(data, vmin, vmax)
                    
                    if vmax > vmin:
                        normalized = (clipped - vmin) / (vmax - vmin) * 255.0
                    else:
                        normalized = np.zeros_like(clipped)
                    
                    img_8bit = normalized.astype(np.uint8)
                    
                    # FITS coordinates start bottom-left; standard images start top-left
                    img_8bit = np.flipud(img_8bit)

                    # Save image
                    img = Image.fromarray(img_8bit)
                    if self.fmt == 'jpg':
                        img.save(out_path, quality=95)
                    else:
                        img.save(out_path)
                        
                    success += 1
                    self.progress.emit(int((i + 1) / total * 100), f"Converted: {filename} -> {self.fmt.upper()}")
            except Exception as e:
                fail += 1
                self.progress.emit(int((i + 1) / total * 100), f"Failed {filename}: {str(e)}")

        self.finished.emit(success, fail)

class FitsConverterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FITS to PNG/JPG Batch Converter")
        self.resize(700, 520)
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        grid = QGridLayout()

        # Source Directory Selection
        grid.addWidget(QLabel("Source Directory:"), 0, 0)
        self.src_input = QLineEdit()
        grid.addWidget(self.src_input, 0, 1)
        src_btn = QPushButton("Browse...")
        src_btn.clicked.connect(self.browse_src)
        grid.addWidget(src_btn, 0, 2)

        # Output Directory Selection
        grid.addWidget(QLabel("Output Directory:"), 1, 0)
        self.out_input = QLineEdit()
        grid.addWidget(self.out_input, 1, 1)
        out_btn = QPushButton("Browse...")
        out_btn.clicked.connect(self.browse_out)
        grid.addWidget(out_btn, 1, 2)

        # Format Dropdown
        grid.addWidget(QLabel("Output Format:"), 2, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "JPG"])
        grid.addWidget(self.format_combo, 2, 1, 1, 2)

        # Stretch Low Percentile
        grid.addWidget(QLabel("Stretch Low (%):"), 3, 0)
        self.p_low_input = QLineEdit("1.0")
        grid.addWidget(self.p_low_input, 3, 1, 1, 2)

        # Stretch High Percentile
        grid.addWidget(QLabel("Stretch High (%):"), 4, 0)
        self.p_high_input = QLineEdit("99.0")
        grid.addWidget(self.p_high_input, 4, 1, 1, 2)

        layout.addLayout(grid)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Conversion Log Terminal
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Conversion logs will appear here...")
        layout.addWidget(self.log_box)

        # Action Button
        self.convert_btn = QPushButton("Start Batch Conversion")
        self.convert_btn.clicked.connect(self.start_conversion)
        layout.addWidget(self.convert_btn)

    def browse_src(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Source Directory with FITS Files")
        if dir_name:
            self.src_input.setText(dir_name)
            # Auto-suggest an output folder inside the source directory
            if not self.out_input.text():
                self.out_input.setText(os.path.join(dir_name, "converted_output"))

    def browse_out(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_name:
            self.out_input.setText(dir_name)

    def start_conversion(self):
        src = self.src_input.text().strip()
        out = self.out_input.text().strip()
        fmt = self.format_combo.currentText()

        if not src or not os.path.isdir(src):
            QMessageBox.warning(self, "Invalid Input", "Please select a valid source directory.")
            return
        if not out:
            QMessageBox.warning(self, "Invalid Input", "Please specify an output directory.")
            return

        try:
            p_low = float(self.p_low_input.text())
            p_high = float(self.p_high_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Stretch percentiles must be valid numbers (e.g., 1.0 and 99.0).")
            return

        self.convert_btn.setEnabled(False)
        self.log_box.clear()
        self.progress_bar.setValue(0)

        # Run conversion in a separate thread to keep UI fluid
        self.worker = ConversionWorker(src, out, fmt, p_low, p_high)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.conversion_finished)
        self.worker.start()

    def update_progress(self, val, msg):
        self.progress_bar.setValue(val)
        self.log_box.append(msg)

    def conversion_finished(self, success, fail):
        self.convert_btn.setEnabled(True)
        QMessageBox.information(
            self, 
            "Batch Complete", 
            f"Conversion finished!\nSuccessfully converted: {success}\nFailed: {fail}"
        )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FitsConverterApp()
    window.show()
    sys.exit(app.exec())