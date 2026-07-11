# This code is meant to be run on Windows/Linux with PyQt5 installed.

import sys
import os
import pandas as pd
import numpy as np
import scipy.ndimage as ndi
from astropy.io import fits
from astropy.coordinates import Angle
from astropy import units as u

# --- HIGH-SPEED RENDERING ENGINE ---
# Replaced slow Matplotlib canvas with blazing-fast OpenCV C++ pixel buffers
import cv2

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QLineEdit, QFileDialog, QTableWidget, 
    QTableWidgetItem, QHeaderView, QMessageBox, QGroupBox, QGridLayout, 
    QStyleFactory, QCheckBox
)
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtCore import Qt

class FitsTelemetryExtractor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZWO FITS Hardware Telemetry & Manifest Generator")
        self.setGeometry(150, 150, 1300, 720)
        
        self.telemetry_data = []
        self.init_ui()

    def init_ui(self):
        # 1. Directory Selection Group
        file_group = QGroupBox("1. Select Hardware Target Directory & Output Options")
        file_layout = QGridLayout()
        
        self.dir_input = QLineEdit()
        self.dir_input.setPlaceholderText("Select folder containing .fits images...")
        dir_btn = QPushButton("Browse Folder")
        dir_btn.clicked.connect(self.browse_directory)
        
        self.out_input = QLineEdit()
        self.out_input.setText("dataset_manifest.csv")
        
        # Checkbox for High-Speed PNG Generation
        self.chk_save_png = QCheckBox("Generate Annotated PNGs with Detected Stars Circled (High-Speed OpenCV Engine)")
        self.chk_save_png.setChecked(True)
        self.chk_save_png.setStyleSheet("font-weight: bold; color: #0277BD; margin-top: 5px;")
        
        file_layout.addWidget(QLabel("Target Directory:"), 0, 0)
        file_layout.addWidget(self.dir_input, 0, 1)
        file_layout.addWidget(dir_btn, 0, 2)
        
        file_layout.addWidget(QLabel("Output CSV Name:"), 1, 0)
        file_layout.addWidget(self.out_input, 1, 1)
        
        file_layout.addWidget(self.chk_save_png, 2, 0, 1, 3)
        file_group.setLayout(file_layout)
        
        # 2. Execution Trigger
        self.run_btn = QPushButton("Scan Directory & Generate Universal Manifest")
        self.run_btn.setFixedHeight(38)
        self.run_btn.setStyleSheet("background-color: #0277BD; color: white; font-weight: bold; font-size: 14px;")
        self.run_btn.clicked.connect(self.run_extraction)
        self.run_btn.setEnabled(False) 
        
        # 3. KPI Summary Banner
        kpi_layout = QHBoxLayout()
        self.lbl_found = self._create_kpi_card("FITS Files Found", "0", "#1565C0")
        self.lbl_success = self._create_kpi_card("Manifest Rows Generated", "0", "#2E7D32")
        self.lbl_failed = self._create_kpi_card("Read Errors", "0", "#C62828")
        
        kpi_layout.addWidget(self.lbl_found)
        kpi_layout.addWidget(self.lbl_success)
        kpi_layout.addWidget(self.lbl_failed)
        
        # 4. Results Ledger Table (Mirrors all 21 GalSim Manifest Columns)
        self.table = QTableWidget()
        self.table.setColumnCount(12)
        self.table.setHorizontalHeaderLabels([
            "Image ID", "Group", "RA (deg)", "DEC (deg)", "Roll", "Focal (mm)", 
            "Exp (s)", "Stars", "Med SNR", "BG Mean", "BG Std", "Anomalies"
        ])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        
        # Master Layout Assembly
        main_layout = QVBoxLayout()
        main_layout.addWidget(file_group)
        main_layout.addWidget(self.run_btn)
        main_layout.addLayout(kpi_layout)
        main_layout.addWidget(QLabel("<b>Generated Universal Manifest Ledger:</b>"))
        main_layout.addWidget(self.table)
        
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def _create_kpi_card(self, title, val, color):
        lbl = QLabel(f"<div style='text-align: center;'><small>{title}</small><br><b style='font-size: 20px; color: {color};'>{val}</b></div>")
        lbl.setFrameStyle(QLabel.StyledPanel | QLabel.Raised)
        lbl.setStyleSheet("background-color: #f8f9fa; padding: 6px; border-radius: 4px; border: 1px solid #ddd;")
        return lbl

    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory with FITS Files", os.path.expanduser("~"))
        if directory:
            self.dir_input.setText(directory)
            self._check_ready_state()

    def _check_ready_state(self):
        if self.dir_input.text() and self.out_input.text():
            self.run_btn.setEnabled(True)

    def _parse_coordinate_to_decimal(self, val, is_ra=True):
        if val == 'NA' or val is None or pd.isna(val):
            return "NA"
        try:
            return str(round(float(val), 4))
        except ValueError:
            try:
                unit = u.hourangle if is_ra else u.deg
                angle = Angle(str(val), unit=unit)
                return str(round(angle.deg, 4))
            except Exception as e:
                print(f"[WARNING] Could not parse astrometry value '{val}': {e}")
                return str(val)

    def _save_annotated_png_fast(self, img_float, star_positions, output_path, bg_mean, bg_std):
        """
        OPENCV HIGH-SPEED RENDERING ENGINE:
        Directly manipulates numpy pixel arrays and uses C++ primitives to draw
        circles and write to disk, avoiding Matplotlib's heavy canvas overhead.
        """
        try:
            # 1. Calculate dynamic astronomical stretch limits
            vmin = bg_mean - (1.0 * bg_std)
            vmax = bg_mean + (10.0 * bg_std) if bg_std > 0 else np.max(img_float)
            if vmax <= vmin:
                vmax = vmin + 1.0
                
            # 2. Normalize array to 0-255 uint8 grayscale image
            img_norm = np.clip((img_float - vmin) / (vmax - vmin), 0.0, 1.0)
            img_uint8 = (img_norm * 255.0).astype(np.uint8)
            
            # 3. Convert Grayscale to BGR color space so we can draw RED circles
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
            
            # 4. Draw anti-aliased red circles around every star's (y, x) centroid
            # In OpenCV, color is BGR format: (0, 0, 255) = Pure Red
            for (y, x) in star_positions:
                cv2.circle(img_bgr, (int(round(x)), int(round(y))), radius=12, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                
            # 5. Flip vertically because FITS files store coordinates bottom-to-top, while PNGs are top-to-bottom
            img_bgr_flipped = cv2.flip(img_bgr, 0)
                
            # 6. Save to disk instantly using native C++ encoder
            cv2.imwrite(output_path, img_bgr_flipped)
        except Exception as e:
            print(f"[WARNING] Failed to save fast PNG to {output_path}: {e}")

    def _calculate_merline_howell_telemetry(self, img_data, png_output_path=None):
        """
        MERLINE & HOWELL (1995) INTEGRATED SNR ENGINE (Equation 6.1):
        Calculates robust background statistics and evaluates true integrated SNR
        across the full pixel footprint (n_px) of every detected star or streak.
        """
        if img_data is None:
            return "NA", "NA", "NA", "NA"
        try:
            img_float = img_data.astype(np.float32)
            total_pixels = img_float.size
            
            # 1. Robust Background Statistics using MAD (Median Absolute Deviation)
            bg_mean = float(np.median(img_float))
            mad = float(np.median(np.abs(img_float - bg_mean)))
            bg_std = float(1.4826 * mad)
            
            if bg_std == 0:
                bg_std = float(np.std(img_float))
                if bg_std == 0: bg_std = 1.0 # Prevent division by zero
                
            # Total per-pixel background variance: (N_s + N_d + N_r^2 + G^2 * sigma_f^2)
            bg_variance = bg_std ** 2
            
            # 2. 5-Sigma Detection Threshold
            threshold = bg_mean + (5.0 * bg_std)
            
            # 3. Isolate contiguous pixel islands above the threshold
            structure = np.ones((3, 3), dtype=int)
            labeled_array, num_stars = ndi.label(img_float > threshold, structure=structure)
            
            star_positions = []
            if num_stars > 0:
                # Get centroids for optional OpenCV 1:1 PNG circle drawing
                star_positions = ndi.maximum_position(img_float, labeled_array, index=range(1, num_stars + 1))
                if isinstance(star_positions, tuple):
                    star_positions = [star_positions]
                
                # --- EQUATION 6.1 VECTORIZED IMPLEMENTATION ---
                # n_px: Number of pixels in each detected star/streak island
                n_px = np.array(ndi.sum(np.ones_like(img_float), labeled_array, index=range(1, num_stars + 1)))
                
                # N_star: Total integrated object electrons accumulated over n_px pixels
                net_signal_img = img_float - bg_mean
                N_star = np.array(ndi.sum(net_signal_img, labeled_array, index=range(1, num_stars + 1)))
                
                # n_B: Number of background pixels used to estimate bg_mean
                total_star_pixels = np.sum(n_px)
                n_B = max(1.0, float(total_pixels - total_star_pixels))
                
                # Calculate the radicand (terms inside the square root)
                bg_term = n_px * (1.0 + (n_px / n_B)) * bg_variance
                radicand = np.maximum(1e-6, N_star + bg_term)
                
                # S / N = N_* / sqrt(N_* + n_px * (1 + n_px / n_B) * sigma_bg^2)
                star_snrs = N_star / np.sqrt(radicand)
                
                median_snr = float(np.median(star_snrs))
            else:
                median_snr = 0.0
                
            # Trigger high-speed OpenCV annotated PNG export if enabled
            if png_output_path and hasattr(self, '_save_annotated_png_fast'):
                self._save_annotated_png_fast(img_float, star_positions, png_output_path, bg_mean, bg_std)
                
            # Returns formatted strings for GUI Table & Manifest CSV
            return str(num_stars), str(round(median_snr, 2)), str(round(bg_mean, 2)), str(round(bg_std, 2))
        except Exception as e:
            print(f"[WARNING] Merline & Howell calculation failed: {e}")
            return "NA", "NA", "NA", "NA"
        
    def run_extraction(self):
        target_directory = self.dir_input.text()
        output_csv_name = self.out_input.text()
        save_pngs = self.chk_save_png.isChecked()
        
        if not os.path.exists(target_directory):
            QMessageBox.critical(self, "Error", "The selected directory does not exist.")
            return

        self.run_btn.setText("Scanning & Generating Manifest... Please Wait")
        self.run_btn.setEnabled(False)
        QApplication.processEvents()
        
        try:
            fits_files = [f for f in os.listdir(target_directory) if f.lower().endswith(('.fits', '.fit'))]
            
            if not fits_files:
                QMessageBox.information(self, "No Files", f"No .fits or .fit files found in:\n{target_directory}")
                self.run_btn.setText("Scan Directory & Generate Universal Manifest")
                self.run_btn.setEnabled(True)
                return

            png_dir = os.path.join(target_directory, "png")
            if save_pngs:
                os.makedirs(png_dir, exist_ok=True)

            self.telemetry_data = []
            self.table.setRowCount(0)
            
            self.lbl_found.setText(self._create_kpi_card("FITS Files Found", str(len(fits_files)), "#1565C0").text())
            QApplication.processEvents()

            success_cnt = 0
            failed_cnt = 0
            dataset_group_name = os.path.basename(os.path.normpath(target_directory))
            
            raw_ids = [os.path.splitext(f)[0] for f in fits_files]
            generate_unique_ids = len(set(raw_ids)) != len(fits_files)

            for idx, filename in enumerate(fits_files):
                filepath = os.path.join(target_directory, filename)
                try:
                    with fits.open(filepath, mode='readonly') as hdul:
                        header = hdul[0].header
                        img_data = hdul[0].data if len(hdul) > 0 else None
                        
                        if generate_unique_ids:
                            img_id = f"{idx+1:07d}"
                        else:
                            img_id = os.path.splitext(filename)[0]

                        raw_ra = header.get('OBJCTRA', header.get('RA', header.get('CRVAL1', 'NA')))
                        raw_dec = header.get('OBJCTDEC', header.get('DEC', header.get('CRVAL2', 'NA')))
                        
                        ra_val = self._parse_coordinate_to_decimal(raw_ra, is_ra=True)
                        dec_val = self._parse_coordinate_to_decimal(raw_dec, is_ra=False)
                        
                        roll_val = header.get('ROLL', header.get('ROTANG', header.get('CROTA2', header.get('PA', '0.0'))))
                        focal_val = header.get('FOCALLEN', header.get('FOCLEN', 'NA'))
                        exp_val = header.get('EXPTIME', 'NA')
                        
                        png_path = os.path.join(png_dir, f"{img_id}.png") if save_pngs else None
                        
                        # --- EXECUTE HIGH-SPEED DETECTION & RENDERING ---
                        stars_val, snr_val, bg_mean_val, bg_std_val = self._calculate_merline_howell_telemetry(img_data, png_path)
                        
                        file_info = {
                            'image_id': str(img_id),
                            'dataset_group': dataset_group_name,
                            'ra': str(ra_val),
                            'dec': str(dec_val),
                            'camera_roll': str(roll_val),
                            'focal_length_mm': str(focal_val),
                            'exposure_time_s': str(exp_val),
                            'total_stars': str(stars_val),
                            'median_image_snr': str(snr_val),
                            'bg_mean_e': str(bg_mean_val),
                            'bg_std_e': str(bg_std_val),
                            'distorted_stars': 'NA',
                            'dropped_stars': 'NA',
                            'false_stars': 'NA',
                            'smear_px': 'NA',
                            'anom_lens_on': False,
                            'anom_false_on': False,
                            'anom_drop_on': False,
                            'anom_pos_on': False,
                            'anom_mag_on': False,
                            'anom_smear_on': False
                        }
                        
                        self.telemetry_data.append(file_info)
                        
                        row_idx = self.table.rowCount()
                        self.table.insertRow(row_idx)
                        
                        items = [
                            QTableWidgetItem(file_info['image_id']),
                            QTableWidgetItem(file_info['dataset_group']),
                            QTableWidgetItem(file_info['ra']),
                            QTableWidgetItem(file_info['dec']),
                            QTableWidgetItem(file_info['camera_roll']),
                            QTableWidgetItem(file_info['focal_length_mm']),
                            QTableWidgetItem(file_info['exposure_time_s']),
                            QTableWidgetItem(file_info['total_stars']),
                            QTableWidgetItem(file_info['median_image_snr']),
                            QTableWidgetItem(file_info['bg_mean_e']),
                            QTableWidgetItem(file_info['bg_std_e']),
                            QTableWidgetItem("None (Physical Capture)")
                        ]
                        
                        for col_idx, item in enumerate(items):
                            item.setTextAlignment(Qt.AlignCenter)
                            self.table.setItem(row_idx, col_idx, item)
                            
                        success_cnt += 1
                        
                except Exception as e:
                    failed_cnt += 1
                    print(f"Failed to read {filename}: {e}")

                if (success_cnt + failed_cnt) % 10 == 0:
                    QApplication.processEvents()

            self.lbl_success.setText(self._create_kpi_card("Manifest Rows Generated", str(success_cnt), "#2E7D32").text())
            self.lbl_failed.setText(self._create_kpi_card("Read Errors", str(failed_cnt), "#C62828" if failed_cnt > 0 else "#607D8B").text())
            
            if self.telemetry_data:
                galsim_columns = [
                    'image_id', 'dataset_group', 'ra', 'dec', 'camera_roll', 
                    'focal_length_mm', 'exposure_time_s', 'total_stars', 
                    'median_image_snr', 'bg_mean_e', 'bg_std_e', 'distorted_stars', 
                    'dropped_stars', 'false_stars', 'smear_px', 'anom_lens_on', 
                    'anom_false_on', 'anom_drop_on', 'anom_pos_on', 'anom_mag_on', 'anom_smear_on'
                ]
                df = pd.DataFrame(self.telemetry_data)[galsim_columns]
                output_path = os.path.join(target_directory, output_csv_name)
                df.to_csv(output_path, index=False)
                
                msg = f"Successfully generated universal manifest for {success_cnt} physical captures.\n\nManifest saved to:\n{output_path}"
                if save_pngs:
                    msg += f"\n\nAnnotated PNGs saved to:\n{png_dir}"
                QMessageBox.information(self, "Manifest Complete", msg)

        except Exception as e:
            QMessageBox.critical(self, "System Error", f"A fatal error occurred:\n{str(e)}")
        
        finally:
            self.run_btn.setText("Scan Directory & Generate Universal Manifest")
            self.run_btn.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    window = FitsTelemetryExtractor()
    window.show()
    sys.exit(app.exec_())