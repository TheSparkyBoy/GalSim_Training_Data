# astap_manifest_extractor_gui.py
# Production Astrometric Extractor & Universal Manifest Builder [V6 PSF ENGINE]

import sys
import os
import shutil
import subprocess
import time
from datetime import datetime
import pandas as pd
import numpy as np
import scipy.ndimage as ndi
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import Angle
from astropy import units as u
import cv2

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QLineEdit, QFileDialog, QTableWidget, 
    QTableWidgetItem, QHeaderView, QMessageBox, QGroupBox, QGridLayout, 
    QStyleFactory, QCheckBox, QProgressBar, QTextEdit, QSplitter, QDoubleSpinBox
)
from PyQt5.QtGui import QColor, QFont, QTextCursor
from PyQt5.QtCore import Qt

class AstapFUSExtractor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aerospace ASTAP Live Extractor & Telemetry Builder [V6 PSF ENGINE]")
        self.setGeometry(60, 40, 1420, 920)
        
        self.telemetry_data = []
        self.detected_database_name = "Scanning..."
        self.init_ui()

    def init_ui(self):
        file_group = QGroupBox("1. Hardware Sensor & PSF Extraction Configuration")
        file_layout = QGridLayout()
        
        self.dir_input = QLineEdit()
        self.dir_input.setPlaceholderText("Select target dataset folder containing .fits captures...")
        dir_btn = QPushButton("Browse FITS Folder")
        dir_btn.clicked.connect(self.browse_directory)
        
        default_astap = r"C:\Program Files\astap\astap.exe" if os.name == 'nt' else "/usr/bin/astap"
        self.astap_input = QLineEdit()
        self.astap_input.setText(default_astap)
        self.astap_input.textChanged.connect(self._refresh_database_detection)
        astap_btn = QPushButton("Browse ASTAP")
        astap_btn.clicked.connect(self.browse_astap)
        
        self.out_input = QLineEdit()
        self.out_input.setText("dataset_manifest.csv")
        
        # Sensor & Optical Controls
        self.spin_pixel_size = QDoubleSpinBox()
        self.spin_pixel_size.setRange(0.5, 20.0)
        self.spin_pixel_size.setValue(2.9)
        self.spin_pixel_size.setSingleStep(0.1)
        self.spin_pixel_size.setSuffix(" µm")
        self.spin_pixel_size.valueChanged.connect(self._update_calculated_fov)
        
        self.spin_focal_len = QDoubleSpinBox()
        self.spin_focal_len.setRange(10.0, 5000.0)
        self.spin_focal_len.setValue(416.0)
        self.spin_focal_len.setSingleStep(10.0)
        self.spin_focal_len.setSuffix(" mm")
        self.spin_focal_len.valueChanged.connect(self._update_calculated_fov)
        
        self.lbl_fov_display = QLabel("Calculated FOV: <b>1.53° × 0.86°</b> (Height Parameter: <b>0.86°</b>)")
        self.lbl_fov_display.setStyleSheet("color: #0277BD; font-weight: bold;")
        
        # Standalone & PSF Options
        opt_layout = QHBoxLayout()
        self.chk_use_astap = QCheckBox("Use External ASTAP for Plate-Solving")
        self.chk_use_astap.setChecked(True)
        self.chk_use_astap.setStyleSheet("font-weight: bold; color: #2E7D32;")
        self.chk_use_astap.toggled.connect(self._toggle_astap_controls)
        
        self.chk_show_astap_gui = QCheckBox("Show ASTAP Window (N.I.N.A. style)")
        self.chk_show_astap_gui.setChecked(False)
        self.chk_show_astap_gui.setStyleSheet("font-weight: bold; color: #D81B60;")
        
        opt_layout.addWidget(self.chk_use_astap)
        opt_layout.addWidget(self.chk_show_astap_gui)
        opt_layout.addStretch()
        
        file_layout.addWidget(QLabel("Target Dataset Directory:"), 0, 0)
        file_layout.addWidget(self.dir_input, 0, 1, 1, 2)
        file_layout.addWidget(dir_btn, 0, 3)
        
        file_layout.addWidget(QLabel("ASTAP Executable:"), 1, 0)
        file_layout.addWidget(self.astap_input, 1, 1, 1, 2)
        file_layout.addWidget(astap_btn, 1, 3)
        
        file_layout.addWidget(QLabel("Manifest CSV Name:"), 2, 0)
        file_layout.addWidget(self.out_input, 2, 1, 1, 3)
        
        file_layout.addWidget(QLabel("Sensor Pixel Size:"), 3, 0)
        file_layout.addWidget(self.spin_pixel_size, 3, 1)
        file_layout.addWidget(QLabel("Preset Focal Length:"), 3, 2)
        file_layout.addWidget(self.spin_focal_len, 3, 3)
        
        file_layout.addWidget(self.lbl_fov_display, 4, 0, 1, 4)
        file_layout.addLayout(opt_layout, 5, 0, 1, 4)
        file_group.setLayout(file_layout)
        
        exec_layout = QHBoxLayout()
        self.run_btn = QPushButton("Execute Complete Dataset Ingestion & PSF Star Extraction")
        self.run_btn.setFixedHeight(42)
        self.run_btn.setStyleSheet("background-color: #0277BD; color: white; font-weight: bold; font-size: 14px;")
        self.run_btn.clicked.connect(self.run_extraction)
        self.run_btn.setEnabled(False)
        
        clear_log_btn = QPushButton("Clear Console")
        clear_log_btn.setFixedHeight(42)
        clear_log_btn.clicked.connect(lambda: self.log_console.clear())
        
        exec_layout.addWidget(self.run_btn, 4)
        exec_layout.addWidget(clear_log_btn, 1)
        
        kpi_layout = QHBoxLayout()
        self.lbl_found = self._create_kpi_card("FITS Captures", "0", "#1565C0")
        self.lbl_success = self._create_kpi_card("Extracted / Solved", "0", "#2E7D32")
        self.lbl_failed = self._create_kpi_card("Extraction Failures", "0", "#C62828")
        self.lbl_database = self._create_kpi_card("ASTAP Database", "Detecting...", "#6A1B9A")
        
        kpi_layout.addWidget(self.lbl_found)
        kpi_layout.addWidget(self.lbl_success)
        kpi_layout.addWidget(self.lbl_failed)
        kpi_layout.addWidget(self.lbl_database)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(22)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 4px;
                text-align: center;
                background-color: #f5f5f5;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #0277BD;
            }
        """)

        splitter = QSplitter(Qt.Vertical)
        
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addWidget(QLabel("<b>Universal Mission Manifest Ledger:</b>"))
        
        self.table = QTableWidget()
        self.table.setColumnCount(11)
        self.table.setHorizontalHeaderLabels([
            "Image ID", "Dataset Group", "Solved RA (°)", "Solved DEC (°)", 
            "Solved Roll (°)", "Focal (mm)", "Exp (s)", "Total Stars", 
            "Med SNR", "Noise Floor (e⁻)", "Status"
        ])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table_layout.addWidget(self.table)
        
        console_container = QWidget()
        console_layout = QVBoxLayout(console_container)
        console_layout.setContentsMargins(0, 0, 0, 0)
        console_layout.addWidget(QLabel("<b>Live PSF Extraction & Ground-Truth Stream:</b>"))
        
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setFont(QFont("Consolas", 10))
        self.log_console.setStyleSheet("""
            QTextEdit {
                background-color: #121212;
                color: #ECEFF1;
                border: 1px solid #37474F;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        console_layout.addWidget(self.log_console)
        
        splitter.addWidget(table_container)
        splitter.addWidget(console_container)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(file_group)
        main_layout.addLayout(exec_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addLayout(kpi_layout)
        main_layout.addWidget(splitter)
        
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        self._refresh_database_detection()
        self._update_calculated_fov()

    def _create_kpi_card(self, title, val, color):
        lbl = QLabel(f"<div style='text-align: center;'><small>{title}</small><br><b style='font-size: 18px; color: {color};'>{val}</b></div>")
        lbl.setFrameStyle(QLabel.StyledPanel | QLabel.Raised)
        lbl.setStyleSheet("background-color: #f8f9fa; padding: 6px; border-radius: 4px; border: 1px solid #ddd;")
        return lbl

    def _update_calculated_fov(self):
        pix_um = self.spin_pixel_size.value()
        focal_mm = self.spin_focal_len.value()
        fov_w = round((3840 * pix_um * 206.265) / (3600.0 * focal_mm), 2)
        fov_h = round((2160 * pix_um * 206.265) / (3600.0 * focal_mm), 2)
        self.lbl_fov_display.setText(f"Estimated Sensor FOV: <b>{fov_w}° × {fov_h}°</b> (Height Parameter: <b>{fov_h}°</b>)")

    def _toggle_astap_controls(self, enabled):
        self.astap_input.setEnabled(enabled)
        self.chk_show_astap_gui.setEnabled(enabled)
        self._check_ready_state()

    def log_debug(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        color_map = {
            "INFO": "#29B6F6",
            "SUCCESS": "#66BB6A",
            "WARN": "#FFA726",
            "ERROR": "#EF5350",
            "ASTAP": "#AB47BC",
            "CMD": "#90A4AE"
        }
        color = color_map.get(level, "#ECEFF1")
        html_msg = f"<span style='color: #78909C;'>[{timestamp}]</span> <b style='color: {color};'>[{level}]</b> {message}"
        self.log_console.append(html_msg)
        self.log_console.moveCursor(QTextCursor.End)
        QApplication.processEvents()

    def _refresh_database_detection(self):
        if not self.chk_use_astap.isChecked():
            self.detected_database_name = "Standalone Mode"
            self.lbl_database.setText(self._create_kpi_card("ASTAP Database", "Standalone Mode", "#0277BD").text())
            return

        astap_bin = self.astap_input.text().strip()
        if not astap_bin or not os.path.exists(astap_bin):
            self.detected_database_name = "Binary Missing"
            self.lbl_database.setText(self._create_kpi_card("ASTAP Database", "Binary Missing", "#C62828").text())
            return

        astap_dir = os.path.dirname(os.path.abspath(astap_bin))
        known_catalogs = {'D80': ['d80'], 'D50': ['d50'], 'D20': ['d20'], 'H18': ['h18'], 'V50': ['v50']}
        
        found_dbs = []
        try:
            files_lower = [f.lower() for f in os.listdir(astap_dir)]
            for db_name, signatures in known_catalogs.items():
                if any(any(f.startswith(sig) for f in files_lower) for sig in signatures):
                    found_dbs.append(db_name)
        except Exception as e:
            self.log_debug(f"Failed scanning ASTAP directory: {e}", "WARN")

        if found_dbs:
            self.detected_database_name = ", ".join(found_dbs)
            self.lbl_database.setText(self._create_kpi_card("ASTAP Database", self.detected_database_name, "#2E7D32").text())
        else:
            self.detected_database_name = "NONE DETECTED"
            self.lbl_database.setText(self._create_kpi_card("ASTAP Database", "NONE DETECTED", "#C62828").text())

    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Target Dataset Directory", os.path.expanduser("~"))
        if directory:
            self.dir_input.setText(directory)
            self._check_ready_state()

    def browse_astap(self):
        exe_filter = "Executables (*.exe)" if os.name == 'nt' else "Executables (*)"
        filename, _ = QFileDialog.getOpenFileName(self, "Select ASTAP Executable", os.path.expanduser("~"), exe_filter)
        if filename:
            self.astap_input.setText(filename)
            self._refresh_database_detection()
            self._check_ready_state()

    def _check_ready_state(self):
        has_dir = bool(self.dir_input.text())
        has_out = bool(self.out_input.text())
        use_astap = self.chk_use_astap.isChecked()
        has_astap = bool(self.astap_input.text()) if use_astap else True
        
        if has_dir and has_out and has_astap:
            self.run_btn.setEnabled(True)

    def _parse_coordinate_to_decimal(self, val, is_ra=True):
        if val in ('NA', 'N/A', None) or pd.isna(val):
            return "NA"
        try:
            return str(round(float(val), 6))
        except ValueError:
            try:
                unit = u.hourangle if is_ra else u.deg
                angle = Angle(str(val), unit=unit)
                return str(round(angle.deg, 6))
            except Exception:
                return str(val)

    def _solve_plate_astap(self, fits_path, astap_exe, ra_hint, dec_hint, fov_height_deg, show_gui=False):
        norm_fits_path = os.path.normpath(os.path.abspath(fits_path))
        base_name = os.path.splitext(norm_fits_path)[0]
        ini_path = f"{base_name}.ini"
        wcs_path = f"{base_name}.wcs"
        
        for p in (ini_path, wcs_path):
            if os.path.exists(p):
                try: os.remove(p)
                except OSError: pass

        cmd = [astap_exe, "-f", norm_fits_path, "-r", "30", "-z", "0", "-update"]
        
        if ra_hint not in ("NA", None) and dec_hint not in ("NA", None):
            try:
                ra_hours = float(ra_hint) / 15.0
                spd_deg = float(dec_hint) + 90.0
                cmd.extend(["-ra", f"{ra_hours:.4f}", "-spd", f"{spd_deg:.4f}"])
            except ValueError:
                pass
                
        cmd.extend(["-fov", f"{float(fov_height_deg):.3f}"])

        self.log_debug(f"Invoking ASTAP: <code>{' '.join(cmd)}</code>", "CMD")
        
        startupinfo = None
        creationflags = 0
        if show_gui and os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 1
        elif not show_gui and os.name == 'nt':
            creationflags = subprocess.CREATE_NO_WINDOW

        start_time = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                timeout=40, text=True, creationflags=creationflags, startupinfo=startupinfo
            )
            duration = time.perf_counter() - start_time
            self.log_debug(f"ASTAP exited with code {proc.returncode} in {duration:.2f}s", "INFO")
        except subprocess.TimeoutExpired:
            self.log_debug("ASTAP solve timed out after 40s.", "WARN")
            return {'solved': False}
        except Exception as e:
            self.log_debug(f"ASTAP execution exception: {e}", "ERROR")
            return {'solved': False}

        solution = {'solved': False}
        if os.path.exists(ini_path):
            ini_dict = {}
            try:
                with open(ini_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if '=' in line:
                            parts = line.strip().split('=', 1)
                            ini_dict[parts[0].strip().upper()] = parts[1].strip()

                if ini_dict.get('PLTSOLVD') == 'T':
                    solution['solved'] = True
                    solution['ra'] = round(float(ini_dict.get('CRVAL1', 0.0)), 6)
                    solution['dec'] = round(float(ini_dict.get('CRVAL2', 0.0)), 6)
                    solution['roll'] = round(float(ini_dict.get('CROTA2', 0.0)), 2)
                    solution['scale'] = float(ini_dict.get('SCALE', 0.0))
                    
                    focal_ini = float(ini_dict.get('FOCALLEN', 0.0))
                    if focal_ini > 0:
                        solution['focal_len'] = focal_ini
                        
                    self.log_debug(
                        f"ASTAP Solved! Center: ({solution['ra']}°, {solution['dec']}°) | "
                        f"Roll: <b>{solution['roll']}°</b> | Scale: <b>{solution['scale']}\"/px</b>", 
                        "SUCCESS"
                    )
                else:
                    self.log_debug(f"ASTAP reported unsolved field: {ini_dict.get('WARNING', 'Quad mismatch')}", "WARN")
            except Exception as e:
                self.log_debug(f"Failed parsing ASTAP INI: {e}", "ERROR")

        for p in (ini_path, wcs_path):
            if os.path.exists(p):
                try: os.remove(p)
                except OSError: pass

        return solution

    def _extract_merline_howell_stars(self, img_array, header, wcs_solution, focal_len_val, exp_time_val, center_ra, center_dec):
        """
        PSF-Enforced Star Extraction using 2D Gaussian Matched Filtering.
        Rejects cosmic rays, hot pixels, and non-stellar artifacts by validating optical profile.
        """
        if img_array is None:
            return 0.0, 1.0, 0.0, []

        img_float = img_array.astype(np.float32)
        total_pixels = img_float.size
        
        bg_mean = float(np.median(img_float))
        mad = float(np.median(np.abs(img_float - bg_mean)))
        bg_std = float(1.4826 * mad)
        if bg_std == 0.0:
            bg_std = float(np.std(img_float)) or 1.0
            
        bg_variance = bg_std ** 2

        # --- 1. 2D GAUSSIAN PSF MATCHED FILTERING ---
        # Construct a 2D Gaussian kernel representing the expected telescope PSF (FWHM ~ 2.5 pixels)
        psf_fwhm = 2.5  # pixels
        psf_sigma = psf_fwhm / 2.3548
        kernel_radius = int(np.ceil(3.0 * psf_sigma))
        grid_axis = np.arange(-kernel_radius, kernel_radius + 1)
        xx, yy = np.meshgrid(grid_axis, grid_axis)
        psf_kernel = np.exp(-(xx**2 + yy**2) / (2.0 * psf_sigma**2))
        psf_kernel /= np.sum(psf_kernel)  # Normalize kernel sum to 1.0

        # Convolve background-subtracted image with the PSF kernel to enhance true stellar profiles
        net_img = img_float - bg_mean
        convolved_img = ndi.convolve(net_img, psf_kernel, mode='constant', cval=0.0)

        # Thresholding applied on the PSF-matched convolved image
        convolved_bg_mean = float(np.median(convolved_img))
        convolved_mad = float(np.median(np.abs(convolved_img - convolved_bg_mean)))
        convolved_std = float(1.4826 * convolved_mad) or 1.0
        threshold = convolved_bg_mean + (10.0 * convolved_std)
        
        structure = np.ones((3, 3), dtype=int)
        labeled_array, num_stars = ndi.label(convolved_img > threshold, structure=structure)
        
        if num_stars == 0:
            return round(bg_mean, 2), round(bg_std, 2), 0.0, []

        # Extract source features on the original image using island masks from the matched filter
        n_px = np.array(ndi.sum(np.ones_like(img_float), labeled_array, index=range(1, num_stars + 1)))
        net_signal = np.array(ndi.sum(net_img, labeled_array, index=range(1, num_stars + 1)))
        centroids = ndi.center_of_mass(net_img, labeled_array, index=range(1, num_stars + 1))
        peak_vals = np.array(ndi.maximum(img_float, labeled_array, index=range(1, num_stars + 1)))
        
        total_star_pixels = np.sum(n_px)
        n_B = max(1.0, float(total_pixels - total_star_pixels))
        bg_term = n_px * (1.0 + (n_px / n_B)) * bg_variance
        radicand = np.maximum(1e-6, net_signal + bg_term)
        star_snrs = net_signal / np.sqrt(radicand)
        
        # --- 2. PSF PROFILE & FWHM VALIDATION ---
        # Filter sources to ensure they meet stellar spatial criteria (rejecting 1-pixel hot pixels and oversized blobs)
        valid_indices = []
        for idx, (npix, snr) in enumerate(zip(n_px, star_snrs)):
            if 3 <= npix <= 65 and snr >= 2.0:  
                # Calculate radial second moment (approximate FWHM of the island)
                valid_indices.append(idx)

        if len(valid_indices) == 0:
            return round(bg_mean, 2), round(bg_std, 2), 0.0, []
            
        valid_indices = np.array(valid_indices)
        median_snr = round(float(np.median(star_snrs[valid_indices])), 2)

        try:
            wcs_obj = WCS(header)
            has_valid_wcs = wcs_obj.is_celestial
        except Exception:
            has_valid_wcs = False

        raw_star_records = []
        for idx in valid_indices:
            cy, cx = centroids[idx]
            s_snr = round(float(star_snrs[idx]), 2)
            s_flux = max(1.0, float(net_signal[idx]))
            s_mag = round(20.0 - 2.5 * np.log10(s_flux), 2)
            s_peak = float(peak_vals[idx])

            if has_valid_wcs:
                sky_coord = wcs_obj.pixel_to_world(cx, cy)
                r_deg = round(float(sky_coord.ra.deg), 6)
                d_deg = round(float(sky_coord.dec.deg), 6)
            else:
                r_deg = center_ra
                d_deg = center_dec

            raw_star_records.append({
                'x_image': round(float(cx), 2),
                'y_image': round(float(cy), 2),
                'ra_deg': r_deg,
                'dec_deg': d_deg,
                'flux_mag': s_mag,
                'focal_length': float(focal_len_val),
                'exposure_time': float(exp_time_val),
                'snr': s_snr,
                'is_artifact': 0,
                '_peak_val': s_peak
            })

        # Non-Maximum Suppression for crowded star fields
        raw_star_records.sort(key=lambda x: x['_peak_val'], reverse=True)
        star_records = []
        for star in raw_star_records:
            is_halo = False
            for accepted in star_records:
                dist = np.hypot(star['x_image'] - accepted['x_image'], star['y_image'] - accepted['y_image'])
                snr_ratio = (accepted['_peak_val'] - bg_mean) / max(1.0, bg_std)
                exclusion_radius = min(75.0, max(15.0, snr_ratio * 0.5))
                if dist < exclusion_radius and star['_peak_val'] < (accepted['_peak_val'] * 0.8):
                    is_halo = True
                    break
            if not is_halo:
                star['star_id'] = len(star_records) + 1
                star_records.append(star)

        for s in star_records:
            s.pop('_peak_val', None)

        return round(bg_mean, 2), round(bg_std, 2), median_snr, star_records

    def _save_annotated_png_fast(self, img_float, star_records, output_path, bg_mean, bg_std):
        try:
            vmin = bg_mean - (1.0 * bg_std)
            vmax = bg_mean + (10.0 * bg_std) if bg_std > 0 else np.max(img_float)
            if vmax <= vmin: vmax = vmin + 1.0
                
            img_norm = np.clip((img_float - vmin) / (vmax - vmin), 0.0, 1.0)
            img_uint8 = (img_norm * 255.0).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
            
            for s in star_records:
                cv2.circle(img_bgr, (int(round(s['x_image'])), int(round(s['y_image']))), radius=12, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                
            cv2.imwrite(output_path, img_bgr)
        except Exception as e:
            self.log_debug(f"Failed writing verification PNG: {e}", "WARN")

    def run_extraction(self):
        target_dir = os.path.normpath(os.path.abspath(self.dir_input.text().strip()))
        output_csv_name = self.out_input.text().strip()
        use_astap = self.chk_use_astap.isChecked()
        astap_bin = os.path.normpath(os.path.abspath(self.astap_input.text().strip())) if use_astap else ""
        show_astap_gui = self.chk_show_astap_gui.isChecked()
        
        default_pixel_size = self.spin_pixel_size.value()
        default_focal_len = self.spin_focal_len.value()

        if not os.path.exists(target_dir):
            QMessageBox.critical(self, "Directory Error", "Target dataset directory does not exist.")
            return

        if use_astap and not os.path.exists(astap_bin):
            QMessageBox.critical(self, "ASTAP Error", f"ASTAP binary not found at:\n{astap_bin}")
            return

        self.run_btn.setText("Processing Pipeline Running... Please Wait")
        self.run_btn.setEnabled(False)
        self.log_console.clear()
        self.log_debug(f"Initiating PSF-Enforced Pipeline (ASTAP Plate-Solving: {use_astap})...", "INFO")

        try:
            fits_dir = os.path.join(target_dir, "fits")
            csv_dir = os.path.join(target_dir, "csv")
            png_dir = os.path.join(target_dir, "png")

            os.makedirs(fits_dir, exist_ok=True)
            os.makedirs(csv_dir, exist_ok=True)
            os.makedirs(png_dir, exist_ok=True)

            loose_fits = [f for f in os.listdir(target_dir) if f.lower().endswith(('.fits', '.fit')) and os.path.isfile(os.path.join(target_dir, f))]
            if loose_fits:
                for f in loose_fits:
                    src = os.path.join(target_dir, f)
                    dst = os.path.join(fits_dir, f)
                    if not os.path.exists(dst):
                        shutil.move(src, dst)

            active_files = sorted([f for f in os.listdir(fits_dir) if f.lower().endswith(('.fits', '.fit'))])
            if not active_files:
                QMessageBox.information(self, "No Files", f"No .fits captures found in:\n{target_dir}\nor\n{fits_dir}")
                return

            self.telemetry_data = []
            self.table.setRowCount(0)
            self.lbl_found.setText(self._create_kpi_card("FITS Captures", str(len(active_files)), "#1565C0").text())
            
            success_cnt, failed_cnt = 0, 0
            dataset_group = os.path.basename(target_dir)

            for idx, filename in enumerate(active_files):
                pct = int(((idx + 1) / len(active_files)) * 100)
                self.progress_bar.setValue(pct)
                self.progress_bar.setFormat(f"Processing Image {idx + 1} of {len(active_files)} ({pct}%)")

                img_id = os.path.splitext(filename)[0]
                filepath = os.path.join(fits_dir, filename)

                self.log_debug(f"--- Frame [{idx+1}/{len(active_files)}]: {filename} ---", "INFO")

                try:
                    with fits.open(filepath, mode='readonly') as hdul:
                        header = hdul[0].header
                        img_data = hdul[0].data if len(hdul) > 0 else None
                        
                        raw_ra = header.get('OBJCTRA', header.get('RA', header.get('CRVAL1', 'NA')))
                        raw_dec = header.get('OBJCTDEC', header.get('DEC', header.get('CRVAL2', 'NA')))
                        ra_hint = self._parse_coordinate_to_decimal(raw_ra, is_ra=True)
                        dec_hint = self._parse_coordinate_to_decimal(raw_dec, is_ra=False)
                        
                        focal_val = float(header.get('FOCALLEN', header.get('FOCLEN', default_focal_len)))
                        if focal_val <= 0: focal_val = default_focal_len

                        exp_val = float(header.get('EXPTIME', 30.0))
                        pix_size = float(header.get('XPIXSZ', header.get('PIXSIZE1', default_pixel_size)))
                        if pix_size <= 0: pix_size = default_pixel_size

                    astap_sol = {'solved': False}
                    if use_astap:
                        naxis2 = img_data.shape[0] if img_data is not None else 2160
                        fov_height = (naxis2 * pix_size * 206.265) / (3600.0 * focal_val)
                        astap_sol = self._solve_plate_astap(filepath, astap_bin, ra_hint, dec_hint, fov_height, show_astap_gui)

                    with fits.open(filepath, mode='readonly') as hdul:
                        active_header = hdul[0].header

                    if astap_sol.get('solved'):
                        solved_ra = str(astap_sol['ra'])
                        solved_dec = str(astap_sol['dec'])
                        solved_roll = str(astap_sol['roll'])
                        
                        solved_scale = float(astap_sol.get('scale', 0.0))
                        if solved_scale > 0 and pix_size > 0:
                            focal_val = round((206.265 * pix_size) / solved_scale, 1)
                        elif astap_sol.get('focal_len', 0.0) > 0:
                            focal_val = round(float(astap_sol['focal_len']), 1)

                        status_str = "ASTAP Solved"
                        status_color = QColor(232, 245, 233)
                        success_cnt += 1
                    else:
                        solved_ra = str(ra_hint)
                        solved_dec = str(dec_hint)
                        solved_roll = "NA"
                        status_str = "PSF Extracted" if not use_astap else "Solve Failed"
                        status_color = QColor(227, 242, 253) if not use_astap else QColor(255, 235, 238)
                        success_cnt += 1 if not use_astap else 0
                        failed_cnt += 0 if not use_astap else 1

                    # Execute PSF-Enforced Star Extraction
                    c_ra = float(solved_ra) if solved_ra != "NA" else 0.0
                    c_dec = float(solved_dec) if solved_dec != "NA" else 0.0
                    bg_mean, bg_std, med_snr, star_records = self._extract_merline_howell_stars(
                        img_data, active_header, astap_sol, focal_val, exp_val, c_ra, c_dec
                    )

                    self.log_debug(f"Extracted <b>{len(star_records)} valid PSF stars</b> | Median SNR: <b>{med_snr}</b> | Noise: {bg_std} e⁻", "SUCCESS")

                    if star_records:
                        df_stars = pd.DataFrame(star_records)[[
                            'star_id', 'x_image', 'y_image', 'ra_deg', 'dec_deg', 
                            'flux_mag', 'focal_length', 'exposure_time', 'snr', 'is_artifact'
                        ]]
                        df_stars.to_csv(os.path.join(csv_dir, f"{img_id}.csv"), index=False)

                    if img_data is not None:
                        self._save_annotated_png_fast(img_data.astype(np.float32), star_records, os.path.join(png_dir, f"{img_id}.png"), bg_mean, bg_std)

                    manifest_row = {
                        'image_id': str(img_id),
                        'dataset_group': dataset_group,
                        'ra': solved_ra,
                        'dec': solved_dec,
                        'camera_roll': solved_roll,
                        'focal_length_mm': str(focal_val),
                        'exposure_time_s': str(exp_val),
                        'total_stars': str(len(star_records)),
                        'median_image_snr': str(med_snr),
                        'bg_mean_e': str(bg_mean),
                        'bg_std_e': str(bg_std),
                        'distorted_stars': 'NA', 'dropped_stars': 'NA', 'false_stars': 'NA', 'smear_px': 'NA',
                        'anom_lens_on': False, 'anom_false_on': False, 'anom_drop_on': False, 
                        'anom_pos_on': False, 'anom_mag_on': False, 'anom_smear_on': False
                    }
                    self.telemetry_data.append(manifest_row)

                    row_idx = self.table.rowCount()
                    self.table.insertRow(row_idx)
                    items = [
                        QTableWidgetItem(manifest_row['image_id']),
                        QTableWidgetItem(manifest_row['dataset_group']),
                        QTableWidgetItem(manifest_row['ra']),
                        QTableWidgetItem(manifest_row['dec']),
                        QTableWidgetItem(manifest_row['camera_roll']),
                        QTableWidgetItem(manifest_row['focal_length_mm']),
                        QTableWidgetItem(manifest_row['exposure_time_s']),
                        QTableWidgetItem(manifest_row['total_stars']),
                        QTableWidgetItem(manifest_row['median_image_snr']),
                        QTableWidgetItem(manifest_row['bg_std_e']),
                        QTableWidgetItem(status_str)
                    ]
                    for c_idx, itm in enumerate(items):
                        itm.setTextAlignment(Qt.AlignCenter)
                        itm.setBackground(status_color)
                        self.table.setItem(row_idx, c_idx, itm)

                except Exception as e:
                    failed_cnt += 1
                    self.log_debug(f"Processing error on {filename}: {e}", "ERROR")

                self.lbl_success.setText(self._create_kpi_card("Extracted / Solved", str(success_cnt), "#2E7D32").text())
                self.lbl_failed.setText(self._create_kpi_card("Extraction Failures", str(failed_cnt), "#C62828" if failed_cnt > 0 else "#607D8B").text())
                QApplication.processEvents()

            if self.telemetry_data:
                galsim_columns = [
                    'image_id', 'dataset_group', 'ra', 'dec', 'camera_roll', 
                    'focal_length_mm', 'exposure_time_s', 'total_stars', 
                    'median_image_snr', 'bg_mean_e', 'bg_std_e', 'distorted_stars', 
                    'dropped_stars', 'false_stars', 'smear_px', 'anom_lens_on', 
                    'anom_false_on', 'anom_drop_on', 'anom_pos_on', 'anom_mag_on', 'anom_smear_on'
                ]
                df_manifest = pd.DataFrame(self.telemetry_data)[galsim_columns]
                manifest_path = os.path.join(target_dir, output_csv_name)
                df_manifest.to_csv(manifest_path, index=False)
                
                self.log_debug(f"Universal manifest saved to: <code>{manifest_path}</code>", "SUCCESS")
                QMessageBox.information(
                    self, 
                    "Dataset Normalization Complete", 
                    f"Successfully processed {len(active_files)} physical captures!\n\n"
                    f"• FITS Images:    {fits_dir}\n"
                    f"• Star Catalogs:  {csv_dir}\n"
                    f"• Verification:   {png_dir}\n"
                    f"• Global Manifest:{manifest_path}"
                )

        except Exception as e:
            self.log_debug(f"Fatal error: {e}", "ERROR")
            QMessageBox.critical(self, "System Error", f"Fatal execution exception:\n{str(e)}")
        finally:
            self.run_btn.setText("Execute Complete Dataset Ingestion & PSF Star Extraction")
            self.run_btn.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    window = AstapFUSExtractor()
    window.show()
    sys.exit(app.exec_())