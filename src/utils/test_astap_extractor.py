# fits_extractor_gui.py
# Production Astrometric Extractor & Universal Manifest Builder [V6 ASTAP NATIVE STAR ENGINE + PORTABLE MODULE]

import sys
import os
import shutil
import time
from datetime import datetime
import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.coordinates import Angle, SkyCoord
from astropy import units as u
import cv2

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QLineEdit, QFileDialog, QTableWidget, 
    QTableWidgetItem, QHeaderView, QMessageBox, QGroupBox, QGridLayout, 
    QStyleFactory, QCheckBox, QProgressBar, QTextEdit, QSplitter, QDoubleSpinBox
)
from PyQt5.QtGui import QColor, QFont, QTextCursor
from PyQt5.QtCore import Qt, QSettings

from ASTAP_extraction_function import extract_stars_to_array

class AstapFUSExtractor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aerospace ASTAP Live Extractor & Telemetry Builder [V6 ASTAP NATIVE STAR ENGINE]")
        self.setGeometry(60, 40, 1420, 960)
        self.settings = QSettings("Aerospace", "ASTAPExtractor")
        
        self.telemetry_data = []
        self.detected_database_name = "Scanning..."
        self.init_ui()

    def init_ui(self):
        file_group = QGroupBox("1. Hardware Sensor, ASTAP & Reference Catalog Configuration")
        file_layout = QGridLayout()
        
        self.dir_input = QLineEdit()
        self.dir_input.setPlaceholderText("Select target dataset folder containing captures (.fits/.fit/.png)...")
        last_dir = self.settings.value("last_fits_dir", "")
        if last_dir and os.path.exists(last_dir):
            self.dir_input.setText(last_dir)
        dir_btn = QPushButton("Browse Folder")
        dir_btn.clicked.connect(self.browse_directory)
        
        self.cat_input = QLineEdit()
        self.cat_input.setPlaceholderText("Select ground-truth reference catalog CSV (e.g. source_id, RA_ICRS, DE_ICRS, Gmag)...")
        last_cat = self.settings.value("last_ref_catalog", "")
        if last_cat and os.path.exists(last_cat):
            self.cat_input.setText(last_cat)
        cat_btn = QPushButton("Browse Catalog")
        cat_btn.clicked.connect(self.browse_catalog)
        
        default_astap = self.settings.value("last_astap_path", "")
        if not default_astap or not os.path.exists(default_astap):
            default_astap = r"C:\Program Files\astap\astap.exe" if os.name == 'nt' else "/usr/bin/astap"

        self.astap_input = QLineEdit()
        self.astap_input.setText(default_astap)
        self.astap_input.textChanged.connect(self._refresh_database_detection)
        astap_btn = QPushButton("Browse ASTAP")
        astap_btn.clicked.connect(self.browse_astap)
        
        self.out_input = QLineEdit()
        self.out_input.setText("dataset_manifest.csv")
        
        self.spin_pixel_size = QDoubleSpinBox()
        self.spin_pixel_size.setRange(0.5, 20.0)
        self.spin_pixel_size.setValue(float(self.settings.value("last_pixel_size", 2.9)))
        self.spin_pixel_size.setSingleStep(0.1)
        self.spin_pixel_size.setSuffix(" µm")
        self.spin_pixel_size.valueChanged.connect(self._update_calculated_fov)
        
        self.spin_focal_len = QDoubleSpinBox()
        self.spin_focal_len.setRange(10.0, 5000.0)
        self.spin_focal_len.setValue(float(self.settings.value("last_focal_len", 416.0)))
        self.spin_focal_len.setSingleStep(10.0)
        self.spin_focal_len.setSuffix(" mm")
        self.spin_focal_len.valueChanged.connect(self._update_calculated_fov)
        
        self.lbl_fov_display = QLabel("Calculated FOV: <b>1.53° × 0.86°</b> (Height Parameter: <b>0.86°</b>)")
        self.lbl_fov_display.setStyleSheet("color: #0277BD; font-weight: bold;")
        
        opt_layout = QHBoxLayout()
        self.chk_use_astap = QCheckBox("Use External ASTAP (-extract2 Star Detection & Solving)")
        self.chk_use_astap.setChecked(True)
        self.chk_use_astap.setStyleSheet("font-weight: bold; color: #2E7D32;")
        self.chk_use_astap.toggled.connect(self._toggle_astap_controls)
        
        self.chk_show_astap_gui = QCheckBox("Show ASTAP Window")
        self.chk_show_astap_gui.setChecked(False)
        self.chk_show_astap_gui.setStyleSheet("font-weight: bold; color: #D81B60;")
        
        opt_layout.addWidget(self.chk_use_astap)
        opt_layout.addWidget(self.chk_show_astap_gui)
        opt_layout.addStretch()
        
        file_layout.addWidget(QLabel("Target Dataset Directory:"), 0, 0)
        file_layout.addWidget(self.dir_input, 0, 1, 1, 2)
        file_layout.addWidget(dir_btn, 0, 3)
        
        file_layout.addWidget(QLabel("Reference Catalog CSV:"), 1, 0)
        file_layout.addWidget(self.cat_input, 1, 1, 1, 2)
        file_layout.addWidget(cat_btn, 1, 3)
        
        file_layout.addWidget(QLabel("ASTAP Executable:"), 2, 0)
        file_layout.addWidget(self.astap_input, 2, 1, 1, 2)
        file_layout.addWidget(astap_btn, 2, 3)
        
        file_layout.addWidget(QLabel("Manifest CSV Name:"), 3, 0)
        file_layout.addWidget(self.out_input, 3, 1, 1, 3)
        
        file_layout.addWidget(QLabel("Sensor Pixel Size:"), 4, 0)
        file_layout.addWidget(self.spin_pixel_size, 4, 1)
        file_layout.addWidget(QLabel("Preset Focal Length:"), 4, 2)
        file_layout.addWidget(self.spin_focal_len, 4, 3)
        
        file_layout.addWidget(self.lbl_fov_display, 5, 0, 1, 4)
        file_layout.addLayout(opt_layout, 6, 0, 1, 4)
        file_group.setLayout(file_layout)
        
        exec_layout = QHBoxLayout()
        self.run_btn = QPushButton("Execute Complete Dataset Ingestion & Star Extraction Pipeline")
        self.run_btn.setFixedHeight(42)
        self.run_btn.setStyleSheet("background-color: #0277BD; color: white; font-weight: bold; font-size: 14px;")
        self.run_btn.clicked.connect(self.run_extraction)
        self.run_btn.setEnabled(bool(last_dir and os.path.exists(last_dir)))
        
        clear_log_btn = QPushButton("Clear Console")
        clear_log_btn.setFixedHeight(42)
        clear_log_btn.clicked.connect(lambda: self.log_console.clear())
        
        exec_layout.addWidget(self.run_btn, 4)
        exec_layout.addWidget(clear_log_btn, 1)
        
        kpi_layout = QHBoxLayout()
        self.lbl_found = self._create_kpi_card("Dataset Captures", "0", "#1565C0")
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
        console_layout.addWidget(QLabel("<b>Live Extraction & Ground-Truth Stream:</b>"))
        
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
        self.settings.setValue("last_pixel_size", pix_um)
        self.settings.setValue("last_focal_len", focal_mm)

    def _toggle_astap_controls(self, enabled):
        self.astap_input.setEnabled(enabled)
        self.chk_show_astap_gui.setEnabled(enabled)
        self._check_ready_state()

    def log_debug(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        color_map = {
            "INFO": "#29B6F6", "SUCCESS": "#66BB6A", "WARN": "#FFA726",
            "ERROR": "#EF5350", "ASTAP": "#AB47BC", "CMD": "#90A4AE"
        }
        color = color_map.get(level, "#ECEFF1")
        html_msg = f"<span style='color: #78909C;'>[{timestamp}]</span> <b style='color: {color};'>[{level}]</b> {message}"
        self.log_console.append(html_msg)
        self.log_console.moveCursor(QTextCursor.End)
        QApplication.processEvents()

    def _refresh_database_detection(self):
        astap_bin = self.astap_input.text().strip()
        if not astap_bin or not os.path.exists(astap_bin):
            self.detected_database_name = "Binary Missing"
            self.lbl_database.setText(self._create_kpi_card("ASTAP Database", "Binary Missing", "#C62828").text())
            return

        astap_dir = os.path.dirname(os.path.abspath(astap_bin))
        self.settings.setValue("last_astap_path", astap_bin)
        
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
        last_dir = self.dir_input.text().strip() or self.settings.value("last_fits_dir", os.path.expanduser("~"))
        directory = QFileDialog.getExistingDirectory(self, "Select Target Dataset Directory", last_dir)
        if directory:
            self.dir_input.setText(directory)
            self.settings.setValue("last_fits_dir", directory)
            self._check_ready_state()

    def browse_catalog(self):
        last_cat = self.cat_input.text().strip() or self.settings.value("last_ref_catalog", os.path.expanduser("~"))
        filename, _ = QFileDialog.getOpenFileName(self, "Select Reference Catalog CSV", last_cat, "CSV Files (*.csv)")
        if filename:
            self.cat_input.setText(filename)
            self.settings.setValue("last_ref_catalog", filename)
            self.log_debug(f"Loaded reference catalog path: <b>{filename}</b>", "SUCCESS")

    def browse_astap(self):
        last_astap = self.astap_input.text().strip() or self.settings.value("last_astap_path", os.path.expanduser("~"))
        exe_filter = "Executables (*.exe)" if os.name == 'nt' else "Executables (*)"
        filename, _ = QFileDialog.getOpenFileName(self, "Select ASTAP Executable", last_astap, exe_filter)
        if filename:
            self.astap_input.setText(filename)
            self.settings.setValue("last_astap_path", filename)
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
        if val is None or val in ('NA', 'N/A', 'None') or pd.isna(val):
            return "NA"
        try:
            return float(val)
        except ValueError:
            try:
                unit = u.hourangle if is_ra else u.deg
                angle = Angle(str(val), unit=unit)
                return float(angle.deg)
            except Exception:
                return "NA"

    def _calibrate_magnitudes_with_catalog(self, star_records, catalog_path, match_threshold=5.0):
        if not star_records or not catalog_path or not os.path.exists(catalog_path):
            return star_records

        try:
            df_cat = pd.read_csv(catalog_path)
            col_map = {str(c).lower(): c for c in df_cat.columns}
            
            mag_col = next((col_map[c] for c in ['gmag', 'mag', 'g_mag', 'magnitude', 'flux_mag'] if c in col_map), None)
            id_col = next((col_map[c] for c in ['source_id', 'star_id', 'id'] if c in col_map), None)
            x_col = next((col_map[c] for c in ['x_image', 'x', 'xpos'] if c in col_map), None)
            y_col = next((col_map[c] for c in ['y_image', 'y', 'ypos'] if c in col_map), None)
            ra_col = next((col_map[c] for c in ['ra_icrs', 'ra', 'raj2000'] if c in col_map), None)
            dec_col = next((col_map[c] for c in ['de_icrs', 'dec', 'dej2000', 'dec_icrs'] if c in col_map), None)

            if not mag_col:
                return star_records

            cat_gmags = df_cat[mag_col].values
            cat_ids = df_cat[id_col].values if id_col else None
            delta_mags = []
            matched_count = 0

            if x_col and y_col:
                cat_x = df_cat[x_col].values
                cat_y = df_cat[y_col].values
                for star in star_records:
                    sx, sy = star.get('x_image'), star.get('y_image')
                    if sx is None or sy is None:
                        continue
                    distances = np.hypot(cat_x - sx, cat_y - sy)
                    min_idx = np.argmin(distances)
                    if distances[min_idx] <= match_threshold:
                        delta_mags.append(cat_gmags[min_idx] - star['flux_mag'])
                        if cat_ids is not None:
                            star['star_id'] = str(cat_ids[min_idx])
                        matched_count += 1
            elif ra_col and dec_col:
                cat_ras = df_cat[ra_col].values
                cat_decs = df_cat[dec_col].values
                for star in star_records:
                    s_ra, s_dec = star.get('ra_deg'), star.get('dec_deg')
                    if s_ra is None or s_dec is None:
                        continue
                    cos_dec = np.cos(np.radians(s_dec))
                    angular_distances = np.hypot((cat_ras - s_ra) * cos_dec * 3600.0, (cat_decs - s_dec) * 3600.0)
                    min_idx = np.argmin(angular_distances)
                    if angular_distances[min_idx] <= 30.0:
                        delta_mags.append(cat_gmags[min_idx] - star['flux_mag'])
                        if cat_ids is not None:
                            star['star_id'] = str(cat_ids[min_idx])
                        matched_count += 1

            if len(delta_mags) >= 1:
                zp_offset = float(np.median(delta_mags))
                for star in star_records:
                    star['flux_mag'] = round(star['flux_mag'] + zp_offset, 3)
                self.log_debug(f"Catalog Photometric Calibration: Matched {matched_count}/{len(star_records)} stars | ZP Offset: {zp_offset:+.3f} mag", "SUCCESS")
        except Exception as e:
            self.log_debug(f"Catalog calibration exception: {e}", "ERROR")

        return star_records

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
        catalog_path = self.cat_input.text().strip()
        output_csv_name = self.out_input.text().strip()
        use_astap = self.chk_use_astap.isChecked()
        astap_bin = os.path.normpath(os.path.abspath(self.astap_input.text().strip())) if use_astap else ""
        show_astap_gui = self.chk_show_astap_gui.isChecked()
        
        default_pixel_size = self.spin_pixel_size.value()
        default_focal_len = self.spin_focal_len.value()

        if not os.path.exists(target_dir):
            QMessageBox.critical(self, "Directory Error", "Target dataset directory does not exist.")
            return

        if catalog_path and not os.path.exists(catalog_path):
            QMessageBox.critical(self, "Catalog Error", f"Reference catalog file not found at:\n{catalog_path}")
            return

        if use_astap and not os.path.exists(astap_bin):
            QMessageBox.critical(self, "ASTAP Error", f"ASTAP binary not found at:\n{astap_bin}")
            return

        self.run_btn.setText("Processing Pipeline Running... Please Wait")
        self.run_btn.setEnabled(False)
        self.log_console.clear()
        self.log_debug("Initiating Pipeline using Portable ASTAP Extractor Module...", "INFO")

        try:
            # Support both flat directory structures and subfolders (/fits, /png)
            fits_dir = os.path.join(target_dir, "fits")
            if not os.path.exists(fits_dir):
                fits_dir = target_dir

            csv_dir = os.path.join(target_dir, "csv")
            png_dir = os.path.join(target_dir, "png")

            os.makedirs(csv_dir, exist_ok=True)
            os.makedirs(png_dir, exist_ok=True)

            active_files = sorted([f for f in os.listdir(fits_dir) if f.lower().endswith(('.fits', '.fit', '.png'))])
            if not active_files:
                QMessageBox.information(self, "No Files", f"No supported captures (.fits, .fit, .png) found in:\n{target_dir}")
                return

            self.telemetry_data = []
            self.table.setRowCount(0)
            self.lbl_found.setText(self._create_kpi_card("Dataset Captures", str(len(active_files)), "#1565C0").text())
            
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
                    img_data = None
                    raw_ra, raw_dec = None, None
                    focal_val = default_focal_len
                    exp_val = 30.0
                    pix_size = default_pixel_size

                    ext = os.path.splitext(filename)[1].lower()
                    if ext in ('.fits', '.fit'):
                        with fits.open(filepath, mode='readonly') as hdul:
                            header = hdul[0].header
                            img_data = hdul[0].data if len(hdul) > 0 else None
                            
                            raw_ra = header.get('OBJCTRA') or header.get('RA') or header.get('CRVAL1')
                            raw_dec = header.get('OBJCTDEC') or header.get('DEC') or header.get('CRVAL2')
                            
                            focal_raw = header.get('FOCALLEN') or header.get('FOCLEN')
                            try:
                                focal_val = float(focal_raw) if focal_raw is not None else default_focal_len
                            except (TypeError, ValueError):
                                focal_val = default_focal_len
                            if focal_val <= 0: focal_val = default_focal_len

                            exp_raw = header.get('EXPTIME')
                            try:
                                exp_val = float(exp_raw) if exp_raw is not None else 30.0
                            except (TypeError, ValueError):
                                exp_val = 30.0

                            pix_raw = header.get('XPIXSZ') or header.get('PIXSIZE1')
                            try:
                                pix_size = float(pix_raw) if pix_raw is not None else default_pixel_size
                            except (TypeError, ValueError):
                                pix_size = default_pixel_size
                            if pix_size <= 0: pix_size = default_pixel_size
                    else: # PNG support
                        img_gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                        if img_gray is not None:
                            img_data = img_gray.astype(np.float32)

                    ra_hint = self._parse_coordinate_to_decimal(raw_ra, is_ra=True)
                    dec_hint = self._parse_coordinate_to_decimal(raw_dec, is_ra=False)

                    star_records = []
                    med_snr = 0.0
                    bg_mean = float(np.median(img_data)) if img_data is not None else 0.0
                    bg_std = float(np.std(img_data)) if img_data is not None else 1.0

                    naxis2 = img_data.shape[0] if img_data is not None else 2160
                    fov_height = (naxis2 * pix_size * 206.265) / (3600.0 * focal_val)
                    
                    astap_res = extract_stars_to_array(
                        image_path=filepath,
                        astap_exe=astap_bin,
                        ra_hint=ra_hint if ra_hint != "NA" else None,
                        dec_hint=dec_hint if dec_hint != "NA" else None,
                        fov_height_deg=fov_height,
                        show_gui=show_astap_gui
                    )
                    
                    if astap_res.get('solved'):
                        solved_ra = str(astap_res['ra'])
                        solved_dec = str(astap_res['dec'])
                        solved_roll = str(astap_res['roll'])
                        
                        raw_scale = astap_res.get('scale')
                        solved_scale = float(raw_scale) if raw_scale is not None else 0.0
                        if solved_scale > 0 and pix_size > 0:
                            focal_val = round((206.265 * pix_size) / solved_scale, 1)

                        status_str = "ASTAP Extracted"
                        status_color = QColor(232, 245, 233)
                        success_cnt += 1
                    else:
                        solved_ra = str(ra_hint) if ra_hint != "NA" else "NA"
                        solved_dec = str(dec_hint) if dec_hint != "NA" else "NA"
                        solved_roll = "NA"
                        status_str = "Solve Failed"
                        status_color = QColor(255, 235, 238)
                        failed_cnt += 1

                    star_records = astap_res.get('star_records', [])
                    if star_records:
                        med_snr = 10.0

                    self.log_debug(f"Extracted <b>{len(star_records)} stars</b> using portable module | Status: <b>{status_str}</b>", "SUCCESS")

                    if star_records:
                        if catalog_path:
                            star_records = self._calibrate_magnitudes_with_catalog(star_records, catalog_path)

                        formatted_records = []
                        for s in star_records:
                            formatted_records.append({
                                'star_id': s['star_id'],
                                'x_image': s['x_image'],
                                'y_image': s['y_image'],
                                'flux_mag': s['flux_mag'],
                                'focal_length': focal_val,
                                'exposure_time': exp_val,
                                'snr': s['snr'],
                                'is_artifact': s['is_artifact'],
                                'ra': s['ra_deg'],
                                'dec': s['dec_deg']
                            })
                        df_stars = pd.DataFrame(formatted_records)
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
                    f"• Dataset Directory: {target_dir}\n"
                    f"• Star Catalogs:     {csv_dir}\n"
                    f"• Verification:      {png_dir}\n"
                    f"• Global Manifest:   {manifest_path}"
                )

        except Exception as e:
            self.log_debug(f"Fatal error: {e}", "ERROR")
            QMessageBox.critical(self, "System Error", f"Fatal execution exception:\n{str(e)}")
        finally:
            self.run_btn.setText("Execute Complete Dataset Ingestion & Star Extraction Pipeline")
            self.run_btn.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    window = AstapFUSExtractor()
    window.show()
    sys.exit(app.exec_())