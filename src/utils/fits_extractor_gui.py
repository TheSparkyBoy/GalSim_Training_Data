# This code is meant to be run on Windows/Linux with PyQt5 installed.
import sys
import os
import pandas as pd
from astropy.io import fits
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QLineEdit, QFileDialog, QTableWidget, 
    QTableWidgetItem, QHeaderView, QMessageBox, QGroupBox, QGridLayout, QStyleFactory
)
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtCore import Qt

class FitsTelemetryExtractor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZWO FITS Hardware Telemetry Extractor")
        self.setGeometry(150, 150, 1200, 650)
        
        self.telemetry_data = []
        self.init_ui()

    def init_ui(self):
        # 1. Directory Selection Group
        file_group = QGroupBox("1. Select Hardware Target Directory")
        file_layout = QGridLayout()
        
        self.dir_input = QLineEdit()
        self.dir_input.setPlaceholderText("Select folder containing .fits images...")
        dir_btn = QPushButton("Browse Folder")
        dir_btn.clicked.connect(self.browse_directory)
        
        self.out_input = QLineEdit()
        self.out_input.setText("zwo_fits_telemetry.csv")
        
        file_layout.addWidget(QLabel("Target Directory:"), 0, 0)
        file_layout.addWidget(self.dir_input, 0, 1)
        file_layout.addWidget(dir_btn, 0, 2)
        
        file_layout.addWidget(QLabel("Output CSV Name:"), 1, 0)
        file_layout.addWidget(self.out_input, 1, 1)
        file_group.setLayout(file_layout)
        
        # 2. Execution Trigger
        self.run_btn = QPushButton("Scan Directory & Extract Telemetry")
        self.run_btn.setFixedHeight(38)
        self.run_btn.setStyleSheet("background-color: #0277BD; color: white; font-weight: bold; font-size: 14px;")
        self.run_btn.clicked.connect(self.run_extraction)
        self.run_btn.setEnabled(False) 
        
        # 3. KPI Summary Banner
        kpi_layout = QHBoxLayout()
        self.lbl_found = self._create_kpi_card("FITS Files Found", "0", "#1565C0")
        self.lbl_success = self._create_kpi_card("Successfully Extracted", "0", "#2E7D32")
        self.lbl_failed = self._create_kpi_card("Read Errors", "0", "#C62828")
        
        kpi_layout.addWidget(self.lbl_found)
        kpi_layout.addWidget(self.lbl_success)
        kpi_layout.addWidget(self.lbl_failed)
        
        # 4. Results Ledger Table (Updated Columns to match Manifest)
        self.table = QTableWidget()
        self.table.setColumnCount(11)
        self.table.setHorizontalHeaderLabels([
            "Image ID", "RA", "DEC", "Roll (deg)", "Exposure (s)", 
            "Camera Model", "Temp (°C)", "Gain", "Offset", "Binning", "Date Observed"
        ])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents) # Image ID / Filename space
        
        # Master Layout Assembly
        main_layout = QVBoxLayout()
        main_layout.addWidget(file_group)
        main_layout.addWidget(self.run_btn)
        main_layout.addLayout(kpi_layout)
        main_layout.addWidget(QLabel("<b>Extracted FITS Header Ledger (Manifest Format):</b>"))
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

    def run_extraction(self):
        target_directory = self.dir_input.text()
        output_csv_name = self.out_input.text()
        
        if not os.path.exists(target_directory):
            QMessageBox.critical(self, "Error", "The selected directory does not exist.")
            return

        self.run_btn.setText("Scanning & Extracting... Please Wait")
        self.run_btn.setEnabled(False)
        QApplication.processEvents()
        
        try:
            fits_files = [f for f in os.listdir(target_directory) if f.lower().endswith(('.fits', '.fit'))]
            
            if not fits_files:
                QMessageBox.information(self, "No Files", f"No .fits or .fit files found in:\n{target_directory}")
                self.run_btn.setText("Scan Directory & Extract Telemetry")
                self.run_btn.setEnabled(True)
                return

            # Reset state
            self.telemetry_data = []
            self.table.setRowCount(0)
            
            self.lbl_found.setText(self._create_kpi_card("FITS Files Found", str(len(fits_files)), "#1565C0").text())
            QApplication.processEvents()

            success_cnt = 0
            failed_cnt = 0
            
            for filename in fits_files:
                filepath = os.path.join(target_directory, filename)
                try:
                    with fits.open(filepath, mode='readonly') as hdul:
                        header = hdul[0].header
                        
                        # --- CASCADING FALLBACK SEARCH FOR ASTROMETRY METRICS ---
                        # Searches standard WCS, NINA, SharpCap, and ZWO capture keys
                        ra_val = header.get('OBJCTRA', header.get('RA', header.get('CRVAL1', 'N/A')))
                        dec_val = header.get('OBJCTDEC', header.get('DEC', header.get('CRVAL2', 'N/A')))
                        roll_val = header.get('ROLL', header.get('ROTANG', header.get('CROTA2', header.get('PA', '0.0'))))
                        
                        # Format image_id to match GalSim manifest (strips extension)
                        img_id = os.path.splitext(filename)[0]

                        file_info = {
                            'image_id': img_id,
                            'ra': str(ra_val),
                            'dec': str(dec_val),
                            'camera_roll': str(roll_val),
                            'exposure_time_s': str(header.get('EXPTIME', 'N/A')),
                            'camera_model': str(header.get('INSTRUME', header.get('CAMERA', 'Unknown'))),
                            'sensor_temp_c': str(header.get('CCD-TEMP', 'N/A')),
                            'gain': str(header.get('GAIN', 'N/A')),
                            'offset': str(header.get('OFFSET', header.get('BLKLEVEL', 'N/A'))),
                            'binning': f"{header.get('XBINNING', 1)}x{header.get('YBINNING', 1)}",
                            'observation_date': str(header.get('DATE-OBS', 'N/A')),
                            'original_filename': filename
                        }
                        
                        self.telemetry_data.append(file_info)
                        
                        # Populate GUI Table
                        row_idx = self.table.rowCount()
                        self.table.insertRow(row_idx)
                        
                        items = [
                            QTableWidgetItem(file_info['image_id']),
                            QTableWidgetItem(file_info['ra']),
                            QTableWidgetItem(file_info['dec']),
                            QTableWidgetItem(file_info['camera_roll']),
                            QTableWidgetItem(file_info['exposure_time_s']),
                            QTableWidgetItem(file_info['camera_model']),
                            QTableWidgetItem(file_info['sensor_temp_c']),
                            QTableWidgetItem(file_info['gain']),
                            QTableWidgetItem(file_info['offset']),
                            QTableWidgetItem(file_info['binning']),
                            QTableWidgetItem(file_info['observation_date'])
                        ]
                        
                        for col_idx, item in enumerate(items):
                            item.setTextAlignment(Qt.AlignCenter)
                            self.table.setItem(row_idx, col_idx, item)
                            
                        success_cnt += 1
                        
                except Exception as e:
                    failed_cnt += 1
                    print(f"Failed to read {filename}: {e}")

                # Keep GUI responsive during heavy folder scans
                if (success_cnt + failed_cnt) % 10 == 0:
                    QApplication.processEvents()

            # Update final KPIs
            self.lbl_success.setText(self._create_kpi_card("Successfully Extracted", str(success_cnt), "#2E7D32").text())
            self.lbl_failed.setText(self._create_kpi_card("Read Errors", str(failed_cnt), "#C62828" if failed_cnt > 0 else "#607D8B").text())
            
            # Export to CSV automatically
            if self.telemetry_data:
                df = pd.DataFrame(self.telemetry_data)
                output_path = os.path.join(target_directory, output_csv_name)
                df.to_csv(output_path, index=False)
                QMessageBox.information(self, "Extraction Complete", f"Successfully extracted {success_cnt} FITS headers.\n\nManifest saved to:\n{output_path}")

        except Exception as e:
            QMessageBox.critical(self, "System Error", f"A fatal error occurred:\n{str(e)}")
        
        finally:
            self.run_btn.setText("Scan Directory & Extract Telemetry")
            self.run_btn.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    window = FitsTelemetryExtractor()
    window.show()
    sys.exit(app.exec_())