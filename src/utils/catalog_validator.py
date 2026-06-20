# This code is meant to be run on Windows.
import sys
import os
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QLineEdit, QFileDialog, QTableWidget, 
    QTableWidgetItem, QHeaderView, QMessageBox, QGroupBox, QGridLayout, QStyleFactory
)
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtCore import Qt

class StarCatalogValidator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aerospace Star Catalog Cross-Validator")
        self.setGeometry(100, 100, 1050, 650)
        
        # Internal Data Storage
        self.output_df = None
        self.catalog_df = None
        self.audit_report_data = []
        
        self.init_ui()

    def init_ui(self):
        # 1. File Selection Header Group
        file_group = QGroupBox("1. Load Verification Datasets")
        file_layout = QGridLayout()
        
        self.out_path_input = QLineEdit()
        self.out_path_input.setPlaceholderText("Select simulated image output CSV (e.g., 0000001.csv)...")
        out_btn = QPushButton("Browse Output CSV")
        out_btn.clicked.connect(self.browse_output)
        
        self.cat_path_input = QLineEdit()
        self.cat_path_input.setPlaceholderText("Select Master Reference Catalog CSV (e.g., GAIADR3_cache.csv)...")
        cat_btn = QPushButton("Browse Master Catalog")
        cat_btn.clicked.connect(self.browse_catalog)
        
        file_layout.addWidget(QLabel("Simulated Output CSV:"), 0, 0)
        file_layout.addWidget(self.out_path_input, 0, 1)
        file_layout.addWidget(out_btn, 0, 2)
        
        file_layout.addWidget(QLabel("Master Catalog CSV:"), 1, 0)
        file_layout.addWidget(self.cat_path_input, 1, 1)
        file_layout.addWidget(cat_btn, 1, 2)
        file_group.setLayout(file_layout)
        
        # 2. Execution Trigger
        self.run_btn = QPushButton("Execute Catalog Cross-Check")
        self.run_btn.setFixedHeight(38)
        self.run_btn.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; font-size: 14px;")
        self.run_btn.clicked.connect(self.run_validation)
        self.run_btn.setEnabled(False) 
        
        # 3. KPI Summary Banner
        kpi_layout = QHBoxLayout()
        self.lbl_total = self._create_kpi_card("Total Output Stars", "0", "#1565C0")
        self.lbl_verified = self._create_kpi_card("Verified in Catalog", "0", "#2E7D32")
        self.lbl_artifacts = self._create_kpi_card("Injected Artifacts (-1)", "0", "#E65100")
        self.lbl_orphans = self._create_kpi_card("Orphan IDs (Missing)", "0", "#C62828")
        
        kpi_layout.addWidget(self.lbl_total)
        kpi_layout.addWidget(self.lbl_verified)
        kpi_layout.addWidget(self.lbl_artifacts)
        kpi_layout.addWidget(self.lbl_orphans)
        
        # 4. Results Ledger Table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Output Star ID", "Measured Mag", "Measured SNR", 
            "Catalog Status", "Reference Mag", "Mag Discrepancy (Δ)"
        ])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        
        # 5. Export Footer Button
        self.export_btn = QPushButton("Export Cross-Check Audit Ledger (CSV)")
        self.export_btn.setFixedHeight(32)
        self.export_btn.clicked.connect(self.export_audit)
        self.export_btn.setEnabled(False)
        
        # Master Layout Assembly
        main_layout = QVBoxLayout()
        main_layout.addWidget(file_group)
        main_layout.addWidget(self.run_btn)
        main_layout.addLayout(kpi_layout)
        main_layout.addWidget(QLabel("<b>Itemized Output Star Cross-Check Ledger:</b>"))
        main_layout.addWidget(self.table)
        main_layout.addWidget(self.export_btn)
        
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def _create_kpi_card(self, title, val, color):
        lbl = QLabel(f"<div style='text-align: center;'><small>{title}</small><br><b style='font-size: 20px; color: {color};'>{val}</b></div>")
        lbl.setFrameStyle(QLabel.StyledPanel | QLabel.Raised)
        lbl.setStyleSheet("background-color: #f8f9fa; padding: 6px; border-radius: 4px; border: 1px solid #ddd;")
        return lbl

    def browse_output(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Output CSV", os.path.expanduser("~"), "CSV Files (*.csv)")
        if filename:
            self.out_path_input.setText(filename)
            self._check_ready_state()

    def browse_catalog(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Master Catalog CSV", os.path.expanduser("~"), "CSV Files (*.csv)")
        if filename:
            self.cat_path_input.setText(filename)
            self._check_ready_state()

    def _check_ready_state(self):
        if self.out_path_input.text() and self.cat_path_input.text():
            self.run_btn.setEnabled(True)

    def run_validation(self):
        self.run_btn.setText("Analyzing Datasets... Please Wait")
        self.run_btn.setEnabled(False)
        QApplication.processEvents() # Keep GUI alive during heavy read
        
        try:
            # 1. Peek at headers to dynamically identify exact primary key columns
            out_sample = pd.read_csv(self.out_path_input.text(), nrows=0)
            cat_sample = pd.read_csv(self.cat_path_input.text(), nrows=0)
            
            out_id_col = 'star_id' if 'star_id' in out_sample.columns else ('source_id' if 'source_id' in out_sample.columns else out_sample.columns[0])
            cat_id_col = 'source_id' if 'source_id' in cat_sample.columns else ('star_id' if 'star_id' in cat_sample.columns else cat_sample.columns[0])
            
            # 2. Load full datasets, strictly locking detected ID columns as text strings
            out_df = pd.read_csv(self.out_path_input.text(), dtype={out_id_col: str})
            cat_df = pd.read_csv(self.cat_path_input.text(), dtype={cat_id_col: str})
            
            # 3. Scrub strings (strip whitespace and strip trailing float '.0' artifacts)
            out_df['__clean_id'] = out_df[out_id_col].astype(str).str.strip().str.replace('.0', '', regex=False)
            cat_df['__clean_id'] = cat_df[cat_id_col].astype(str).str.strip().str.replace('.0', '', regex=False)
            
            # 4. Build instant O(1) string lookup set
            catalog_id_set = set(cat_df['__clean_id'])
            
            # 5. Build safe magnitude lookup table
            cat_mag_col = 'Gmag' if 'Gmag' in cat_df.columns else ('flux_mag' if 'flux_mag' in cat_df.columns else None)
            if cat_mag_col:
                cat_df['__clean_mag'] = pd.to_numeric(cat_df[cat_mag_col], errors='coerce')
                mag_lookup = cat_df.set_index('__clean_id')['__clean_mag'].to_dict()
            else:
                mag_lookup = {}
                
            # Reset Auditing Storage
            self.audit_report_data = []
            verified_cnt, artifact_count, orphan_count = 0, 0, 0
            
            self.table.setRowCount(len(out_df))
            
            # Process and grade every output star
            for idx, row in out_df.iterrows():
                star_id_str = row['__clean_id']
                meas_mag = pd.to_numeric(row.get('flux_mag', 0.0), errors='coerce')
                meas_snr = pd.to_numeric(row.get('snr', 0.0), errors='coerce')
                
                if star_id_str == '-1':
                    status, cat_mag, delta_mag = "Artifact (-1)", "N/A", "N/A"
                    artifact_count += 1
                    row_color = QColor(255, 243, 224) # Soft Orange
                elif star_id_str in catalog_id_set:
                    status = "Verified in Catalog"
                    cat_mag = mag_lookup.get(star_id_str, "Unknown")
                    delta_mag = f"{meas_mag - cat_mag:+.2f}" if isinstance(cat_mag, (int, float)) else "N/A"
                    verified_cnt += 1
                    row_color = QColor(232, 245, 233) # Soft Green
                else:
                    status, cat_mag, delta_mag = "Orphan ID (Missing)", "Missing", "N/A"
                    orphan_count += 1
                    row_color = QColor(255, 235, 238) # Soft Red
                    
                # Format text values for table display
                cat_mag_str = f"{cat_mag:.2f}" if isinstance(cat_mag, (int, float)) else str(cat_mag)
                meas_mag_str = f"{meas_mag:.2f}" if not pd.isna(meas_mag) else "0.00"
                meas_snr_str = f"{meas_snr:.2f}" if not pd.isna(meas_snr) else "0.00"
                
                # Append to GUI Table
                items = [
                    QTableWidgetItem(star_id_str), QTableWidgetItem(meas_mag_str),
                    QTableWidgetItem(meas_snr_str), QTableWidgetItem(status),
                    QTableWidgetItem(cat_mag_str), QTableWidgetItem(delta_mag)
                ]
                
                for col_idx, item in enumerate(items):
                    item.setBackground(row_color)
                    item.setTextAlignment(Qt.AlignCenter)
                    self.table.setItem(idx, col_idx, item)
                    
                # Save raw row for CSV exporter
                self.audit_report_data.append({
                    'output_star_id': star_id_str, 'measured_magnitude': meas_mag,
                    'measured_snr': meas_snr, 'catalog_validation_status': status,
                    'reference_catalog_mag': cat_mag, 'magnitude_discrepancy': delta_mag
                })

            # Update KPI Dashboards
            self.lbl_total.setText(self._create_kpi_card("Total Output Stars", str(len(out_df)), "#1565C0").text())
            self.lbl_verified.setText(self._create_kpi_card("Verified in Catalog", str(verified_cnt), "#2E7D32").text())
            self.lbl_artifacts.setText(self._create_kpi_card("Injected Artifacts (-1)", str(artifact_count), "#E65100").text())
            self.lbl_orphans.setText(self._create_kpi_card("Orphan IDs (Missing)", str(orphan_count), "#C62828").text())
            
            self.export_btn.setEnabled(True)
            self.run_btn.setText("Execute Catalog Cross-Check")
            self.run_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Auditing Error", f"Failed to parse datasets:\n{str(e)}")
            self.run_btn.setText("Execute Catalog Cross-Check")
            self.run_btn.setEnabled(True)

    def export_audit(self):
        if not self.audit_report_data: return
        
        path, _ = QFileDialog.getSaveFileName(self, "Save Verification Ledger", os.path.expanduser("~"), "CSV Files (*.csv)")
        if path:
            try:
                pd.DataFrame(self.audit_report_data).to_csv(path, index=False)
                QMessageBox.information(self, "Export Complete", f"Audit report successfully saved to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"Could not write CSV:\n{str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion")) # Clean cross-platform aesthetic
    window = StarCatalogValidator()
    window.show()
    sys.exit(app.exec_())