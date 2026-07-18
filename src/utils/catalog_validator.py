# This code is meant to be run on Windows.
import sys
import os
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QLineEdit, QFileDialog, QTableWidget, 
    QTableWidgetItem, QHeaderView, QMessageBox, QGroupBox, QGridLayout, 
    QStyleFactory, QCheckBox, QAction
)
from PyQt5.QtGui import QColor, QFont, QKeySequence
from PyQt5.QtCore import Qt

class StarCatalogValidator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aerospace Star Catalog Cross-Validator")
        self.setGeometry(100, 100, 1200, 680)
        
        # Internal Data Storage
        self.output_df = None
        self.catalog_df = None
        self.audit_report_data = []
        
        self.init_menu()
        self.init_ui()

    def init_menu(self):
        # --- Native Windows Menu Bar with Ctrl+S Support ---
        menubar = self.menuBar()
        file_menu = menubar.addMenu('&File')
        
        self.export_action = QAction('&Save Enriched CSV As...', self)
        self.export_action.setShortcut(QKeySequence.Save) # Binds Ctrl+S
        self.export_action.setStatusTip('Prompt to save the full enriched cross-check results to a CSV file')
        self.export_action.triggered.connect(self.export_audit)
        self.export_action.setEnabled(False)
        
        exit_action = QAction('&Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

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
        
        # 2. Execution & Interactive Save Prompt Controls
        exec_layout = QHBoxLayout()
        self.run_btn = QPushButton("Execute Catalog Cross-Check")
        self.run_btn.setFixedHeight(38)
        self.run_btn.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; font-size: 14px;")
        self.run_btn.clicked.connect(self.run_validation)
        self.run_btn.setEnabled(False)
        
        # New: Top-level Save As button for immediate access
        self.top_save_btn = QPushButton("Save Enriched CSV As...")
        self.top_save_btn.setFixedHeight(38)
        self.top_save_btn.setStyleSheet("background-color: #1565C0; color: white; font-weight: bold; font-size: 14px;")
        self.top_save_btn.clicked.connect(self.export_audit)
        self.top_save_btn.setEnabled(False)
        
        self.prompt_save_chk = QCheckBox("Prompt for CSV save location after check completes")
        self.prompt_save_chk.setChecked(True) # Checked by default so it prompts you immediately!
        self.prompt_save_chk.setStyleSheet("font-weight: bold; color: #333333; padding-left: 10px;")
        
        exec_layout.addWidget(self.run_btn, stretch=2)
        exec_layout.addWidget(self.top_save_btn, stretch=1)
        exec_layout.addWidget(self.prompt_save_chk, stretch=1)
        
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
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "Output Star ID", "Measured Mag", "Measured SNR", 
            "Catalog Status", "Reference Mag", "Catalog RA", "Catalog DEC", "Mag Discrepancy (Δ)"
        ])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        
        # 5. Export Footer Button
        self.export_btn = QPushButton("Save Enriched Cross-Check Results As... (CSV) | Shortcut: Ctrl+S")
        self.export_btn.setFixedHeight(34)
        self.export_btn.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.export_btn.clicked.connect(self.export_audit)
        self.export_btn.setEnabled(False)
        
        # Master Layout Assembly
        main_layout = QVBoxLayout()
        main_layout.addWidget(file_group)
        main_layout.addLayout(exec_layout)
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
        QApplication.processEvents()
        
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

            # 6. Build safe RA/DEC lookup tables dynamically
            cat_ra_col = next((c for c in cat_df.columns if 'ra' in c.lower()), None)
            cat_dec_col = next((c for c in cat_df.columns if 'de' in c.lower() or 'dec' in c.lower()), None)

            if cat_ra_col:
                cat_df['__clean_ra'] = pd.to_numeric(cat_df[cat_ra_col], errors='coerce')
                ra_lookup = cat_df.set_index('__clean_id')['__clean_ra'].to_dict()
            else:
                ra_lookup = {}

            if cat_dec_col:
                cat_df['__clean_dec'] = pd.to_numeric(cat_df[cat_dec_col], errors='coerce')
                dec_lookup = cat_df.set_index('__clean_id')['__clean_dec'].to_dict()
            else:
                dec_lookup = {}
                
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
                    status = "Artifact (-1)"
                    cat_mag, cat_ra, cat_dec, delta_mag = "N/A", "N/A", "N/A", "N/A"
                    artifact_count += 1
                    row_color = QColor(255, 243, 224)
                elif star_id_str in catalog_id_set:
                    status = "Verified in Catalog"
                    cat_mag = mag_lookup.get(star_id_str, "Unknown")
                    cat_ra = ra_lookup.get(star_id_str, "Unknown")
                    cat_dec = dec_lookup.get(star_id_str, "Unknown")
                    delta_mag = f"{meas_mag - cat_mag:+.2f}" if isinstance(cat_mag, (int, float)) and not pd.isna(cat_mag) else "N/A"
                    verified_cnt += 1
                    row_color = QColor(232, 245, 233)
                else:
                    status = "Orphan ID (Missing)"
                    cat_mag, cat_ra, cat_dec, delta_mag = "Missing", "N/A", "N/A", "N/A"
                    orphan_count += 1
                    row_color = QColor(255, 235, 238)
                    
                # Format text values for table display
                cat_mag_str = f"{cat_mag:.2f}" if isinstance(cat_mag, (int, float)) and not pd.isna(cat_mag) else str(cat_mag)
                cat_ra_str = f"{cat_ra:.5f}" if isinstance(cat_ra, (int, float)) and not pd.isna(cat_ra) else str(cat_ra)
                cat_dec_str = f"{cat_dec:.5f}" if isinstance(cat_dec, (int, float)) and not pd.isna(cat_dec) else str(cat_dec)
                meas_mag_str = f"{meas_mag:.2f}" if not pd.isna(meas_mag) else "0.00"
                meas_snr_str = f"{meas_snr:.2f}" if not pd.isna(meas_snr) else "0.00"
                
                # Append to GUI Table
                items = [
                    QTableWidgetItem(star_id_str), QTableWidgetItem(meas_mag_str),
                    QTableWidgetItem(meas_snr_str), QTableWidgetItem(status),
                    QTableWidgetItem(cat_mag_str), QTableWidgetItem(cat_ra_str),
                    QTableWidgetItem(cat_dec_str), QTableWidgetItem(delta_mag)
                ]
                
                for col_idx, item in enumerate(items):
                    item.setBackground(row_color)
                    item.setTextAlignment(Qt.AlignCenter)
                    self.table.setItem(idx, col_idx, item)
                    
                # --- FULL TELEMETRY ENRICHMENT ---
                enriched_row = row.to_dict()
                enriched_row.pop('__clean_id', None)
                enriched_row.update({
                    'catalog_validation_status': status,
                    'reference_catalog_mag': cat_mag,
                    'reference_catalog_ra': cat_ra,
                    'reference_catalog_dec': cat_dec,
                    'magnitude_discrepancy': delta_mag
                })
                self.audit_report_data.append(enriched_row)

            # Update KPI Dashboards
            self.lbl_total.setText(self._create_kpi_card("Total Output Stars", str(len(out_df)), "#1565C0").text())
            self.lbl_verified.setText(self._create_kpi_card("Verified in Catalog", str(verified_cnt), "#2E7D32").text())
            self.lbl_artifacts.setText(self._create_kpi_card("Injected Artifacts (-1)", str(artifact_count), "#E65100").text())
            self.lbl_orphans.setText(self._create_kpi_card("Orphan IDs (Missing)", str(orphan_count), "#C62828").text())
            
            # Enable all save/export buttons
            self.export_btn.setEnabled(True)
            self.top_save_btn.setEnabled(True)
            self.export_action.setEnabled(True)
            self.run_btn.setText("Execute Catalog Cross-Check")
            self.run_btn.setEnabled(True)
            
            # --- INTERACTIVE SAVE PROMPT TRIGGER ---
            # If checked, automatically pop up the save dialog as soon as analysis completes!
            if self.prompt_save_chk.isChecked():
                self.export_audit()
            
        except Exception as e:
            QMessageBox.critical(self, "Auditing Error", f"Failed to parse datasets:\n{str(e)}")
            self.run_btn.setText("Execute Catalog Cross-Check")
            self.run_btn.setEnabled(True)

    def export_audit(self):
        if not self.audit_report_data: 
            return
        
        # Default starting location sets to the directory of the analyzed CSV
        start_dir = os.path.dirname(self.out_path_input.text()) if self.out_path_input.text() else os.path.expanduser("~")
        default_name = os.path.join(start_dir, "enriched_crosscheck_results.csv")
        
        # Opens the standard Windows Save As dialog prompt
        path, _ = QFileDialog.getSaveFileName(self, "Save Enriched Verification Results As...", default_name, "CSV Files (*.csv)")
        
        if path:
            try:
                pd.DataFrame(self.audit_report_data).to_csv(path, index=False)
                self.statusBar().showMessage(f"Successfully saved to: {path}", 8000)
                QMessageBox.information(self, "Export Complete", f"Enriched results successfully saved to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"Could not write CSV:\n{str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    window = StarCatalogValidator()
    window.show()
    sys.exit(app.exec_())