# gui_aggregator.py
# This code is meant to be run on Windows.
import sys
import os
import glob
import time
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox, QGroupBox, 
    QGridLayout, QStyleFactory, QProgressBar
)
from PyQt5.QtCore import Qt

class LedgerAggregatorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Massive Star Ledger Aggregator")
        self.setGeometry(150, 150, 800, 350)
        self.init_ui()

    def init_ui(self):
        # 1. Path Selection Group
        path_group = QGroupBox("1. Select Datasets")
        path_layout = QGridLayout()
        
        # Input CSV Directory
        self.input_dir_edit = QLineEdit()
        self.input_dir_edit.setPlaceholderText("Select folder containing the 000000X.csv image files...")
        input_btn = QPushButton("Browse Folder")
        input_btn.clicked.connect(self.browse_input_dir)
        
        # Master Catalog
        self.cat_path_edit = QLineEdit()
        self.cat_path_edit.setPlaceholderText("Select Master Reference Catalog CSV (e.g., GAIADR3_cache.csv)...")
        cat_btn = QPushButton("Browse File")
        cat_btn.clicked.connect(self.browse_catalog)
        
        # Output Aggregated CSV
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Select where to save the final aggregated ledger...")
        output_btn = QPushButton("Save As...")
        output_btn.clicked.connect(self.browse_output)
        
        path_layout.addWidget(QLabel("Image CSV Folder:"), 0, 0)
        path_layout.addWidget(self.input_dir_edit, 0, 1)
        path_layout.addWidget(input_btn, 0, 2)
        
        path_layout.addWidget(QLabel("Master Catalog CSV:"), 1, 0)
        path_layout.addWidget(self.cat_path_edit, 1, 1)
        path_layout.addWidget(cat_btn, 1, 2)
        
        path_layout.addWidget(QLabel("Output Ledger CSV:"), 2, 0)
        path_layout.addWidget(self.output_path_edit, 2, 1)
        path_layout.addWidget(output_btn, 2, 2)
        path_group.setLayout(path_layout)
        
        # 2. Execution Group
        exec_group = QGroupBox("2. Aggregate Data")
        exec_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        
        self.status_lbl = QLabel("Ready.")
        self.status_lbl.setStyleSheet("color: #333;")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        
        self.run_btn = QPushButton("Execute Aggregation")
        self.run_btn.setFixedHeight(40)
        self.run_btn.setStyleSheet("background-color: #1565C0; color: white; font-weight: bold; font-size: 14px;")
        self.run_btn.clicked.connect(self.run_aggregation)
        
        exec_layout.addWidget(self.progress_bar)
        exec_layout.addWidget(self.status_lbl)
        exec_layout.addWidget(self.run_btn)
        exec_group.setLayout(exec_layout)
        
        # Master Layout Assembly
        main_layout = QVBoxLayout()
        main_layout.addWidget(path_group)
        main_layout.addWidget(exec_group)
        main_layout.addStretch()
        
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    # --- UI Interactions ---
    def browse_input_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image CSV Directory", os.path.expanduser("~"))
        if folder:
            self.input_dir_edit.setText(folder)

    def browse_catalog(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Master Catalog", os.path.expanduser("~"), "CSV Files (*.csv)")
        if filename:
            self.cat_path_edit.setText(filename)

    def browse_output(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Aggregated Ledger As...", os.path.expanduser("~"), "CSV Files (*.csv)")
        if filename:
            self.output_path_edit.setText(filename)

    # --- Core Logic ---
    def run_aggregation(self):
        input_dir = self.input_dir_edit.text().strip()
        cat_path = self.cat_path_edit.text().strip()
        out_path = self.output_path_edit.text().strip()
        
        if not input_dir or not cat_path or not out_path:
            QMessageBox.warning(self, "Missing Paths", "Please ensure all three paths are selected before executing.")
            return
            
        csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
        if not csv_files:
            QMessageBox.critical(self, "No Files Found", f"Could not find any CSV files in:\n{input_dir}")
            return

        # Lock UI during execution
        self.run_btn.setEnabled(False)
        self.progress_bar.setMaximum(len(csv_files))
        self.progress_bar.setValue(0)
        start_time = time.time()
        
        try:
            # 1. Load Master Catalog
            self.status_lbl.setText("Loading Master Catalog into RAM... (This may take a moment)")
            QApplication.processEvents() # Keep GUI responsive
            
            sample = pd.read_csv(cat_path, nrows=0)
            cat_id_col = 'source_id' if 'source_id' in sample.columns else ('star_id' if 'star_id' in sample.columns else sample.columns[0])
            ra_col = next((c for c in sample.columns if 'ra' in c.lower()), None)
            dec_col = next((c for c in sample.columns if 'de' in c.lower() or 'dec' in c.lower()), None)
            
            if not ra_col or not dec_col:
                raise ValueError("Could not automatically identify RA and DEC columns in the Master Catalog.")

            cat_df = pd.read_csv(cat_path, dtype={cat_id_col: str})
            cat_df['__clean_id'] = cat_df[cat_id_col].astype(str).str.strip().str.replace('.0', '', regex=False)
            cat_df.set_index('__clean_id', inplace=True)
            
            # 2. Iterate through image CSVs
            all_dataframes = []
            total_stars = 0
            
            for i, file_path in enumerate(csv_files):
                self.status_lbl.setText(f"Processing image {i+1} of {len(csv_files)}...")
                self.progress_bar.setValue(i + 1)
                QApplication.processEvents() # Update progress bar visually
                
                img_id = os.path.basename(file_path).replace('.csv', '')
                img_df = pd.read_csv(file_path, dtype={'star_id': str, 'source_id': str})
                
                if img_df.empty: continue
                
                img_id_col = 'star_id' if 'star_id' in img_df.columns else ('source_id' if 'source_id' in img_df.columns else img_df.columns[0])
                img_df['__clean_id'] = img_df[img_id_col].astype(str).str.strip().str.replace('.0', '', regex=False)
                
                # Vectorized lookup
                img_df['True_RA'] = img_df['__clean_id'].map(cat_df[ra_col])
                img_df['True_DEC'] = img_df['__clean_id'].map(cat_df[dec_col])
                
                # Tag Artifacts and Orphans
                img_df.loc[img_df['__clean_id'] == '-1', ['True_RA', 'True_DEC']] = 'Artifact'
                img_df['True_RA'] = img_df['True_RA'].fillna('Orphan/Missing')
                img_df['True_DEC'] = img_df['True_DEC'].fillna('Orphan/Missing')
                
                img_df.insert(0, 'source_image_id', img_id)
                img_df.drop(columns=['__clean_id'], inplace=True)
                
                all_dataframes.append(img_df)
                total_stars += len(img_df)
                
            # 3. Concatenate and Save
            self.status_lbl.setText("Concatenating files and writing to disk...")
            QApplication.processEvents()
            
            if all_dataframes:
                massive_df = pd.concat(all_dataframes, ignore_index=True)
                massive_df.to_csv(out_path, index=False)
                
                elapsed = time.time() - start_time
                self.status_lbl.setText(f"Complete! Processed {total_stars} stars in {elapsed:.1f} seconds.")
                self.status_lbl.setStyleSheet("color: #2E7D32; font-weight: bold;")
                QMessageBox.information(self, "Success", f"Successfully aggregated {total_stars} stars into:\n{out_path}")
            else:
                self.status_lbl.setText("No data found to aggregate.")
                
        except Exception as e:
            self.status_lbl.setText("Error occurred during processing.")
            self.status_lbl.setStyleSheet("color: #C62828; font-weight: bold;")
            QMessageBox.critical(self, "Aggregation Error", f"An error occurred:\n{str(e)}")
            
        finally:
            self.run_btn.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    window = LedgerAggregatorGUI()
    window.show()
    sys.exit(app.exec_())