import os
import sys

# --- HIGH-SPEED RENDERING ENGINE ---
import cv2
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from astropy import units as u
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.wcs import WCS
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStyleFactory,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class FitsTelemetryExtractor(QMainWindow):

  def __init__(self):
    super().__init__()
    self.setWindowTitle("ZWO FITS Hardware Telemetry & Manifest Generator")
    self.setGeometry(150, 150, 1300, 720)

    self.telemetry_data = []
    self.init_ui()

  def init_ui(self):
    # 1. Directory Selection Group
    file_group = QGroupBox(
        "1. Select Hardware Target Directory & Output Options"
    )
    file_layout = QGridLayout()

    self.dir_input = QLineEdit()
    self.dir_input.setPlaceholderText(
        "Select folder containing .fits images..."
    )
    dir_btn = QPushButton("Browse Folder")
    dir_btn.clicked.connect(self.browse_directory)

    self.out_input = QLineEdit()
    self.out_input.setText("dataset_manifest.csv")

    self.chk_save_png = QCheckBox(
        "Generate Annotated PNGs with Detected Stars Circled (1:1 Native"
        " Resolution)"
    )
    self.chk_save_png.setChecked(True)
    self.chk_save_png.setStyleSheet(
        "font-weight: bold; color: #0277BD; margin-top: 4px;"
    )

    self.chk_save_csv = QCheckBox(
        "Generate Per-Image Star Catalog CSVs with Calculated RA/DEC (saves to"
        " /csv folder)"
    )
    self.chk_save_csv.setChecked(True)
    self.chk_save_csv.setStyleSheet(
        "font-weight: bold; color: #2E7D32; margin-top: 2px;"
    )

    file_layout.addWidget(QLabel("Target Directory:"), 0, 0)
    file_layout.addWidget(self.dir_input, 0, 1)
    file_layout.addWidget(dir_btn, 0, 2)

    file_layout.addWidget(QLabel("Output CSV Name:"), 1, 0)
    file_layout.addWidget(self.out_input, 1, 1)

    file_layout.addWidget(self.chk_save_png, 2, 0, 1, 3)
    file_layout.addWidget(self.chk_save_csv, 3, 0, 1, 3)
    file_group.setLayout(file_layout)

    # 2. Execution Trigger
    self.run_btn = QPushButton("Scan Directory & Generate Universal Manifest")
    self.run_btn.setFixedHeight(38)
    self.run_btn.setStyleSheet(
        "background-color: #0277BD; color: white; font-weight: bold; font-size:"
        " 14px;"
    )
    self.run_btn.clicked.connect(self.run_extraction)
    self.run_btn.setEnabled(False)

    # 3. KPI Summary Banner
    kpi_layout = QHBoxLayout()
    self.lbl_found = self._create_kpi_card("FITS Files Found", "0", "#1565C0")
    self.lbl_success = self._create_kpi_card(
        "Manifest Rows Generated", "0", "#2E7D32"
    )
    self.lbl_failed = self._create_kpi_card("Read Errors", "0", "#C62828")

    kpi_layout.addWidget(self.lbl_found)
    kpi_layout.addWidget(self.lbl_success)
    kpi_layout.addWidget(self.lbl_failed)

    # 4. Results Ledger Table
    self.table = QTableWidget()
    self.table.setColumnCount(12)
    self.table.setHorizontalHeaderLabels([
        "Image ID",
        "Group",
        "RA (deg)",
        "DEC (deg)",
        "Roll",
        "Focal (mm)",
        "Exp (s)",
        "Stars",
        "Med SNR",
        "BG Mean",
        "BG Std",
        "Anomalies",
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
    lbl = QLabel(
        f"<div style='text-align: center;'><small>{title}</small><br><b"
        f" style='font-size: 20px; color: {color};'>{val}</b></div>"
    )
    lbl.setFrameStyle(QLabel.StyledPanel | QLabel.Raised)
    lbl.setStyleSheet(
        "background-color: #f8f9fa; padding: 6px; border-radius: 4px; border:"
        " 1px solid #ddd;"
    )
    return lbl

  def browse_directory(self):
    directory = QFileDialog.getExistingDirectory(
        self,
        "Select Directory with FITS Files",
        os.path.expanduser("~"),
    )
    if directory:
      self.dir_input.setText(directory)
      self._check_ready_state()

  def _check_ready_state(self):
    if self.dir_input.text() and self.out_input.text():
      self.run_btn.setEnabled(True)

  def _parse_coordinate_to_decimal(self, val, is_ra=True):
    if val == "NA" or val is None or pd.isna(val):
      return "NA"
    try:
      return str(round(float(val), 4))
    except ValueError:
      try:
        unit = u.hourangle if is_ra else u.deg
        angle = Angle(str(val), unit=unit)
        return str(round(angle.deg, 4))
      except Exception:
        return "NA"

  def _pixel_to_world_coords(self, star_positions, header, img_shape):
    xs = np.array([pos[1] for pos in star_positions], dtype=np.float64)
    ys = np.array([pos[0] for pos in star_positions], dtype=np.float64)

    try:
      wcs = WCS(header)
      if wcs.has_celestial:
        world = wcs.pixel_to_world_values(xs, ys)
        return np.round(world[0], 6), np.round(world[1], 6)
    except Exception:
      pass

    try:
      raw_ra = header.get(
          "OBJCTRA", header.get("RA", header.get("CRVAL1", "NA"))
      )
      raw_dec = header.get(
          "OBJCTDEC", header.get("DEC", header.get("CRVAL2", "NA"))
      )
      ra_center = float(self._parse_coordinate_to_decimal(raw_ra, is_ra=True))
      dec_center = float(
          self._parse_coordinate_to_decimal(raw_dec, is_ra=False)
      )

      focal_mm = float(header.get("FOCALLEN", header.get("FOCLEN", 100.0)))
      pix_um = float(header.get("XPIXSZ", header.get("PIXSIZE1", 2.9)))

      roll_val = header.get(
          "ROLL",
          header.get("ROTANG", header.get("CROTA2", header.get("PA", "NA"))),
      )
      roll_deg = float(roll_val) if roll_val != "NA" else 0.0

      deg_per_pix = (206.264806 * (pix_um / focal_mm)) / 3600.0
      height, width = img_shape
      dx = (xs - (width / 2.0)) * deg_per_pix
      dy = (ys - (height / 2.0)) * deg_per_pix

      roll_rad = np.radians(roll_deg)
      dx_rot = dx * np.cos(roll_rad) - dy * np.sin(roll_rad)
      dy_rot = dx * np.sin(roll_rad) + dy * np.cos(roll_rad)

      dec_deg = dec_center + dy_rot
      ra_deg = ra_center + (dx_rot / np.cos(np.radians(dec_center)))

      return np.round(ra_deg % 360.0, 6), np.round(dec_deg, 6)
    except Exception:
      return np.zeros_like(xs), np.zeros_like(ys)

  def _save_annotated_png_fast(
      self, img_float, star_positions, output_path, bg_mean, bg_std
  ):
    try:
      height, width = img_float.shape
      min_dim = min(height, width)
      radius = max(10, int(min_dim / 150))
      thickness = max(1, int(radius / 6))

      vmin = bg_mean - (1.0 * bg_std)
      vmax = bg_mean + (10.0 * bg_std) if bg_std > 0 else np.max(img_float)
      if vmax <= vmin:
        vmax = vmin + 1.0

      img_norm = np.clip((img_float - vmin) / (vmax - vmin), 0.0, 1.0)
      img_uint8 = (img_norm * 255.0).astype(np.uint8)
      img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

      for y, x in star_positions:
        cv2.circle(
            img_bgr,
            (int(round(x)), int(round(y))),
            radius=radius,
            color=(0, 0, 255),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

      img_bgr_flipped = cv2.flip(img_bgr, 0)
      cv2.imwrite(output_path, img_bgr_flipped)
    except Exception:
      pass

  def _calculate_merline_howell_telemetry(
      self, img_data, header, png_output_path=None, csv_output_path=None
  ):
    if img_data is None:
      return "NA", "NA", "NA", "NA"
    try:
      img_float = img_data.astype(np.float32)
      total_pixels = img_float.size

      bg_mean = float(np.median(img_float))
      mad = float(np.median(np.abs(img_float - bg_mean)))
      bg_std = float(1.4826 * mad)
      if bg_std == 0:
        bg_std = max(1.0, float(np.std(img_float)))

      bg_variance = bg_std**2
      threshold = bg_mean + (5.0 * bg_std)

      structure = np.ones((3, 3), dtype=int)
      labeled_array, num_stars = ndi.label(
          img_float > threshold, structure=structure
      )

      if num_stars > 0:
        star_positions = ndi.maximum_position(
            img_float, labeled_array, index=range(1, num_stars + 1)
        )
        if isinstance(star_positions, tuple):
          star_positions = [star_positions]

        n_px = np.array(
            ndi.sum(
                np.ones_like(img_float),
                labeled_array,
                index=range(1, num_stars + 1),
            )
        )
        net_signal_img = img_float - bg_mean
        N_star = np.array(
            ndi.sum(
                net_signal_img, labeled_array, index=range(1, num_stars + 1)
            )
        )

        total_star_pixels = np.sum(n_px)
        n_B = max(1.0, float(total_pixels - total_star_pixels))

        bg_term = n_px * (1.0 + (n_px / n_B)) * bg_variance
        radicand = np.maximum(1e-6, N_star + bg_term)

        star_snrs = N_star / np.sqrt(radicand)
        median_snr = float(np.median(star_snrs))

        if csv_output_path:
          calc_ra, calc_dec = self._pixel_to_world_coords(
              star_positions, header, img_float.shape
          )

          star_df = pd.DataFrame({
              "star_id": np.arange(1, num_stars + 1),
              "pixel_x": np.round([p[1] for p in star_positions], 2),
              "pixel_y": np.round([p[0] for p in star_positions], 2),
              "ra_deg": calc_ra,
              "dec_deg": calc_dec,
              "integrated_signal_e": np.round(N_star, 2),
              "snr_eq61": np.round(star_snrs, 2),
              "area_px": n_px,
          })
          star_df.to_csv(csv_output_path, index=False)
      else:
        median_snr = 0.0
        if csv_output_path:
          pd.DataFrame(
              columns=[
                  "star_id",
                  "pixel_x",
                  "pixel_y",
                  "ra_deg",
                  "dec_deg",
                  "integrated_signal_e",
                  "snr_eq61",
                  "area_px",
              ]
          ).to_csv(csv_output_path, index=False)

      if png_output_path:
        self._save_annotated_png_fast(
            img_float, star_positions, png_output_path, bg_mean, bg_std
        )

      return (
          str(num_stars),
          str(round(median_snr, 2)),
          str(round(bg_mean, 2)),
          str(round(bg_std, 2)),
      )
    except Exception:
      return "NA", "NA", "NA", "NA"

  def run_extraction(self):
    target_directory = self.dir_input.text()
    output_csv_name = self.out_input.text()
    save_pngs = self.chk_save_png.isChecked()
    save_csvs = self.chk_save_csv.isChecked()

    if not os.path.exists(target_directory):
      QMessageBox.critical(
          self, "Error", "The selected directory does not exist."
      )
      return

    self.run_btn.setText("Scanning & Generating Manifest... Please Wait")
    self.run_btn.setEnabled(False)
    QApplication.processEvents()

    try:
      fits_files = [
          f
          for f in os.listdir(target_directory)
          if f.lower().endswith((".fits", ".fit"))
      ]

      if not fits_files:
        QMessageBox.information(
            self,
            "No Files",
            f"No .fits or .fit files found in:\n{target_directory}",
        )
        self.run_btn.setText("Scan Directory & Generate Universal Manifest")
        self.run_btn.setEnabled(True)
        return

      png_dir = os.path.join(target_directory, "png")
      csv_dir = os.path.join(target_directory, "csv")
      if save_pngs:
        os.makedirs(png_dir, exist_ok=True)
      if save_csvs:
        os.makedirs(csv_dir, exist_ok=True)

      self.telemetry_data = []
      self.table.setRowCount(0)

      self.lbl_found.setText(
          self._create_kpi_card(
              "FITS Files Found", str(len(fits_files)), "#1565C0"
          ).text()
      )
      QApplication.processEvents()

      success_cnt = 0
      failed_cnt = 0
      dataset_group_name = os.path.basename(os.path.normpath(target_directory))

      raw_ids = [os.path.splitext(f)[0] for f in fits_files]
      generate_unique_ids = len(set(raw_ids)) != len(fits_files)

      for idx, filename in enumerate(fits_files):
        filepath = os.path.join(target_directory, filename)
        try:
          with fits.open(filepath, mode="readonly") as hdul:
            header = hdul[0].header
            img_data = hdul[0].data if len(hdul) > 0 else None

            if generate_unique_ids:
              img_id = f"{idx+1:07d}"
            else:
              img_id = os.path.splitext(filename)[0]

            raw_ra = header.get(
                "OBJCTRA", header.get("RA", header.get("CRVAL1", "NA"))
            )
            raw_dec = header.get(
                "OBJCTDEC", header.get("DEC", header.get("CRVAL2", "NA"))
            )

            ra_val = self._parse_coordinate_to_decimal(raw_ra, is_ra=True)
            dec_val = self._parse_coordinate_to_decimal(raw_dec, is_ra=False)

            roll_val = header.get(
                "ROLL",
                header.get(
                    "ROTANG", header.get("CROTA2", header.get("PA", "NA"))
                ),
            )
            if roll_val != "NA":
              try:
                roll_val = str(round(float(roll_val), 2))
              except ValueError:
                roll_val = "NA"

            focal_val = header.get("FOCALLEN", header.get("FOCLEN", "NA"))
            exp_val = header.get("EXPTIME", "NA")

            png_path = os.path.join(png_dir, f"{img_id}.png") if save_pngs else None
            csv_path = os.path.join(csv_dir, f"{img_id}.csv") if save_csvs else None

            stars_val, snr_val, bg_mean_val, bg_std_val = (
                self._calculate_merline_howell_telemetry(
                    img_data, header, png_path, csv_path
                )
            )

            file_info = {
                            "image_id": str(img_id),
                            "dataset_group": dataset_group_name,
                            "ra": str(ra_val),
                            "dec": str(dec_val),  # <--- FIXED: Changed "dec' to "dec"
                            "camera_roll": str(roll_val),
                            "focal_length_mm": str(focal_val),
                            "exposure_time_s": str(exp_val),
                            "total_stars": str(stars_val),
                            "median_image_snr": str(snr_val),
                            "bg_mean_e": str(bg_mean_val),
                            "bg_std_e": str(bg_std_val),
                            "distorted_stars": "NA",
                            "dropped_stars": "NA",
                            "false_stars": "NA",
                            "smear_px": "NA",
                            "anom_lens_on": False,
                            "anom_false_on": False,
                            "anom_drop_on": False,
                            "anom_pos_on": False,
                            "anom_mag_on": False,
                            "anom_smear_on": False,
                        }

            self.telemetry_data.append(file_info)

            row_idx = self.table.rowCount()
            self.table.insertRow(row_idx)

            items = [
                QTableWidgetItem(file_info["image_id"]),
                QTableWidgetItem(file_info["dataset_group"]),
                QTableWidgetItem(file_info["ra"]),
                QTableWidgetItem(file_info["dec"]),
                QTableWidgetItem(file_info["camera_roll"]),
                QTableWidgetItem(file_info["focal_length_mm"]),
                QTableWidgetItem(file_info["exposure_time_s"]),
                QTableWidgetItem(file_info["total_stars"]),
                QTableWidgetItem(file_info["median_image_snr"]),
                QTableWidgetItem(file_info["bg_mean_e"]),
                QTableWidgetItem(file_info["bg_std_e"]),
                QTableWidgetItem("None (Physical Capture)"),
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

      self.lbl_success.setText(
          self._create_kpi_card(
              "Manifest Rows Generated", str(success_cnt), "#2E7D32"
          ).text()
      )
      self.lbl_failed.setText(
          self._create_kpi_card(
              "Read Errors",
              str(failed_cnt),
              "#C62828" if failed_cnt > 0 else "#607D8B",
          ).text()
      )

      if self.telemetry_data:
        galsim_columns = [
            "image_id",
            "dataset_group",
            "ra",
            "dec",
            "camera_roll",
            "focal_length_mm",
            "exposure_time_s",
            "total_stars",
            "median_image_snr",
            "bg_mean_e",
            "bg_std_e",
            "distorted_stars",
            "dropped_stars",
            "false_stars",
            "smear_px",
            "anom_lens_on",
            "anom_false_on",
            "anom_drop_on",
            "anom_pos_on",
            "anom_mag_on",
            "anom_smear_on",
        ]
        df = pd.DataFrame(self.telemetry_data)[galsim_columns]
        output_path = os.path.join(target_directory, output_csv_name)
        df.to_csv(output_path, index=False)

        msg = (
            f"Successfully generated universal manifest for {success_cnt}"
            f" physical captures.\n\nManifest saved to:\n{output_path}"
        )
        if save_pngs:
          msg += f"\n\nAnnotated PNGs saved to:\n{png_dir}"
        if save_csvs:
          msg += f"\n\nPer-Image Star Catalogs saved to:\n{csv_dir}"
        QMessageBox.information(self, "Manifest Complete", msg)

    except Exception as e:
      QMessageBox.critical(
          self, "System Error", f"A fatal error occurred:\n{str(e)}"
      )

    finally:
      self.run_btn.setText("Scan Directory & Generate Universal Manifest")
      self.run_btn.setEnabled(True)


if __name__ == "__main__":
  app = QApplication(sys.argv)
  app.setStyle(QStyleFactory.create("Fusion"))
  window = FitsTelemetryExtractor()
  window.show()
  sys.exit(app.exec_())