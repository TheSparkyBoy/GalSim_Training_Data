"""
Portable ASTAP Star Extraction Module (In-Memory Array Return)
Invokes ASTAP via CLI subprocess and returns extracted star records directly 
as a native Python array (list of dictionaries) to the calling program.
Supports both .fits/.fit and .png image inputs.
"""

import os
import subprocess
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def extract_stars_to_array(
    image_path=None,
    fits_path=None,
    astap_exe=r"C:\Program Files\astap\astap.exe" if os.name == 'nt' else "/usr/bin/astap",
    ra_hint=None,
    dec_hint=None,
    fov_height_deg=1.5,
    show_gui=False
):
    target_path = image_path if image_path is not None else fits_path
    if target_path is None:
        return {'success': False, 'error': "No image path provided to extract_stars_to_array.", 'star_records': []}

    norm_path = os.path.normpath(os.path.abspath(target_path))
    if not os.path.exists(norm_path):
        return {'success': False, 'error': f"Image file not found: {norm_path}", 'star_records': []}

    ext = os.path.splitext(norm_path)[1].lower()
    if ext not in ('.fits', '.fit', '.png'):
        return {'success': False, 'error': f"Unsupported image format '{ext}'. Supported formats: .fits, .fit, .png", 'star_records': []}

    if not os.path.exists(astap_exe):
        return {'success': False, 'error': f"ASTAP executable not found at: {astap_exe}", 'star_records': []}

    base_name = os.path.splitext(norm_path)[0]
    ini_path = f"{base_name}.ini"
    wcs_path = f"{base_name}.wcs"
    astap_csv_path = f"{base_name}.csv"

    for p in (ini_path, wcs_path, astap_csv_path):
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    cmd = [astap_exe, "-f", norm_path, "-r", "30", "-z", "0", "-extract2", "5", "-update"]

    if ra_hint is not None and dec_hint is not None and ra_hint != "NA" and dec_hint != "NA":
        try:
            ra_hours = float(ra_hint) / 15.0
            spd_deg = float(dec_hint) + 90.0
            cmd.extend(["-ra", f"{ra_hours:.4f}", "-spd", f"{spd_deg:.4f}"])
        except (TypeError, ValueError):
            pass

    try:
        safe_fov = float(fov_height_deg) if fov_height_deg is not None else 1.5
    except (TypeError, ValueError):
        safe_fov = 1.5
    cmd.extend(["-fov", f"{safe_fov:.3f}"])

    startupinfo = None
    creationflags = 0
    if show_gui and os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 1
    elif not show_gui and os.name == 'nt':
        creationflags = subprocess.CREATE_NO_WINDOW

    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=40, text=True, creationflags=creationflags, startupinfo=startupinfo
        )
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': "ASTAP execution timed out after 40 seconds.", 'star_records': []}
    except Exception as e:
        return {'success': False, 'error': f"ASTAP subprocess exception: {str(e)}", 'star_records': []}

    solution = {
        'success': True,
        'solved': False,
        'ra': None,
        'dec': None,
        'roll': None,
        'scale': None,
        'star_records': []
    }

    if os.path.exists(ini_path):
        try:
            with open(ini_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if '=' in line:
                        parts = line.strip().split('=', 1)
                        key = parts[0].strip().upper()
                        val = parts[1].strip()
                        if not val or val.upper() == 'NONE':
                            continue
                        if key == 'PLTSOLVD' and val == 'T':
                            solution['solved'] = True
                        elif key == 'CRVAL1':
                            try: solution['ra'] = round(float(val), 6)
                            except (TypeError, ValueError): pass
                        elif key == 'CRVAL2':
                            try: solution['dec'] = round(float(val), 6)
                            except (TypeError, ValueError): pass
                        elif key == 'CROTA2':
                            try: solution['roll'] = round(float(val), 2)
                            except (TypeError, ValueError): pass
                        elif key == 'SCALE':
                            try: solution['scale'] = float(val)
                            except (TypeError, ValueError): pass
        except Exception:
            pass

    wcs_obj = None
    if ext in ('.fits', '.fit'):
        try:
            with fits.open(norm_path, mode='readonly') as hdul:
                wcs_obj = WCS(hdul[0].header)
                if not wcs_obj.is_celestial:
                    wcs_obj = None
        except Exception:
            wcs_obj = None

    if wcs_obj is None and os.path.exists(wcs_path):
        try:
            wcs_obj = WCS(wcs_path)
            if not wcs_obj.is_celestial:
                wcs_obj = None
        except Exception:
            wcs_obj = None

    if os.path.exists(astap_csv_path):
        try:
            import pandas as pd
            df_astap = pd.read_csv(astap_csv_path, sep=None, engine='python')
            star_records = []
            
            for rank, row in df_astap.iterrows():
                try:
                    cx = float(row.iloc[0])
                    cy = float(row.iloc[1])
                    s_flux = float(row.iloc[4]) if len(row) > 4 else 15.0

                    s_ra = solution.get('ra', 0.0) or 0.0
                    s_dec = solution.get('dec', 0.0) or 0.0

                    if wcs_obj is not None:
                        sky_coord = wcs_obj.pixel_to_world(cx, cy)
                        s_ra = round(float(sky_coord.ra.deg), 6)
                        s_dec = round(float(sky_coord.dec.deg), 6)

                    star_records.append({
                        'star_id': int(rank + 1),
                        'x_image': round(cx, 2),
                        'y_image': round(cy, 2),
                        'ra_deg': s_ra,
                        'dec_deg': s_dec,
                        'flux_mag': round(20.0 - 2.5 * np.log10(s_flux), 2),
                        'snr': 10.0,
                        'is_artifact': 0
                    })
                except Exception:
                    continue

            solution['star_records'] = star_records

        except Exception as e:
            solution['error'] = f"Failed parsing ASTAP CSV: {e}"

    for p in (ini_path, wcs_path, astap_csv_path):
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    return solution