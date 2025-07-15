import cv2
import numpy as np
from pyzbar.pyzbar import decode
import hashlib
import re

def extract_qr_data_from_image(image_path: str) -> str:
    """
    Extracts QR data (usually a UPI string) from an image file.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found or unreadable")

        # Optional preprocessing for noisy images
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        decoded_objs = decode(blurred)
        for obj in decoded_objs:
            data = obj.data.decode("utf-8")
            return data
    except Exception as e:
        print(f"⚠️ Error decoding QR: {e}")
        return None

def extract_upi_id_from_qr(qr_string: str) -> str:
    """
    Extracts UPI ID from the QR string.
    Example input: 'upi://pay?pa=merchant@upi&pn=Some+Merchant'
    """
    match = re.search(r"pa=([^&]+)", qr_string)
    return match.group(1) if match else None

def get_qr_hash(qr_string: str) -> str:
    """
    Returns SHA-256 hash of the QR string for unique fingerprinting.
    """
    return hashlib.sha256(qr_string.encode()).hexdigest()

def is_static_qr(qr_string: str) -> bool:
    """
    Detects if the QR is static (missing fields usually present in dynamic QR codes).
    """
    dynamic_fields = ['tr=', 'tn=', 'am=', 'mc=', 'tid=', 'orgid=']
    return not any(field in qr_string for field in dynamic_fields)

def parse_qr_image(image_path: str) -> dict:
    """
    End-to-end pipeline: image → QR string → parsed data
    """
    result = {
        "upi_id": None,
        "qr_data": None,
        "qr_hash": None,
        "is_static_qr": None
    }

    qr_data = extract_qr_data_from_image(image_path)
    if not qr_data:
        return result

    upi_id = extract_upi_id_from_qr(qr_data)
    qr_hash = get_qr_hash(qr_data)
    static_flag = is_static_qr(qr_data)

    result.update({
        "upi_id": upi_id,
        "qr_data": qr_data,
        "qr_hash": qr_hash,
        "is_static_qr": static_flag
    })

    return result
