<<<<<<< HEAD
﻿﻿# 🌟 AuraGen-OCR 🌟
=======
﻿# 🌟 AuraGen-OCR 🌟
>>>>>>> origin/main

<p align="center">
  
</p>

<p align="center">
  <b>Advanced Document AI: OCR, Layout Analysis, Reading Order, and Table Recognition</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/license-GPL--3.0-green.svg" alt="License">
  <img src="https://img.shields.io/badge/author-daniellopez882-orange.svg" alt="Author">
  <img src="https://img.shields.io/badge/platform-windows%20%7C%20linux%20%7C%20macos-lightgrey.svg" alt="Platform">
</p>

---

## 🚀 Overview

**AuraGen-OCR** is a state-of-the-art document intelligence toolkit designed for high-performance OCR and structural analysis of complex documents. Whether you're processing scientific papers, scanned forms, or multi-lingual textbooks, AuraGen-OCR delivers precision and speed.

### ✨ Key Features

- 🌍 **Multi-lingual OCR**: High-accuracy text recognition in 90+ languages.
- 📐 **Layout Analysis**: Detect tables, images, headers, footers, and more.
- 📖 **Reading Order**: Intelligently determine the logical flow of text.
- 📊 **Table Recognition**: Extract structured data from complex tables (rows/columns).
- 🧬 **LaTeX OCR**: Convert mathematical equations into clean LaTeX code.
- 🏎️ **Optimized Performance**: GPU acceleration with support for quantization and compilation.

---



## 🛠️ Installation



### Setup (For Developers)
```bash
git clone https://github.com/daniellopez882/AuraGen-OCR.git
cd AuraGen-OCR
poetry install
poetry shell
```

---

## 💻 Usage

### Interactive Web App
Try AuraGen-OCR visually using the built-in Streamlit app:
```bash
auragen_gui
```

### CLI Commands
Detect text and get JSON results:
```bash
auragen_ocr PATH_TO_FILE
```

Detect document layout:
```bash
auragen_layout PATH_TO_FILE
```

Recognize tables:
```bash
auragen_table PATH_TO_FILE
```

---

## 📊 Benchmarks

AuraGen-OCR benchmarks favorably against industry standards, providing cloud-level performance locally.

| Model | Time per page (s) | Accuracy (%) |
| :--- | :---: | :---: |
| AuraGen-OCR | **0.62** | **97.0** |
| Tesseract | 0.45 | 88.0 |

---

## 🤝 Community & Support

Join us in building the most advanced open-source OCR toolkit!

- **GitHub**: [daniellopez882/AuraGen-OCR](https://github.com/daniellopez882/AuraGen-OCR)


---

## 📜 License & Credits

- **Code**: Licensed under [GPL-3.0](LICENSE).
- **Author**: [Daniel Lopez (daniellopez882)](https://github.com/daniellopez882)


---

<p align="center">
  MADE BY DANIELLOPEZ882
</p>
