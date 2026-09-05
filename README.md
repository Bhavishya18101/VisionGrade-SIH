# 🧅 VisionGrade: Automated Mandi Quality Assaying

**Built by Team Mutex for the Smart India Hackathon (SIH) 2026**

VisionGrade is a professional, high-performance web portal designed to bridge the gap between physical agricultural produce and digital e-NAM financial payouts. It provides a "Zero-CapEx" software-centric approach to automated onion sizing and defect detection at APMC Mandis across India.

## 🚀 The Solution
By leveraging a custom-trained YOLOv8 Edge AI pipeline, VisionGrade eliminates the need for expensive industrial hardware. Any mandi operator can use a standard web camera or mobile browser to capture batches of agricultural produce, receive instant quality assaying based on Agmark standards, and automatically generate e-NAM compliance receipts.

## ✨ Key Features
* **Real-Time Edge Inference:** Custom YOLOv8 model optimized to detect Grade A, B, C sizes and spoilage with sub-200ms latency.
* **Zero-CapEx Architecture:** Hardware-agnostic web platform deployable on any device.
* **Automated Audit Ledger:** Live SQLite database integration for transparent tracking of lot IDs, farmer tokens, and total metric tonnage.
* **Vernacular Accessibility:** Dynamic Hindi and English text-to-speech (TTS) audio alerts for immediate batch verification.
* **e-NAM Ready:** Instant, downloadable PDF assaying slips and CSV audit trails for mandi officials.

## 🛠️ Tech Stack
* **Frontend:** Streamlit, Custom Glassmorphism CSS, Plotly (Interactive Data Dashboards)
* **Computer Vision:** Ultralytics YOLOv8, OpenCV (CLAHE preprocessing)
* **Backend:** Python, SQLite3, Pandas
* **Utilities:** gTTS (Audio feedback), FPDF2 (Dynamic receipts)

## 💻 Local Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/VisionGrade-SIH.git](https://github.com/your-username/VisionGrade-SIH.git)
   cd VisionGrade-SIH
