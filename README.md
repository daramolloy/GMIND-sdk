# GMIND-sdk

![GMIND Overview](misc/GMIND-Overall.png)

GMIND-sdk is a toolkit for working with the GMIND ITS infrastructure node dataset. It provides scripts and utilities for image signal processing (ISP), video generation and compression, calibration, validation, and machine learning data loading.

## Features

- **Image Signal Processing (ISP):**
  - Run ISP on RAW images.
  - Output processed images as video files.
  - Compress videos with customizable settings.

- **Calibration & Validation:**
  - Tools for camera and sensor calibration.
  - Scripts for validating dataset alignment and sensor fusion.

- **LIDAR & Camera Reprojection:**
  - Overlay LIDAR point clouds onto camera images using calibration data.

- **PyTorch DataLoader:**
  - Unified DataLoader for all supported data formats.
  - Enables consistent training across work packages and models.
  - Facilitates benchmarking and sensor selection for ITS use cases.

## Getting Started

1. Clone the repository and install dependencies:
   ```sh
   git clone <repo-url>
   cd GMIND-sdk
   pip install -r requirements.txt
   ```
2. Explore the `Calibration/`, `ImageSignalProcessing/`, and `Validation/` folders for scripts and utilities.
3. Use the provided DataLoader for training models with PyTorch.

## Example Workflows

- **Run ISP and Export Video:**
  - Use scripts in `ImageSignalProcessing/` to process RAW images and export videos.
- **Calibrate and Validate Sensors:**
  - Use scripts in `Calibration/` and `Validation/` to calibrate cameras/LIDARs and validate dataset alignment.
- **Train Models:**
  - Use the DataLoader to train models on the dataset and compare sensor performance.

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, new features, or improvements.

## License

[Specify your license here]

---

**GMIND-sdk** aims to provide a complete, consistent, and extensible toolkit for research and development with the GMIND ITS infrastructure node dataset.
