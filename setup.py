from setuptools import setup, find_packages

setup(
    name="card-detection-app",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "flask",
        "werkzeug",
        "pandas",
        "numpy",
        "pillow",
        "torch --index-url https://download.pytorch.org/whl/cpu",
        "torchvision --index-url https://download.pytorch.org/whl/cpu", 
        "ultralytics",
        "opencv-python-headless",
        "paddleocr",
        "paddlepaddle",
        "openai",
        "jiwer",
        "python-dotenv"
    ],
    python_requires=">=3.9",
)