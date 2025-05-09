# 📊 Real-Time Option Pricing with PINN


[https://bsm-pinn.streamlit.app/](https://bsm-pinn.streamlit.app/)


A **Streamlit-based web application** that provides advanced tools for option pricing using both traditional and modern approaches:

1. **Black-Scholes Model** - The classical analytical solution
2. **Physics-Informed Neural Network (PINN) Model** - A modern deep learning approach

This application is designed for financial analysts, researchers, and students interested in exploring both traditional and cutting-edge approaches to option pricing.

---

## 📒 Table of Contents

1. [Features](#-features)
2. [Technology Stack](#-technology-stack)
3. [Installation](#-installation)
4. [Usage](#-usage)
5. [Application Structure](#-application-structure)
6. [Models Overview](#-models-overview)
7. [Contributing](#-contributing)
8. [Contact](#-contact)

---

## 🌟 Features

- **Interactive UI**: User-friendly interface for model selection and parameter input
- **Dual-Model Approach**: Compare traditional Black-Scholes with modern PINN implementation
- **Real-Time Calculations**: Instant computation of option prices
- **Visualization Tools**: Dynamic plots for model comparison and analysis
- **Modern Architecture**: Integration of deep learning with traditional financial models

---

## 🛠️ Technology Stack

- **Python**: Core programming language
- **Streamlit**: Web framework for building the interactive UI
- **TensorFlow**: Deep learning framework for PINN implementation
- **NumPy**: For numerical computations
- **SciPy**: Statistical functions
- **Matplotlib**: For data visualization
- **Seaborn**: For enhanced visualizations

---

## 📦 Installation

1. **Clone the Repository**

```bash
git clone <your-repository-url>
cd <repository-name>
```

2. **Create a Virtual Environment (Recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. **Install Required Packages**

```bash
pip install -r requirements.txt
```

4. **Run the Application**

```bash
streamlit run app.py
```

---

## 🔍 Usage

1. Launch the application using the command above
2. Select your preferred model from the main page:
   - **Black-Scholes Model**: Traditional analytical solution
   - **PINN Model**: Physics-Informed Neural Network approach
3. Input relevant parameters for your analysis
4. View the results and visualizations
5. Compare the performance of both models

---

## 🔄 Application Structure

```plaintext
project/
│
├── app.py                        # Main application file
├── pages/
│   ├── 1_black-scholes-model.py  # Black-Scholes model implementation
│   └── 2_pinn-model.py          # PINN model implementation
├── code-bsm.py                   # Black-Scholes model core calculations
├── static_pinn_results.npz       # Pre-computed PINN results
├── requirements.txt              # Python dependencies
└── README.md                    # Project documentation
```

---

## 📈 Models Overview

### 1. **Black-Scholes Model**
The classical approach to option pricing that provides analytical solutions for European options.

- **Key Features:**
  - Analytical solution for European options
  - Closed-form pricing formula
  - Well-established theoretical foundation
  - Fast computation time

### 2. **Physics-Informed Neural Network (PINN) Model**
A modern deep learning approach that combines neural networks with financial physics.

- **Key Features:**
  - Deep learning-based solution
  - Incorporates financial physics constraints
  - Can handle complex market conditions
  - Potential for improved accuracy in certain scenarios

---

## 💪 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add new feature'`)
5. Push to the branch (`git push origin feature-branch`)
6. Create a Pull Request

---

## 📧 Contact

[Sri Sahitih S]  
[sahithi-sss] | [https://github.com/sahithi-sss]

---

> This application is designed for educational and research purposes. The PINN model represents an experimental approach to option pricing and should be used with appropriate caution in real-world applications. 
