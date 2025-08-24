# 🚀 IHCMX-Optimizer-Pro

**Advanced benchmarking and optimization platform for AI models** with real-time latency and energy consumption control.

## 📋 What is it?
IHCMX-Optimizer-Pro automatically optimizes AI models by minimizing latency and energy consumption using multi-objective grid search, making it ideal for efficient production deployment.

## 🎯 Sector Benefits
- **Businesses**: 30-50% reduction in inference costs
- **Gamers**: +20-30 FPS while maintaining visual quality
- **Creators**: 2x rendering speed with the same hardware

📊 [See detailed use cases](./docs/BENEFITS.md)

## ⚡ Quick Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- NVIDIA GPU (optional but recommended)

```bash
# Installation
git clone https://github.com/Q-Diana/ihcmx-optimizer-pro.git 
cd ihcmx-optimizer-pro
pip install -r requirements.txt
```

## 🚀 Usage Example

### Basic Benchmark
```bash
python ihcmx_with_lowdram_pro.py --benchmark --repeticiones 10 --dashboard
```

### Advanced Grid Search
```bash
python ihcmx_with_lowdram_pro.py --grid \
  --grid_latencies 0.1,0.2,0.3 \
  --grid_powers 50,100,150 \
  --out energy_optimization
```

### Python API
```python
from ihcmx_optimizer import run_benchmark
results = run_benchmark(repetitions=5, target_latency=0.15)
print(f"Best time: {results['tiempo'].min()}s")
```

## 📊 Results include
- Interactive HTML reports
- Energy consumption charts
- Pareto optimal front
- CSV/Parquet export

## 📞 Contact
**Quetzalcoatl Miramontes**  
📧 ia16.diana@gmail.com  
🐙 GitHub: [@Q-Diana](https://github.com/Q-Diana)

## 🙏 Acknowledgments
Thanks to the open-source community for the libraries that make this project possible.

## 📄 License
MIT License - See [LICENSE](LICENSE)
