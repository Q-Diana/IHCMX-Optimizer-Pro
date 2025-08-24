# 1. Install dependencies
pip install -r requirements.txt

# 2. Standard benchmark
python ihcmx_with_lowdram_pro.py --benchmark --repetitions 10 --parquet --plot --out benchmark_results

# 3. Complete grid search
python ihcmx_with_lowdram_pro.py --grid --repetitions 5 \
  --grid_latencies 0.1,0.2,0.3 \
  --grid_powers None,100,150 \
  --plot --out grid_results --report

# 4. Generate independent HTML report
python ihcmx_with_lowdram_pro.py --report --report_in results/grid_results.parquet --report_out report.html

# 5. Show dashboard
python ihcmx_with_lowdram_pro.py --benchmark --dashboard
