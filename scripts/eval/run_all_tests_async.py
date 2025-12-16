"""
run_all_tests_async.py - Uruchamia wszystkie testy RÓWNOLEGLE

Każdy test w osobnym procesie Python - maksymalne wykorzystanie GPU!
"""

import subprocess
import time
from pathlib import Path

# Tworzymy 5 osobnych skryptów testujących
tests = [
    'test_baseline.py',
    'test_trajectory.py', 
    'test_patchwise.py',
    'test_commutator.py',
    'test_combined.py'
]

print("="*70)
print("🚀 LAUNCHING 5 PARALLEL TESTS")
print("   Each test in separate Python process!")
print("="*70)

# Uruchom wszystkie naraz
processes = []

for i, test_name in enumerate(tests):
    print(f"\n[{i+1}/5] Starting: {test_name}")
    
    # Uruchom w tle
    proc = subprocess.Popen(
        ['python', f'scripts/eval/{test_name}'],
        cwd='e:\\AI iNflu\\Kenczuks',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    processes.append((test_name, proc))
    time.sleep(2)  # 2s opóźnienia między startami

print("\n" + "="*70)
print("✅ ALL 5 TESTS RUNNING IN PARALLEL!")
print("="*70)
print("\nWaiting for completion...")

# Czekaj na wszystkie
results = {}
for test_name, proc in processes:
    print(f"\n⏳ Waiting for {test_name}...")
    stdout, stderr = proc.communicate()
    
    if proc.returncode == 0:
        print(f"✅ {test_name} DONE!")
        results[test_name] = 'SUCCESS'
    else:
        print(f"❌ {test_name} FAILED!")
        print(f"Error: {stderr[:500]}")
        results[test_name] = 'FAILED'

print("\n" + "="*70)
print("📊 FINAL RESULTS")
print("="*70)

for test_name, status in results.items():
    symbol = "✅" if status == 'SUCCESS' else "❌"
    print(f"{symbol} {test_name}: {status}")

print("\n✅ All tests completed! Check results/optimization_tests/")
