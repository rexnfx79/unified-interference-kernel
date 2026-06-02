#!/usr/bin/env python3
"""
QA Test Runner

Runs all QA tests and generates a summary report.
"""

import sys
import os
import unittest
import io
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_all_tests():
    """Run all QA tests and return results."""
    
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test modules
    test_dir = os.path.dirname(__file__)
    
    # Import test modules
    from test_alternative_kernels import (
        TestKernelMathematics,
        TestReproducibility,
        TestClockworkSolution,
        TestNumericalStability,
        TestKernelRegistry,
    )
    from test_reproducibility import (
        TestOptimizationReproducibility,
        TestSolutionStability,
        TestCrossValidation,
    )
    import test_qed_information as tqed
    import test_experimental_fisher as tef
    import test_fisher_transfer as tft
    import test_open_system_decoherence as tos
    import test_neutrino_observables as tnu
    import test_lepton_observables as tlep
    
    # Add all test classes
    test_classes = [
        TestKernelMathematics,
        TestReproducibility,
        TestClockworkSolution,
        TestNumericalStability,
        TestKernelRegistry,
        TestOptimizationReproducibility,
        TestSolutionStability,
        TestCrossValidation,
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    # QED information module (function-style tests)
    for name in (
        "test_density_matrix_normalized",
        "test_coherence_diagonal_dominant",
        "test_qfi_nonnegative_perturbation",
        "test_compute_qed_keys",
        "test_include_qed_flag",
    ):
        suite.addTest(unittest.FunctionTestCase(getattr(tqed, name)))

    for mod in (tef, tft, tos, tnu, tlep):
        for name in dir(mod):
            if name.startswith("test_") and callable(getattr(mod, name)):
                suite.addTest(unittest.FunctionTestCase(getattr(mod, name)))
    
    # Run tests with verbose output
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)
    
    return result, stream.getvalue()


def generate_report(result, output):
    """Generate a QA report."""
    
    report = []
    report.append("=" * 70)
    report.append("QA TEST REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)
    
    # Summary
    report.append("\nSUMMARY")
    report.append("-" * 40)
    report.append(f"Tests run:    {result.testsRun}")
    report.append(f"Failures:     {len(result.failures)}")
    report.append(f"Errors:       {len(result.errors)}")
    report.append(f"Skipped:      {len(result.skipped)}")
    
    success = result.testsRun - len(result.failures) - len(result.errors)
    success_rate = 100 * success / result.testsRun if result.testsRun > 0 else 0
    report.append(f"Success rate: {success_rate:.1f}%")
    
    # Status
    if result.wasSuccessful():
        report.append("\nSTATUS: ALL TESTS PASSED ✓")
    else:
        report.append("\nSTATUS: SOME TESTS FAILED ✗")
    
    # Failures
    if result.failures:
        report.append("\nFAILURES")
        report.append("-" * 40)
        for test, traceback in result.failures:
            report.append(f"\n{test}:")
            report.append(traceback)
    
    # Errors
    if result.errors:
        report.append("\nERRORS")
        report.append("-" * 40)
        for test, traceback in result.errors:
            report.append(f"\n{test}:")
            report.append(traceback)
    
    # Full output
    report.append("\n" + "=" * 70)
    report.append("DETAILED OUTPUT")
    report.append("=" * 70)
    report.append(output)
    
    return "\n".join(report)


def main():
    print("Running QA tests...")
    print()
    
    result, output = run_all_tests()
    report = generate_report(result, output)
    
    # Print report
    print(report)
    
    # Save report
    report_dir = os.path.join(os.path.dirname(__file__), '..', 'diagnostics', 'results')
    os.makedirs(report_dir, exist_ok=True)
    
    report_path = os.path.join(report_dir, 'qa_test_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {report_path}")
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
