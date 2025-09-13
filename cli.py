#!/usr/bin/env python3
"""
TrustworthyCodeLLM Command Line Interface

Simple CLI for evaluating code trustworthiness from the command line.

Usage:
    python cli.py evaluate --code "def hello(): return 'world'"
    python cli.py evaluate --file my_code.py
    python cli.py benchmark --dataset enhanced
    python cli.py dashboard --start
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.framework import MultiModalEvaluationFramework, CodeSample
from src.evaluators.enhanced_communication import EnhancedCommunicationEvaluator
from src.evaluators.execution_based import (
    ExecutionBasedRobustnessEvaluator,
    ExecutionBasedSecurityEvaluator,
    ExecutionBasedPerformanceEvaluator
)
from src.evaluators.static_analysis import (
    StaticSecurityAnalyzer,
    StaticMaintainabilityAnalyzer,
    StaticEthicalAnalyzer
)


class TrustworthyCodeCLI:
    """Command line interface for TrustworthyCodeLLM."""
    
    def __init__(self):
        self.framework = MultiModalEvaluationFramework()
        self._setup_framework()
    
    def _setup_framework(self):
        """Setup the evaluation framework with all evaluators."""
        self.framework.register_evaluator(EnhancedCommunicationEvaluator())
        self.framework.register_evaluator(ExecutionBasedRobustnessEvaluator())
        self.framework.register_evaluator(ExecutionBasedSecurityEvaluator())
        self.framework.register_evaluator(ExecutionBasedPerformanceEvaluator())
        self.framework.register_evaluator(StaticSecurityAnalyzer())
        self.framework.register_evaluator(StaticMaintainabilityAnalyzer())
        self.framework.register_evaluator(StaticEthicalAnalyzer())
    
    async def evaluate_code(self, code: str, problem_description: str = "", output_format: str = "table") -> None:
        """Evaluate code trustworthiness."""
        
        print("🔍 Evaluating code trustworthiness...")
        
        # Create code sample
        sample = CodeSample(
            id="cli_eval",
            source_code=code,
            problem_description=problem_description,
            language="python"
        )
        
        # Run evaluation
        results = await self.framework.evaluate_code_sample(sample)
        
        # Display results
        if output_format == "json":
            self._display_json_results(results)
        else:
            self._display_table_results(results)
    
    def _display_table_results(self, results):
        """Display results in table format."""
        print("\n" + "="*60)
        print("🛡️  TRUSTWORTHINESS EVALUATION RESULTS")
        print("="*60)
        
        # Calculate overall score
        if results:
            total_score = sum(r.score * r.confidence for r in results)
            total_weight = sum(r.confidence for r in results)
            overall_score = total_score / total_weight if total_weight > 0 else 0
            
            print(f"\n🎯 Overall Score: {overall_score:.2f}/1.00 ({overall_score*100:.1f}%)")
            
            print(f"\n📊 Category Breakdown:")
            print("-" * 50)
            
            for result in results:
                category = result.category.value.title()
                score = result.score
                confidence = result.confidence
                method = result.method.value
                
                # Score bar
                bar_length = 20
                filled = int(score * bar_length)
                bar = "█" * filled + "░" * (bar_length - filled)
                
                print(f"{category:15} │ {bar} │ {score:.2f} ({method[:4]})")
            
            print("-" * 50)
            print(f"{'Confidence':15} │ {'Method abbreviations:':25}")
            print(f"{'High: >0.7':15} │ {'exec=execution, stat=static':25}")
            print(f"{'Med:  0.4-0.7':15} │ {'llm=LLM ensemble':25}")
            print(f"{'Low:  <0.4':15} │ {'':25}")
        else:
            print("❌ No evaluation results generated")
        
        print("="*60)
    
    def _display_json_results(self, results):
        """Display results in JSON format."""
        output = {
            "overall_score": 0.0,
            "categories": {},
            "details": {}
        }
        
        if results:
            total_score = sum(r.score * r.confidence for r in results)
            total_weight = sum(r.confidence for r in results)
            output["overall_score"] = total_score / total_weight if total_weight > 0 else 0
            
            for result in results:
                category = result.category.value
                output["categories"][category] = {
                    "score": result.score,
                    "confidence": result.confidence,
                    "method": result.method.value
                }
                output["details"][category] = result.details
        
        print(json.dumps(output, indent=2))
    
    async def evaluate_file(self, file_path: str, output_format: str = "table") -> None:
        """Evaluate code from file."""
        
        path = Path(file_path)
        if not path.exists():
            print(f"❌ File not found: {file_path}")
            return
        
        try:
            code = path.read_text(encoding='utf-8')
            problem_description = f"Evaluate code from {path.name}"
            await self.evaluate_code(code, problem_description, output_format)
        except Exception as e:
            print(f"❌ Error reading file: {e}")
    
    def start_dashboard(self, port: int = 8000) -> None:
        """Start the web dashboard."""
        print(f"🚀 Starting TrustworthyCodeLLM Dashboard on port {port}")
        print(f"📱 Open http://localhost:{port} in your browser")
        print("⏹️  Press Ctrl+C to stop")
        
        try:
            import uvicorn
            from web_dashboard.app import app
            
            uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
        except ImportError:
            print("❌ Dashboard dependencies not installed. Run: pip install fastapi uvicorn")
        except Exception as e:
            print(f"❌ Failed to start dashboard: {e}")
    
    async def run_benchmark(self, dataset: str = "enhanced") -> None:
        """Run benchmark evaluation."""
        print(f"📊 Running benchmark on {dataset} dataset...")
        
        try:
            from experiments.experimental_validation import ExperimentalValidation
            
            validator = ExperimentalValidation()
            
            print("🔄 Running reliability experiment...")
            reliability_results = await validator.run_reliability_experiment()
            
            print("🔄 Running coverage experiment...")
            coverage_results = await validator.run_coverage_experiment()
            
            print("\n" + "="*60)
            print("📈 BENCHMARK RESULTS")
            print("="*60)
            print(f"Average Variance: {reliability_results['average_variance']:.4f}")
            print(f"Coverage: {coverage_results['coverage_percentage']:.1f}%")
            print(f"Samples Evaluated: {coverage_results['samples_evaluated']}")
            print("="*60)
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")


async def main():
    """Main CLI entry point."""
    
    parser = argparse.ArgumentParser(
        description="TrustworthyCodeLLM - Code LLM Trustworthiness Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate code trustworthiness')
    eval_group = eval_parser.add_mutually_exclusive_group(required=True)
    eval_group.add_argument('--code', type=str, help='Code string to evaluate')
    eval_group.add_argument('--file', type=str, help='Python file to evaluate')
    eval_parser.add_argument('--problem', type=str, default='', help='Problem description')
    eval_parser.add_argument('--format', choices=['table', 'json'], default='table', help='Output format')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Start web dashboard')
    dashboard_parser.add_argument('--port', type=int, default=8000, help='Port number (default: 8000)')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmark evaluation')
    benchmark_parser.add_argument('--dataset', choices=['enhanced', 'original'], default='enhanced', help='Dataset to use')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version information')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = TrustworthyCodeCLI()
    
    try:
        if args.command == 'evaluate':
            if args.code:
                await cli.evaluate_code(args.code, args.problem, args.format)
            elif args.file:
                await cli.evaluate_file(args.file, args.format)
        
        elif args.command == 'dashboard':
            cli.start_dashboard(args.port)
        
        elif args.command == 'benchmark':
            await cli.run_benchmark(args.dataset)
        
        elif args.command == 'version':
            print("TrustworthyCodeLLM v1.0.0")
            print("Enhanced Multi-Modal Evaluation Framework for Code LLM Trustworthiness")
            print("https://github.com/your-username/trustworthy-code-llm-eval")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
