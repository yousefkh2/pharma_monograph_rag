#!/usr/bin/env python3
"""
Create a comparison report across multiple LLM evaluations.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON data from file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_model_name(file_path: str) -> str:
    """Extract model name from file path."""
    name = Path(file_path).stem
    if 'mistral' in name.lower():
        return 'Mistral 7B Instruct'
    elif 'claude' in name.lower():
        return 'Claude 3.5 Sonnet'
    elif 'deepseek' in name.lower():
        return 'DeepSeek Chat'
    else:
        return name.replace('_', ' ').title()


def create_comparison_report(eval_files: List[str], judge_files: List[str], output_path: str):
    """Create a comprehensive comparison report across multiple models."""
    
    models_data = []
    
    for eval_file, judge_file in zip(eval_files, judge_files):
        if not Path(eval_file).exists() or not Path(judge_file).exists():
            print(f"⚠️ Skipping {eval_file} - files not found")
            continue
            
        eval_data = load_json(eval_file)
        judge_data = load_json(judge_file)
        
        model_name = extract_model_name(eval_file)
        
        models_data.append({
            'name': model_name,
            'eval_data': eval_data,
            'judge_data': judge_data,
            'eval_file': eval_file,
            'judge_file': judge_file
        })
    
    if not models_data:
        print("❌ No valid model data found")
        return
    
    # Create HTML comparison report
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Pharmacy Copilot Comparison</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            background-color: #f8f9fa;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        
        .comparison-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .model-card {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .model-header {{
            padding: 20px;
            font-weight: bold;
            font-size: 1.2em;
            text-align: center;
        }}
        
        .mistral {{ background: linear-gradient(135deg, #ff6b6b, #ee5a52); color: white; }}
        .claude {{ background: linear-gradient(135deg, #4ecdc4, #44a08d); color: white; }}
        .deepseek {{ background: linear-gradient(135deg, #45b7d1, #2980b9); color: white; }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            padding: 20px;
        }}
        
        .metric {{
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .metric-label {{
            color: #6c757d;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        
        .detailed-comparison {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            overflow: hidden;
        }}
        
        .comparison-header {{
            background: #2c3e50;
            color: white;
            padding: 20px;
            font-weight: bold;
            font-size: 1.2em;
        }}
        
        .question-comparison {{
            border-bottom: 1px solid #dee2e6;
        }}
        
        .question-title {{
            background: #e9ecef;
            padding: 15px;
            font-weight: bold;
        }}
        
        .answers-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1px;
            background: #dee2e6;
        }}
        
        .answer-cell {{
            background: white;
            padding: 15px;
        }}
        
        .score-badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 15px;
            font-weight: bold;
            font-size: 0.9em;
            margin-bottom: 10px;
        }}
        
        .score-high {{ background: #d4edda; color: #155724; }}
        .score-med {{ background: #fff3cd; color: #856404; }}
        .score-low {{ background: #f8d7da; color: #721c24; }}
        
        .answer-text {{
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            max-height: 200px;
            overflow-y: auto;
        }}
        
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        .summary-table th,
        .summary-table td {{
            padding: 12px;
            text-align: center;
            border: 1px solid #dee2e6;
        }}
        
        .summary-table th {{
            background: #2c3e50;
            color: white;
        }}
        
        .summary-table tr:nth-child(even) {{
            background: #f8f9fa;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 LLM Pharmacy Copilot Comparison</h1>
        <p>Comprehensive Analysis Across Multiple Models</p>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="comparison-grid">
"""

    # Add model summary cards
    for i, model in enumerate(models_data):
        judge_summary = model['judge_data']['summary']
        retrieval_metrics = model['eval_data']['aggregated_metrics']['retrieval']
        
        model_class = ['mistral', 'claude', 'deepseek'][i % 3]
        
        pass_count = sum(1 for result in model['judge_data']['results'] if result['verdict'] == 'pass')
        
        html_content += f"""
        <div class="model-card">
            <div class="model-header {model_class}">
                {model['name']}
            </div>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value">{judge_summary['mean_score']:.1f}</div>
                    <div class="metric-label">Mean Score</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{judge_summary['median_score']:.1f}</div>
                    <div class="metric-label">Median Score</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{pass_count}</div>
                    <div class="metric-label">Passed</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{retrieval_metrics['recall_at_k']['5']:.1%}</div>
                    <div class="metric-label">Recall@5</div>
                </div>
            </div>
        </div>
"""

    # Add summary table
    html_content += f"""
    </div>
    
    <div class="detailed-comparison">
        <div class="comparison-header">
            📊 Summary Comparison
        </div>
        <table class="summary-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Mean Score</th>
                    <th>Median Score</th>
                    <th>Pass Count</th>
                    <th>Manual Review</th>
                    <th>Recall@5</th>
                    <th>Precision@5</th>
                </tr>
            </thead>
            <tbody>
"""

    for model in models_data:
        judge_summary = model['judge_data']['summary']
        retrieval_metrics = model['eval_data']['aggregated_metrics']['retrieval']
        
        pass_count = sum(1 for result in model['judge_data']['results'] if result['verdict'] == 'pass')
        manual_count = sum(1 for result in model['judge_data']['results'] if result['verdict'] == 'manual_review')
        
        html_content += f"""
                <tr>
                    <td><strong>{model['name']}</strong></td>
                    <td>{judge_summary['mean_score']:.1f}</td>
                    <td>{judge_summary['median_score']:.1f}</td>
                    <td>{pass_count}</td>
                    <td>{manual_count}</td>
                    <td>{retrieval_metrics['recall_at_k']['5']:.1%}</td>
                    <td>{retrieval_metrics['precision_at_k']['5']:.1%}</td>
                </tr>
"""

    html_content += """
            </tbody>
        </table>
    </div>
    
    <div class="detailed-comparison">
        <div class="comparison-header">
            🔍 Question-by-Question Comparison
        </div>
"""

    # Create question-by-question comparison
    judge_results_by_model = {}
    eval_results_by_model = {}
    
    for model in models_data:
        judge_results_by_model[model['name']] = {
            result['question_id']: result for result in model['judge_data']['results']
        }
        eval_results_by_model[model['name']] = {
            case['question_id']: case for case in model['eval_data']['failure_cases']
        }
    
    # Get all question IDs
    all_question_ids = set()
    for model in models_data:
        all_question_ids.update(judge_results_by_model[model['name']].keys())
    
    for question_id in sorted(all_question_ids):
        # Get question text from first available model
        question_text = ""
        for model in models_data:
            if question_id in judge_results_by_model[model['name']]:
                question_text = judge_results_by_model[model['name']][question_id]['question']
                break
        
        html_content += f"""
        <div class="question-comparison">
            <div class="question-title">
                {question_id.upper()}: {question_text}
            </div>
            <div class="answers-grid">
"""
        
        for model in models_data:
            if question_id in judge_results_by_model[model['name']]:
                judge_result = judge_results_by_model[model['name']][question_id]
                eval_result = eval_results_by_model[model['name']].get(question_id, {})
                
                score = judge_result['score']
                score_class = 'score-high' if score >= 80 else 'score-med' if score >= 60 else 'score-low'
                
                # Clean the generated answer
                generated_answer = eval_result.get('generated_answer', 'N/A')
                if generated_answer.startswith('<s>'):
                    generated_answer = generated_answer[3:].strip()
                if generated_answer.endswith('</s>'):
                    generated_answer = generated_answer[:-4].strip()
                
                # Try to parse and pretty print JSON
                try:
                    parsed = json.loads(generated_answer)
                    clean_answer = json.dumps(parsed, indent=2)
                except:
                    clean_answer = generated_answer[:300] + ('...' if len(generated_answer) > 300 else '')
                
                html_content += f"""
                <div class="answer-cell">
                    <div class="score-badge {score_class}">
                        {model['name']}: {score}/100
                    </div>
                    <div class="answer-text">{clean_answer}</div>
                </div>
"""
            else:
                html_content += f"""
                <div class="answer-cell">
                    <div class="score-badge score-low">
                        {model['name']}: N/A
                    </div>
                    <div class="answer-text">No data available</div>
                </div>
"""
        
        html_content += """
            </div>
        </div>
"""

    html_content += """
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html_content)


def main():
    parser = argparse.ArgumentParser(description='Create LLM comparison report')
    parser.add_argument('--eval-files', nargs='+', required=True, help='Evaluation JSON files')
    parser.add_argument('--judge-files', nargs='+', required=True, help='Judge JSON files')
    parser.add_argument('--output', '-o', default='eval/reports/model_comparison.html', help='Output HTML file')
    
    args = parser.parse_args()
    
    if len(args.eval_files) != len(args.judge_files):
        print("❌ Number of eval files must match number of judge files")
        return 1
    
    print(f"🔄 Creating comparison report for {len(args.eval_files)} models...")
    create_comparison_report(args.eval_files, args.judge_files, args.output)
    print(f"✅ Comparison report created: {args.output}")


if __name__ == "__main__":
    main()