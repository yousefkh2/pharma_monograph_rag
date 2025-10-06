#!/usr/bin/env python3
"""
Create a comprehensive evaluation report that combines LLM answers with judge scores
for easy review and potential second-opinion analysis.
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


def extract_answer_from_generated(generated_answer: str) -> str:
    """Extract and clean the JSON answer from the generated response."""
    # Remove <s> tags and clean up
    cleaned = generated_answer.replace('<s>', '').replace('</s>', '').strip()
    
    # Try to parse as JSON to pretty print it
    try:
        parsed = json.loads(cleaned)
        return json.dumps(parsed, indent=2)
    except json.JSONDecodeError:
        return cleaned


def create_html_report(eval_data: Dict, judge_data: Dict, output_path: str):
    """Create a beautiful HTML report combining evaluation and judge data."""
    
    # Create mapping from question_id to judge results
    judge_results = {result['question_id']: result for result in judge_data['results']}
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pharmacy Copilot Evaluation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
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
        
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .question-card {{
            background: white;
            margin-bottom: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .question-header {{
            background: #2c3e50;
            color: white;
            padding: 15px 20px;
            font-weight: bold;
            font-size: 1.1em;
        }}
        
        .question-content {{
            padding: 20px;
        }}
        
        .question-text {{
            background: #e8f4f8;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin-bottom: 20px;
            font-style: italic;
        }}
        
        .answer-section {{
            margin-bottom: 25px;
        }}
        
        .answer-title {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        
        .answer-content {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #dee2e6;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            overflow-x: auto;
        }}
        
        .expected-answer {{
            background: #d4edda;
            border-color: #c3e6cb;
        }}
        
        .judge-section {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
        }}
        
        .judge-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .score {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .verdict {{
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            text-transform: uppercase;
            font-size: 0.9em;
        }}
        
        .pass {{
            background: #d4edda;
            color: #155724;
        }}
        
        .manual_review {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .fail {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .rationale {{
            font-style: italic;
            color: #495057;
            line-height: 1.7;
        }}
        
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .metric-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #dee2e6;
        }}
        
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #495057;
        }}
        
        .metric-label {{
            color: #6c757d;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        
        .retrieval-info {{
            margin-top: 15px;
            font-size: 0.9em;
            color: #6c757d;
        }}
        
        .chunks-found {{
            background: #d4edda;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }}
        
        .chunks-missed {{
            background: #f8d7da;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 Pharmacy Copilot Evaluation Report</h1>
        <p>Comprehensive Analysis: LLM Answers + Judge Scores</p>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <h2>📊 Executive Summary</h2>
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{judge_data['summary']['count']}</div>
                <div class="metric-label">Total Questions</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{judge_data['summary']['mean_score']:.1f}</div>
                <div class="metric-label">Mean Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{judge_data['summary']['median_score']:.1f}</div>
                <div class="metric-label">Median Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{eval_data['aggregated_metrics']['retrieval']['recall_at_k']['5']:.1%}</div>
                <div class="metric-label">Recall@5</div>
            </div>
        </div>
    </div>
"""

    # Process each question
    for failure_case in eval_data['failure_cases']:
        question_id = failure_case['question_id']
        judge_result = judge_results.get(question_id, {})
        
        # Get verdict class for styling
        verdict = judge_result.get('verdict', 'unknown')
        verdict_class = verdict.replace('_', '')
        
        # Clean the generated answer
        clean_answer = extract_answer_from_generated(failure_case['generated_answer'])
        
        html_content += f"""
    <div class="question-card">
        <div class="question-header">
            Question {question_id.upper()}: Score {judge_result.get('score', '?')}/100
        </div>
        <div class="question-content">
            <div class="question-text">
                <strong>Question:</strong> {failure_case['question']}
            </div>
            
            <div class="answer-section">
                <div class="answer-title">🤖 LLM Generated Answer</div>
                <div class="answer-content">{clean_answer}</div>
            </div>
            
            <div class="answer-section">
                <div class="answer-title">✅ Expected Answer</div>
                <div class="answer-content expected-answer">{failure_case['expected_answer']}</div>
            </div>
            
            <div class="judge-section">
                <div class="judge-header">
                    <div>
                        <span class="score">{judge_result.get('score', '?')}</span>
                        <span style="font-size: 0.7em; color: #6c757d;">/100</span>
                    </div>
                    <div class="verdict {verdict_class}">{verdict}</div>
                </div>
                <div class="rationale">
                    <strong>Judge's Rationale:</strong><br>
                    {judge_result.get('rationale', 'No rationale available')}
                </div>
            </div>
            
            <div class="retrieval-info">
                <strong>📑 Retrieval Performance:</strong>
                <div class="chunks-found">
                    <strong>Relevant Chunks Found:</strong> {', '.join(failure_case.get('relevant_chunks_found', []))}
                </div>
                {f'<div class="chunks-missed"><strong>Essential Chunks Missed:</strong> {", ".join(failure_case.get("essential_chunks_missed", []))}</div>' if failure_case.get('essential_chunks_missed') else ''}
            </div>
        </div>
    </div>
"""

    html_content += """
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html_content)


def create_markdown_report(eval_data: Dict, judge_data: Dict, output_path: str):
    """Create a markdown report for easy reading and potential LLM analysis."""
    
    judge_results = {result['question_id']: result for result in judge_data['results']}
    
    markdown_content = f"""# 🏥 Pharmacy Copilot Evaluation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Executive Summary

- **Total Questions:** {judge_data['summary']['count']}
- **Mean Score:** {judge_data['summary']['mean_score']:.1f}/100
- **Median Score:** {judge_data['summary']['median_score']:.1f}/100
- **Recall@5:** {eval_data['aggregated_metrics']['retrieval']['recall_at_k']['5']:.1%}

---

"""

    for failure_case in eval_data['failure_cases']:
        question_id = failure_case['question_id']
        judge_result = judge_results.get(question_id, {})
        
        clean_answer = extract_answer_from_generated(failure_case['generated_answer'])
        
        markdown_content += f"""## Question {question_id.upper()} - Score: {judge_result.get('score', '?')}/100

### ❓ Question
> {failure_case['question']}

### 🤖 LLM Generated Answer
```json
{clean_answer}
```

### ✅ Expected Answer
```
{failure_case['expected_answer']}
```

### ⚖️ Judge's Assessment
- **Score:** {judge_result.get('score', '?')}/100
- **Verdict:** `{judge_result.get('verdict', 'unknown')}`
- **Rationale:** {judge_result.get('rationale', 'No rationale available')}

### 📑 Retrieval Performance
- **Relevant Chunks Found:** {', '.join(failure_case.get('relevant_chunks_found', []))}
{f"- **Essential Chunks Missed:** {', '.join(failure_case.get('essential_chunks_missed', []))}" if failure_case.get('essential_chunks_missed') else ""}

---

"""

    with open(output_path, 'w') as f:
        f.write(markdown_content)


def create_json_report(eval_data: Dict, judge_data: Dict, output_path: str):
    """Create a comprehensive JSON report combining all data for LLM analysis."""
    
    judge_results = {result['question_id']: result for result in judge_data['results']}
    
    comprehensive_report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "dataset": eval_data["dataset"],
            "evaluation_time": eval_data["metadata"]["evaluation_time"],
            "judge_summary": judge_data["summary"]
        },
        "aggregated_metrics": eval_data["aggregated_metrics"],
        "detailed_results": []
    }
    
    for failure_case in eval_data['failure_cases']:
        question_id = failure_case['question_id']
        judge_result = judge_results.get(question_id, {})
        
        try:
            # Try to parse the generated answer as JSON
            clean_answer_str = failure_case['generated_answer'].replace('<s>', '').replace('</s>', '').strip()
            parsed_answer = json.loads(clean_answer_str)
        except json.JSONDecodeError:
            parsed_answer = clean_answer_str
        
        comprehensive_report["detailed_results"].append({
            "question_id": question_id,
            "question": failure_case['question'],
            "llm_response": {
                "raw_answer": failure_case['generated_answer'],
                "parsed_answer": parsed_answer,
                "failure_types": failure_case.get('failure_types', [])
            },
            "expected_answer": failure_case['expected_answer'],
            "judge_assessment": {
                "score": judge_result.get('score'),
                "verdict": judge_result.get('verdict'),
                "rationale": judge_result.get('rationale')
            },
            "retrieval_performance": {
                "retrieved_chunks": failure_case.get('retrieved_chunks', []),
                "relevant_chunks_found": failure_case.get('relevant_chunks_found', []),
                "essential_chunks_missed": failure_case.get('essential_chunks_missed', [])
            },
            "key_points_missing": failure_case.get('key_points_missing', [])
        })
    
    with open(output_path, 'w') as f:
        json.dump(comprehensive_report, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Create comprehensive evaluation report')
    parser.add_argument('eval_file', help='Path to evaluation results JSON file')
    parser.add_argument('judge_file', help='Path to judge results JSON file')
    parser.add_argument('--output-dir', '-o', default='eval/reports/comprehensive', 
                       help='Output directory for reports')
    parser.add_argument('--name', '-n', default='evaluation_report',
                       help='Base name for output files')
    
    args = parser.parse_args()
    
    # Load data
    eval_data = load_json(args.eval_file)
    judge_data = load_json(args.judge_file)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate reports
    print("🔄 Generating comprehensive evaluation reports...")
    
    # HTML report for beautiful viewing
    html_path = output_dir / f"{args.name}.html"
    create_html_report(eval_data, judge_data, str(html_path))
    print(f"✅ HTML report created: {html_path}")
    
    # Markdown report for easy reading
    md_path = output_dir / f"{args.name}.md"
    create_markdown_report(eval_data, judge_data, str(md_path))
    print(f"✅ Markdown report created: {md_path}")
    
    # JSON report for LLM analysis
    json_path = output_dir / f"{args.name}.json"
    create_json_report(eval_data, judge_data, str(json_path))
    print(f"✅ JSON report created: {json_path}")
    
    print(f"\n🎉 All reports generated in: {output_dir}")
    print(f"👀 Open {html_path} in your browser for the best viewing experience!")


if __name__ == "__main__":
    main()