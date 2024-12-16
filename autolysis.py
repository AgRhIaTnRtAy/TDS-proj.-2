import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from typing import List, Dict, Any

def setup_openai_client():
    """Setup OpenAI client using environment variable."""
    openai.api_base = "https://api.aiproxy.io/v1"
    openai.api_key = os.environ.get("eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjEwMDEwNzlAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.Y0MNKtt-YeXPgbcSmTc7aNIR3-ZZ8rH34QybJtk1Dgw")
    if not openai.api_key:
        raise ValueError("AIPROXY_TOKEN environment variable not set.")
    return openai.OpenAI()

def generate_narrative(df: pd.DataFrame, filename: str) -> str:
    """Generate a narrative analysis using GPT-4o-Mini."""
    client = setup_openai_client()
    
    # Summarize dataset characteristics
    summary_stats = {
        'rows': len(df),
        'columns': list(df.columns),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    
    prompt = f"""Write a compelling data story based on this dataset ({filename}):
    
Dataset Summary:
- Total Rows: {summary_stats['rows']}
- Columns: {', '.join(summary_stats['columns'])}
- Numeric Columns: {', '.join(summary_stats['numeric_columns'])}
- Categorical Columns: {', '.join(summary_stats['categorical_columns'])}

Key Insights to Cover:
1. Major trends or patterns
2. Most surprising or interesting findings
3. Potential implications or stories behind the data
4. Use storytelling techniques, not just dry statistics

Write in Markdown format, approximately 400-500 words."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating narrative: {str(e)}"

def create_visualizations(df: pd.DataFrame) -> List[str]:
    """Create 1-3 data visualizations."""
    plt.close('all')
    visualization_files = []
    
    # Numeric columns for plotting
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Scatter Plot
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(10, 6))
        x_col, y_col = numeric_cols[:2]
        plt.scatter(df[x_col], df[y_col], alpha=0.5)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'{x_col} vs {y_col} Scatter Plot')
        plt.tight_layout()
        scatter_file = 'scatter_plot.png'
        plt.savefig(scatter_file)
        visualization_files.append(scatter_file)
        plt.close()
    
    # Correlation Heatmap
    if len(numeric_cols) > 2:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        heatmap_file = 'correlation_heatmap.png'
        plt.savefig(heatmap_file)
        visualization_files.append(heatmap_file)
        plt.close()
    
    # Categorical Distribution
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        plt.figure(figsize=(12, 6))
        category_col = categorical_cols[0]
        df[category_col].value_counts().plot(kind='bar')
        plt.title(f'Distribution of {category_col}')
        plt.xlabel(category_col)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        category_file = 'category_distribution.png'
        plt.savefig(category_file)
        visualization_files.append(category_file)
        plt.close()
    
    return visualization_files

def main():
    # Check if CSV file is provided
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <filename.csv>")
        sys.exit(1)
    
    # Read CSV
    filename = sys.argv[1]
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    # Generate narrative
    narrative = generate_narrative(df, filename)
    
    # Create visualizations
    visualization_files = create_visualizations(df)
    
    # Write README.md
    with open('README.md', 'w') as f:
        f.write(f"# Data Analysis Report: {filename}\n\n")
        f.write(narrative + "\n\n")
        
        # Add image references
        for img_file in visualization_files:
            f.write(f"![{img_file}]({img_file})\n\n")
    
    print(f"Analysis complete. Check README.md and generated visualizations.")

if __name__ == "__main__":
    main()
