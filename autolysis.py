import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import openai
from typing import Dict, Any, List, Tuple

class DataAnalyzer:
    def __init__(self, filename: str):
        """Initialize the data analyzer with the given CSV file."""
        self.filename = filename
        self.df = pd.read_csv(filename)
        self.original_columns = self.df.columns.tolist()
        self.openai_client = self._setup_openai_client()
        
        # Preprocessing
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()
    
    def _setup_openai_client(self):
        """Setup OpenAI client using environment variable."""
        openai.api_base = "https://api.aiproxy.io/v1"
        openai.api_key = os.environ.get("AIPROXY_TOKEN")
        if not openai.api_key:
            raise ValueError("AIPROXY_TOKEN environment variable not set.")
        return openai.OpenAI()
    
    def _generate_llm_prompt(self, analysis_context: Dict[str, Any]) -> str:
        """Generate a comprehensive prompt for LLM analysis."""
        prompt = f"""Analyze the CSV dataset: {self.filename}

Dataset Overview:
- Total Rows: {len(self.df)}
- Columns: {', '.join(self.original_columns)}
- Numeric Columns: {', '.join(self.numeric_columns)}
- Categorical Columns: {', '.join(self.categorical_columns)}

Analysis Context:
{json.dumps(analysis_context, indent=2)}

Please provide:
1. A concise narrative interpretation of the data and analysis
2. Key insights
3. Potential actionable recommendations
4. Any suggested further investigations"""
        return prompt
    
    def missing_values_analysis(self) -> Dict[str, Any]:
        """Analyze missing values in the dataset."""
        missing_data = self.df.isnull().sum()
        missing_percentages = (missing_data / len(self.df)) * 100
        missing_summary = {
            'total_missing': missing_data.sum(),
            'columns_with_missing': dict(missing_percentages[missing_percentages > 0])
        }
        return missing_summary
    
    def outlier_detection(self) -> Dict[str, Any]:
        """Detect outliers using IQR method for numeric columns."""
        outliers = {}
        for col in self.numeric_columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            column_outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            if len(column_outliers) > 0:
                outliers[col] = {
                    'total_outliers': len(column_outliers),
                    'outlier_percentage': (len(column_outliers) / len(self.df)) * 100,
                    'min_outlier': column_outliers[col].min(),
                    'max_outlier': column_outliers[col].max()
                }
        return outliers
    
    def correlation_analysis(self) -> Dict[str, Any]:
        """Perform correlation analysis for numeric columns."""
        if len(self.numeric_columns) < 2:
            return {}
        
        correlation_matrix = self.df[self.numeric_columns].corr()
        high_correlations = []
        
        for i in range(len(self.numeric_columns)):
            for j in range(i+1, len(self.numeric_columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > 0.5:  # Threshold for significant correlation
                    high_correlations.append({
                        'column1': self.numeric_columns[i],
                        'column2': self.numeric_columns[j],
                        'correlation': corr
                    })
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'high_correlations': high_correlations
        }
    
    def dimensionality_reduction(self) -> Dict[str, Any]:
        """Perform PCA for dimensionality reduction."""
        if len(self.numeric_columns) < 2:
            return {}
        
        # Prepare data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df[self.numeric_columns])
        
        # Perform PCA
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        return {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance_explained': np.cumsum(pca.explained_variance_ratio_).tolist()
        }
    
    def clustering_analysis(self) -> Dict[str, Any]:
        """Perform clustering analysis using K-means."""
        if len(self.numeric_columns) < 2:
            return {}
        
        # Prepare data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df[self.numeric_columns])
        
        # Determine optimal number of clusters using elbow method
        inertias = []
        max_clusters = min(5, len(scaled_data) - 1)
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)
        
        return {
            'inertias': inertias,
            'optimal_clusters': np.argmin(np.diff(inertias)) + 2
        }
    
    def feature_importance(self) -> Dict[str, Any]:
        """Estimate feature importance."""
        if len(self.numeric_columns) < 2:
            return {}
        
        importance_results = {}
        
        # If there's a classification target
        categorical_targets = [col for col in self.categorical_columns if len(self.df[col].unique()) < 10]
        if categorical_targets:
            for target in categorical_targets:
                importance = mutual_info_classif(
                    self.df[self.numeric_columns], 
                    self.df[target]
                )
                importance_results[f'classification_{target}'] = dict(zip(
                    self.numeric_columns, 
                    importance
                ))
        
        # If there's a regression target
        numeric_targets = [col for col in self.numeric_columns if col not in self.original_columns]
        if numeric_targets:
            for target in numeric_targets:
                other_features = [col for col in self.numeric_columns if col != target]
                importance = mutual_info_regression(
                    self.df[other_features], 
                    self.df[target]
                )
                importance_results[f'regression_{target}'] = dict(zip(
                    other_features, 
                    importance
                ))
        
        return importance_results
    
    def generate_visualizations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate visualizations based on analysis results."""
        plt.close('all')
        visualization_files = []
        
        # Correlation Heatmap
        if 'correlation_matrix' in analysis_results:
            plt.figure(figsize=(10, 8))
            correlation_matrix = pd.DataFrame(analysis_results['correlation_matrix'])
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Heatmap')
            plt.tight_layout()
            corr_file = 'correlation_heatmap.png'
            plt.savefig(corr_file)
            visualization_files.append(corr_file)
            plt.close()
        
        # PCA Variance Explanation
        if 'explained_variance_ratio' in analysis_results:
            plt.figure(figsize=(10, 6))
            variance_ratios = analysis_results['explained_variance_ratio']
            plt.bar(range(1, len(variance_ratios) + 1), variance_ratios)
            plt.title('PCA - Explained Variance Ratio')
            plt.xlabel('Principal Components')
            plt.ylabel('Explained Variance Ratio')
            plt.tight_layout()
            pca_file = 'pca_variance.png'
            plt.savefig(pca_file)
            visualization_files.append(pca_file)
            plt.close()
        
        # Clustering Elbow Method
        if 'inertias' in analysis_results:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(analysis_results['inertias']) + 1), analysis_results['inertias'], marker='o')
            plt.title('Elbow Method for Optimal Clusters')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Inertia')
            plt.tight_layout()
            cluster_file = 'clustering_elbow.png'
            plt.savefig(cluster_file)
            visualization_files.append(cluster_file)
            plt.close()
        
        return visualization_files
    
    def generate_narrative(self, analysis_results: Dict[str, Any], visualization_files: List[str]) -> str:
        """Generate a narrative using GPT-4o-Mini with the analysis results."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user", 
                    "content": self._generate_llm_prompt({
                        **analysis_results,
                        'visualization_files': visualization_files
                    })
                }]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating narrative: {str(e)}"
    
    def perform_analysis(self) -> None:
        """Perform comprehensive data analysis and generate outputs."""
        # Perform various analyses
        analysis_results = {
            'missing_values': self.missing_values_analysis(),
            'outliers': self.outlier_detection(),
            'correlations': self.correlation_analysis(),
            'dimensionality_reduction': self.dimensionality_reduction(),
            'clustering': self.clustering_analysis(),
            'feature_importance': self.feature_importance()
        }
        
        # Generate visualizations
        visualization_files = self.generate_visualizations(analysis_results)
        
        # Generate narrative
        narrative = self.generate_narrative(analysis_results, visualization_files)
        
        # Write README.md
        with open('README.md', 'w') as f:
            f.write(f"# Data Analysis Report: {self.filename}\n\n")
            f.write(narrative + "\n\n")
            
            # Add image references
            for img_file in visualization_files:
                f.write(f"![{img_file}]({img_file})\n\n")
        
        print(f"Analysis complete. Check README.md and generated visualizations.")

def main():
    # Check if CSV file is provided
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <filename.csv>")
        sys.exit(1)
    
    # Perform analysis
    analyzer = DataAnalyzer(sys.argv[1])
    analyzer.perform_analysis()

if __name__ == "__main__":
    main()
