#!/usr/bin/env python3
"""
Interactive Threshold Analysis Dashboard

Creates an interactive Plotly dashboard for exploring classification thresholds
and their impact on precision, recall, and F1 scores for churn prediction.

Usage:
    python scripts/interactive_threshold.py
    python scripts/interactive_threshold.py --port 8050
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import BEST_MODEL_FILE, TEST_DATA_FILE, TEST_LABELS_FILE, TEST_ENGINEERED_FILE, IDENTIFIER_COLUMN
from utils import load_parquet, load_pickle


def create_threshold_dashboard(model, X_test, y_test):
    """Create interactive threshold analysis dashboard."""
    
    # Get predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate precision/recall curve
    precision_pts, recall_pts, thresholds = precision_recall_curve(y_test, y_proba)
    
    # Add final point (threshold = 1.0)
    thresholds = np.append(thresholds, 1.0)
    precision_pts = np.append(precision_pts, 1.0)
    recall_pts = np.append(recall_pts, 0.0)
    
    # Calculate F1 scores
    f1_pts = 2 * precision_pts * recall_pts / (precision_pts + recall_pts + 1e-9)
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Precision-Recall Curve',
            'Metrics vs Threshold',
            'Confusion Matrix',
            'Business Impact Calculator'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "heatmap"}, {"type": "scatter"}]
        ]
    )
    
    # 1. Precision-Recall Curve
    fig.add_trace(
        go.Scatter(
            x=recall_pts,
            y=precision_pts,
            mode='lines',
            name='Precision-Recall Curve',
            line=dict(color='blue', width=3),
            hovertemplate='Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add current threshold point (default 0.5)
    default_idx = np.argmin(np.abs(thresholds - 0.5))
    fig.add_trace(
        go.Scatter(
            x=[recall_pts[default_idx]],
            y=[precision_pts[default_idx]],
            mode='markers',
            name='Default (0.50)',
            marker=dict(size=10, color='red'),
            hovertemplate='Default Threshold<br>Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Metrics vs Threshold
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=precision_pts,
            mode='lines',
            name='Precision',
            line=dict(color='blue'),
            hovertemplate='Threshold: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=recall_pts,
            mode='lines',
            name='Recall',
            line=dict(color='green'),
            hovertemplate='Threshold: %{x:.3f}<br>Recall: %{y:.3f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=f1_pts,
            mode='lines',
            name='F1 Score',
            line=dict(color='red'),
            hovertemplate='Threshold: %{x:.3f}<br>F1: %{y:.3f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Confusion Matrix (for default threshold)
    y_pred_default = (y_proba >= 0.5).astype(int)
    cm = np.array([[np.sum((y_test == 0) & (y_pred_default == 0)), 
                    np.sum((y_test == 0) & (y_pred_default == 1))],
                   [np.sum((y_test == 1) & (y_pred_default == 0)), 
                    np.sum((y_test == 1) & (y_pred_default == 1))]])
    
    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=['Predicted No Churn', 'Predicted Churn'],
            y=['Actual No Churn', 'Actual Churn'],
            colorscale='Blues',
            text=cm,
            texttemplate="%d",
            textfont={"size": 14},
            hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. Business Impact Calculator
    # Assume some business metrics
    total_customers = len(y_test)
    avg_monthly_revenue = 70  # Average monthly revenue per customer
    
    # Calculate business metrics for different thresholds
    threshold_range = np.arange(0.1, 0.91, 0.05)
    revenue_retained = []
    retention_cost = []
    
    for threshold in threshold_range:
        y_pred = (y_proba >= threshold).astype(int)
        
        # True positives = churners we caught
        tp = np.sum((y_test == 1) & (y_pred == 1))
        # False positives = non-churners we targeted
        fp = np.sum((y_test == 0) & (y_pred == 1))
        
        # Business calculations
        retained_revenue = tp * avg_monthly_revenue * 12  # Annual revenue saved
        targeting_cost = (tp + fp) * 50  # $50 per retention campaign
        
        revenue_retained.append(retained_revenue)
        retention_cost.append(targeting_cost)
    
    net_impact = np.array(revenue_retained) - np.array(retention_cost)
    
    fig.add_trace(
        go.Scatter(
            x=threshold_range,
            y=revenue_retained,
            mode='lines+markers',
            name='Revenue Retained',
            line=dict(color='green'),
            hovertemplate='Threshold: %{x:.2f}<br>Revenue: $%{y:,.0f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=threshold_range,
            y=retention_cost,
            mode='lines+markers',
            name='Retention Cost',
            line=dict(color='red'),
            hovertemplate='Threshold: %{x:.2f}<br>Cost: $%{y:,.0f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=threshold_range,
            y=net_impact,
            mode='lines+markers',
            name='Net Impact',
            line=dict(color='blue', width=3),
            hovertemplate='Threshold: %{x:.2f}<br>Net: $%{y:,.0f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': '<b>Interactive Churn Prediction Threshold Analysis</b>',
            'x': 0.5,
            'font': {'size': 20}
        },
        height=800,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update subplot titles and axis labels
    fig.update_xaxes(title_text="Recall", row=1, col=1)
    fig.update_yaxes(title_text="Precision", row=1, col=1)
    
    fig.update_xaxes(title_text="Threshold", row=1, col=2)
    fig.update_yaxes(title_text="Score", row=1, col=2)
    
    fig.update_xaxes(title_text="", row=2, col=1)
    fig.update_yaxes(title_text="", row=2, col=1)
    
    fig.update_xaxes(title_text="Threshold", row=2, col=2)
    fig.update_yaxes(title_text="Amount ($)", row=2, col=2)
    
    return fig


def create_customer_segmentation_plot(X_test, y_test, y_proba):
    """Create 3D customer segmentation plot."""
    
    # Select key features for visualization
    feature_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Check if engineered features are available
    if 'clv_proxy' in X_test.columns:
        feature_cols = ['tenure', 'MonthlyCharges', 'clv_proxy']
    elif 'tenure_x_monthly' in X_test.columns:
        feature_cols = ['tenure', 'MonthlyCharges', 'tenure_x_monthly']
    
    # Get the data
    plot_data = X_test[feature_cols].copy()
    plot_data['churn'] = y_test
    plot_data['churn_probability'] = y_proba
    plot_data['risk_level'] = pd.cut(
        y_proba, 
        bins=[0, 0.3, 0.7, 1.0], 
        labels=['Low', 'Medium', 'High']
    )
    
    # Create 3D scatter plot
    fig = px.scatter_3d(
        plot_data,
        x=feature_cols[0],
        y=feature_cols[1],
        z=feature_cols[2],
        color='risk_level',
        symbol='churn',
        size='churn_probability',
        hover_data={
            'churn_probability': ':.3f',
            'churn': True,
            'risk_level': True
        },
        title=f'3D Customer Segmentation<br><sub>{feature_cols[0]} vs {feature_cols[1]} vs {feature_cols[2]}</sub>',
        color_discrete_map={'Low': 'green', 'Medium': 'yellow', 'High': 'red'}
    )
    
    fig.update_traces(
        marker=dict(line=dict(width=1, color='DarkSlateGrey')),
        selector=dict(mode='markers')
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title=feature_cols[0],
            yaxis_title=feature_cols[1],
            zaxis_title=feature_cols[2]
        ),
        height=600
    )
    
    return fig


def main():
    """Run interactive threshold analysis."""
    parser = argparse.ArgumentParser(description="Interactive threshold analysis dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Port for dashboard")
    parser.add_argument("--save", type=str, help="Save dashboard as HTML file")
    args = parser.parse_args()
    
    print("=== Loading Model and Data ===")
    
    # Load model
    model = load_pickle(BEST_MODEL_FILE)
    print(f"Loaded model: {BEST_MODEL_FILE}")
    
    # Load test data (prefer engineered features)
    if TEST_ENGINEERED_FILE.exists():
        X_test = load_parquet(TEST_ENGINEERED_FILE).drop(columns=[IDENTIFIER_COLUMN])
        print(f"Using engineered features: {TEST_ENGINEERED_FILE}")
    else:
        X_test = load_parquet(TEST_DATA_FILE).drop(columns=[IDENTIFIER_COLUMN])
        print(f"Using basic features: {TEST_DATA_FILE}")
    
    y_test = load_parquet(TEST_LABELS_FILE).iloc[:, 0]
    
    # Normalize labels to binary
    if y_test.dtype == object:
        y_test = (y_test == "Yes").astype(int)
    else:
        y_test = y_test.astype(int)
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Churn rate: {y_test.mean():.3f}")
    
    # Create dashboards
    print("\n=== Creating Interactive Dashboards ===")
    
    # Threshold analysis dashboard
    threshold_fig = create_threshold_dashboard(model, X_test, y_test)
    
    # Customer segmentation plot
    y_proba = model.predict_proba(X_test)[:, 1]
    segmentation_fig = create_customer_segmentation_plot(X_test, y_test, y_proba)
    
    # Save or show
    if args.save:
        output_file = Path(args.save)
        threshold_fig.write_html(str(output_file))
        print(f"Dashboard saved to: {output_file}")
    else:
        # Try to open in browser
        try:
            import webbrowser
            import tempfile
            import os
            
            # Create temporary HTML file
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False)
            
            # Combine both figures into one HTML
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Churn Prediction Dashboard</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .dashboard {{ margin-bottom: 50px; }}
                </style>
            </head>
            <body>
                <h1>Churn Prediction Interactive Dashboard</h1>
                
                <div class="dashboard">
                    {threshold_fig.to_html(full_html=False, include_plotlyjs=False)}
                </div>
                
                <div class="dashboard">
                    {segmentation_fig.to_html(full_html=False, include_plotlyjs=False)}
                </div>
                
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </body>
            </html>
            """
            
            temp_file.write(html_content)
            temp_file.close()
            
            # Open in browser
            webbrowser.open(f'file://{temp_file.name}')
            print(f"Dashboard opened in browser: {temp_file.name}")
            print("Close the browser tab when done viewing")
            
        except ImportError:
            print("Plotly dashboard created but webbrowser not available")
            print("Use --save option to save as HTML file")
    
    print("\n=== Dashboard Summary ===")
    print("✅ Threshold Analysis: Interactive precision/recall/F1 curves")
    print("✅ Business Impact: Revenue vs cost calculations")
    print("✅ Customer Segmentation: 3D risk visualization")
    print("✅ Confusion Matrix: Real-time error analysis")


if __name__ == "__main__":
    main()
