import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="ITN Distribution Tracker",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f8f9fa, #e9ecef);
        border-radius: 10px;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2E86AB;
    }
    .filter-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data from GitHub
@st.cache_data
def load_data_from_github():
    """
    Load ITN data from GitHub repository
    Replace the URL below with your actual GitHub CSV file
    """
    # REPLACE THIS URL with your actual GitHub raw CSV file URL
    github_url = "SBD reconciliation.xlsx"
    
    try:
        # Load from GitHub
        df = pd.read_excel(github_url)
        
        # Define your actual column names mapping
        expected_columns = {
            'Total number of ITNs received from PHU after the additional ITNs received': 'Total ITNs Received',
            'Total ITNs distributed': 'Total ITNs Distributed',
            'Total ITNs remaining after distribution': 'Total ITNs Remaining'
        }
        
        # Check if the expected columns exist
        missing_columns = [col for col in expected_columns.keys() if col not in df.columns]
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.info("📋 Expected columns: " + ", ".join(expected_columns.keys()))
            st.stop()
        
        # Rename columns for easier handling
        df = df.rename(columns=expected_columns)
        
        # Ensure administrative columns exist
        admin_columns = ['District', 'Chiefdom', 'PHU/PPS']
        missing_admin = [col for col in admin_columns if col not in df.columns]
        if missing_admin:
            st.error(f"❌ Missing administrative columns: {missing_admin}")
            st.stop()
        
        # Calculate distribution rate
        df['Distribution Rate (%)'] = (df['Total ITNs Distributed'] / df['Total ITNs Received'] * 100).round(1)
        
        # Add last updated column if it doesn't exist
        if 'Last Updated' not in df.columns:
            df['Last Updated'] = datetime.now()
        
        # Ensure data types are correct
        numeric_columns = ['Total ITNs Received', 'Total ITNs Distributed', 'Total ITNs Remaining', 'Distribution Rate (%)']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Data validation check
        validation_issues = 0
        for idx, row in df.iterrows():
            expected_remaining = row['Total ITNs Received'] - row['Total ITNs Distributed']
            if abs(expected_remaining - row['Total ITNs Remaining']) > 0.1:  # Allow for small rounding differences
                validation_issues += 1
        
        if validation_issues > 0:
            st.warning(f"⚠️ Data validation: {validation_issues} records show inconsistent calculations (Remaining ≠ Received - Distributed)")
        
        st.success(f"✅ Successfully loaded {len(df)} records from GitHub")
        return df
        
    except FileNotFoundError:
        st.error("❌ GitHub file not found. Please check your URL.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data from GitHub: {str(e)}")
        st.error("Please check your GitHub URL and ensure the file is publicly accessible.")
        st.stop()

# Load data
df = load_data_from_github()

# Main header
st.markdown('<div class="main-header">🏥 ITN Distribution Tracker Dashboard</div>', unsafe_allow_html=True)

# Data source information
with st.expander("📁 Data Source Configuration"):
    st.markdown("""
    **Setup Instructions:**
    
    1. **Upload your CSV file to GitHub** with these exact columns:
       - `District` - Administrative district name
       - `Chiefdom` - Chiefdom within the district  
       - `PHU/PPS` - Primary Health Unit or Peripheral Health Station name
       - `Total number of ITNs received from PHU after the additional ITNs received`
       - `Total ITNs distributed`
       - `Total ITNs remaining after distribution`
    
    2. **Get the raw GitHub URL:**
       - Go to your CSV file on GitHub → Click "Raw" → Copy URL
    
    3. **Update line 32 in the code:**
       ```python
       github_url = "YOUR_GITHUB_RAW_URL_HERE"
       ```
    
    **Auto-calculated:**
    - `Distribution Rate (%)` = (Distributed/Received) × 100
    
    **Data validation:**
    - Checks: Remaining = Received - Distributed
    
    **Ready to load your real data!** 🚀
    """)

# Sidebar filters
st.sidebar.header("🔍 Filter Options")

# Create radio buttons to select which level to group by
grouping_selection = st.sidebar.radio(
    "Select the level for grouping:",
    ["District", "Chiefdom", "PHU/PPS"],
    index=0  # Default to 'District'
)

# Dictionary to define the hierarchy for each grouping level
hierarchy = {
    "District": ["District"],
    "Chiefdom": ["District", "Chiefdom"], 
    "PHU/PPS": ["District", "Chiefdom", "PHU/PPS"]
}

# Get the columns to show based on grouping selection
grouping_columns = hierarchy[grouping_selection]

# Dynamic filters based on hierarchy
filtered_df = df.copy()

# Apply filters in hierarchical order
for i, column in enumerate(grouping_columns):
    if i == 0:  # First level (District)
        available_options = sorted(filtered_df[column].unique())
        selected_options = st.sidebar.multiselect(
            f"Select {column}(s)",
            options=available_options,
            default=available_options
        )
        filtered_df = filtered_df[filtered_df[column].isin(selected_options)]
    
    elif i == 1:  # Second level (Chiefdom)
        if not filtered_df.empty:
            available_options = sorted(filtered_df[column].unique())
            selected_options = st.sidebar.multiselect(
                f"Select {column}(s)",
                options=available_options,
                default=available_options
            )
            filtered_df = filtered_df[filtered_df[column].isin(selected_options)]
    
    elif i == 2:  # Third level (PHU/PPS)
        if not filtered_df.empty:
            available_options = sorted(filtered_df[column].unique())
            selected_options = st.sidebar.multiselect(
                f"Select {column}(s)",
                options=available_options,
                default=available_options
            )
            filtered_df = filtered_df[filtered_df[column].isin(selected_options)]

# Additional filters
st.sidebar.markdown("### 📊 Performance Filters")

# Distribution rate filter
min_rate, max_rate = st.sidebar.slider(
    "Distribution Rate Range (%)",
    min_value=0,
    max_value=100,
    value=(0, 100),
    step=5
)

# ITN quantity filters
min_received, max_received = st.sidebar.slider(
    "Total ITNs Received Range",
    min_value=int(df['Total ITNs Received'].min()),
    max_value=int(df['Total ITNs Received'].max()),
    value=(int(df['Total ITNs Received'].min()), int(df['Total ITNs Received'].max())),
    step=50
)

# Apply performance filters
filtered_df = filtered_df[
    (filtered_df['Distribution Rate (%)'] >= min_rate) &
    (filtered_df['Distribution Rate (%)'] <= max_rate) &
    (filtered_df['Total ITNs Received'] >= min_received) &
    (filtered_df['Total ITNs Received'] <= max_received)
]

# Main content
if filtered_df.empty:
    st.warning("⚠️ No data available for the selected filters. Please adjust your filter criteria.")
else:
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_received = filtered_df['Total ITNs Received'].sum()
        st.metric("📦 Total ITNs Received", f"{total_received:,}")
    
    with col2:
        total_distributed = filtered_df['Total ITNs Distributed'].sum()
        st.metric("🎯 Total ITNs Distributed", f"{total_distributed:,}")
    
    with col3:
        total_remaining = filtered_df['Total ITNs Remaining'].sum()
        st.metric("📋 Total ITNs Remaining", f"{total_remaining:,}")
    
    with col4:
        avg_distribution_rate = filtered_df['Distribution Rate (%)'].mean()
        st.metric("📈 Avg Distribution Rate", f"{avg_distribution_rate:.1f}%")
    
    # Charts section
    st.markdown("## 📊 Data Visualization")
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # District-wise distribution chart
        district_summary = filtered_df.groupby('District').agg({
            'Total ITNs Received': 'sum',
            'Total ITNs Distributed': 'sum',
            'Total ITNs Remaining': 'sum'
        }).reset_index()
        
        fig1 = px.bar(
            district_summary,
            x='District',
            y=['Total ITNs Received', 'Total ITNs Distributed', 'Total ITNs Remaining'],
            title="📊 ITN Summary by District",
            labels={'value': 'Number of ITNs', 'variable': 'Status'},
            color_discrete_map={
                'Total ITNs Received': '#2E86AB',
                'Total ITNs Distributed': '#A23B72',
                'Total ITNs Remaining': '#F18F01'
            }
        )
        fig1.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Distribution rate by district
        district_rate = filtered_df.groupby('District')['Distribution Rate (%)'].mean().reset_index()
        
        fig2 = px.pie(
            district_rate,
            values='Distribution Rate (%)',
            names='District',
            title="🎯 Distribution Rate by District"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Performance analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Top performers
        st.markdown("### 🏆 Top Performing PHUs")
        top_performers = filtered_df.nlargest(10, 'Distribution Rate (%)')[['PHU/PPS', 'District', 'Total ITNs Received', 'Total ITNs Distributed', 'Distribution Rate (%)']]
        st.dataframe(top_performers, use_container_width=True)
    
    with col2:
        # Largest operations
        st.markdown("### 📦 Largest Operations")
        largest_ops = filtered_df.nlargest(10, 'Total ITNs Received')[['PHU/PPS', 'District', 'Total ITNs Received', 'Total ITNs Distributed', 'Distribution Rate (%)']]
        st.dataframe(largest_ops, use_container_width=True)
    
    # Detailed data table
    st.markdown("## 📋 Detailed Data Table")
    
    # Table display options
    col1, col2, col3 = st.columns(3)
    with col1:
        show_all_columns = st.checkbox("Show All Columns", value=False)
    with col2:
        sort_by = st.selectbox("Sort by", 
                              ["Distribution Rate (%)", "Total ITNs Received", "Total ITNs Distributed", "Total ITNs Remaining"],
                              index=0)
    with col3:
        sort_order = st.selectbox("Sort Order", ["Descending", "Ascending"])
    
    # Sort data
    ascending = sort_order == "Ascending"
    sorted_df = filtered_df.sort_values(by=sort_by, ascending=ascending)
    
    # Display table
    if show_all_columns:
        display_df = sorted_df
    else:
        display_df = sorted_df[['District', 'Chiefdom', 'PHU/PPS', 'Total ITNs Received', 'Total ITNs Distributed', 'Total ITNs Remaining', 'Distribution Rate (%)']]
    
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            'Distribution Rate (%)': st.column_config.ProgressColumn(
                "Distribution Rate (%)",
                help="Percentage of ITNs distributed",
                min_value=0,
                max_value=100,
                format="%.1f%%"
            ),
            'Total ITNs Received': st.column_config.NumberColumn(
                "Total ITNs Received",
                format="%d"
            ),
            'Total ITNs Distributed': st.column_config.NumberColumn(
                "Total ITNs Distributed", 
                format="%d"
            ),
            'Total ITNs Remaining': st.column_config.NumberColumn(
                "Total ITNs Remaining",
                format="%d"
            )
        }
    )
    
    # Export functionality
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Export Filtered Data as CSV"):
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"itn_data_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        st.metric("📊 Records Shown", len(filtered_df))
    
    # Summary statistics
    st.markdown("## 📈 Summary Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Statistical Summary")
        summary_stats = filtered_df[['Total ITNs Received', 'Total ITNs Distributed', 'Total ITNs Remaining', 'Distribution Rate (%)']].describe()
        st.dataframe(summary_stats, use_container_width=True)
    
    with col2:
        st.markdown("### 🗺️ Coverage Summary")
        coverage_stats = {
            'Districts Covered': len(filtered_df['District'].unique()),
            'Chiefdoms Covered': len(filtered_df['Chiefdom'].unique()),
            'PHUs/PPS Covered': len(filtered_df['PHU/PPS'].unique()),
            'Total Records': len(filtered_df)
        }
        
        for key, value in coverage_stats.items():
            st.metric(key, value)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    📱 ITN Distribution Tracker Dashboard | Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
</div>
""", unsafe_allow_html=True)
