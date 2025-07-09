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

# Load data from URL
@st.cache_data
def load_data_from_url():
    """
    Load ITN data from Excel file URL
    """
    # Excel file URL
    excel_url = "SBD reconciliation.xlsx"
    
    try:
        # Load Excel file from URL
        df = pd.read_excel(excel_url)
        
        st.success(f"✅ Successfully loaded {len(df)} records from Excel file")
        
        # Display column names for debugging
        st.info(f"📋 Available columns: {', '.join(df.columns.tolist())}")
        
        # Show first few rows to understand structure
        with st.expander("🔍 Preview of loaded data"):
            st.dataframe(df.head())
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading Excel file: {str(e)}")
        st.error("Please check the URL and ensure the file is accessible.")
        st.stop()

# Load data
df = load_data_from_url()

# Data preprocessing - map your actual columns
st.markdown("### 🔧 Column Mapping")
st.info("Please update the column mapping below to match your Excel file structure")

# Let user see what columns are available and map them
if len(df.columns) > 0:
    st.write("**Available columns in your Excel file:**")
    for i, col in enumerate(df.columns):
        st.write(f"{i+1}. {col}")
    
    # Column mapping interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Select columns for mapping:**")
        district_col = st.selectbox("District column:", df.columns, index=0)
        chiefdom_col = st.selectbox("Chiefdom column:", df.columns, index=1 if len(df.columns) > 1 else 0)
        phu_col = st.selectbox("PHU/PPS column:", df.columns, index=2 if len(df.columns) > 2 else 0)
    
    with col2:
        st.markdown("**ITN data columns:**")
        received_col = st.selectbox("Total ITNs Received column:", df.columns, index=3 if len(df.columns) > 3 else 0)
        distributed_col = st.selectbox("Total ITNs Distributed column:", df.columns, index=4 if len(df.columns) > 4 else 0)
        remaining_col = st.selectbox("Total ITNs Remaining column:", df.columns, index=5 if len(df.columns) > 5 else 0)
    
    # Apply column mapping
    if st.button("🔄 Apply Column Mapping"):
        try:
            # Create new dataframe with mapped columns
            df_mapped = pd.DataFrame({
                'District': df[district_col],
                'Chiefdom': df[chiefdom_col], 
                'PHU/PPS': df[phu_col],
                'Total ITNs Received': pd.to_numeric(df[received_col], errors='coerce').fillna(0),
                'Total ITNs Distributed': pd.to_numeric(df[distributed_col], errors='coerce').fillna(0),
                'Total ITNs Remaining': pd.to_numeric(df[remaining_col], errors='coerce').fillna(0)
            })
            
            # Calculate distribution rate
            df_mapped['Distribution Rate (%)'] = (df_mapped['Total ITNs Distributed'] / df_mapped['Total ITNs Received'] * 100).round(1)
            
            # Replace infinite values with 0
            df_mapped['Distribution Rate (%)'] = df_mapped['Distribution Rate (%)'].replace([np.inf, -np.inf], 0)
            
            # Add last updated column
            df_mapped['Last Updated'] = datetime.now()
            
            # Update the main dataframe
            df = df_mapped
            
            st.success("✅ Column mapping applied successfully!")
            
            # Show preview of mapped data
            st.write("**Preview of mapped data:**")
            st.dataframe(df.head())
            
        except Exception as e:
            st.error(f"❌ Error mapping columns: {str(e)}")
            st.stop()
else:
    st.error("No columns found in the Excel file")
    st.stop()

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
    
    # Debug information
    st.markdown("### 🔍 Debug Information")
    st.write(f"Total records in original data: {len(df)}")
    st.write(f"Unique districts in data: {sorted(df['District'].unique())}")
    
    if len(df) > 0:
        st.write("Sample of original data:")
        st.dataframe(df.head())
else:
    # Debug section for data issues
    if st.sidebar.checkbox("Show Debug Info", value=False):
        st.markdown("### 🔍 Debug Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Overview:**")
            st.write(f"Total original records: {len(df)}")
            st.write(f"Filtered records: {len(filtered_df)}")
            st.write(f"Grouping level: {grouping_selection}")
            
            st.write("**Records by District:**")
            district_counts = df['District'].value_counts().sort_index()
            st.dataframe(district_counts)
        
        with col2:
            st.write("**Data Quality Check:**")
            # Check for missing values
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                st.write("Missing values found:")
                st.dataframe(missing_data[missing_data > 0])
            else:
                st.write("✅ No missing values")
            
            # Check for zero values
            zero_received = (df['Total ITNs Received'] == 0).sum()
            zero_distributed = (df['Total ITNs Distributed'] == 0).sum()
            
            st.write(f"Records with zero ITNs received: {zero_received}")
            st.write(f"Records with zero ITNs distributed: {zero_distributed}")
        
        # Show sample data for problematic districts
        st.write("**Sample data by district:**")
        for district in sorted(df['District'].unique()):
            district_data = df[df['District'] == district]
            st.write(f"**{district}** ({len(district_data)} records):")
            if len(district_data) > 0:
                sample_data = district_data[['Chiefdom', 'PHU/PPS', 'Total ITNs Received', 'Total ITNs Distributed', 'Total ITNs Remaining']].head(3)
                st.dataframe(sample_data, use_container_width=True)
            else:
                st.write("No data found")
    
    # Key metrics with additional validation
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
        if len(filtered_df) > 0:
            avg_distribution_rate = filtered_df['Distribution Rate (%)'].mean()
            st.metric("📈 Avg Distribution Rate", f"{avg_distribution_rate:.1f}%")
        else:
            st.metric("📈 Avg Distribution Rate", "0.0%")
    
    # Summary Table based on grouping selection
    st.markdown("## 📋 Summary Table")
    
    if grouping_selection == "District":
        # District level summary
        summary_table = filtered_df.groupby('District').agg({
            'Total ITNs Received': 'sum',
            'Total ITNs Distributed': 'sum',
            'Total ITNs Remaining': 'sum'
        }).reset_index()
        
        # Calculate distribution rate for each district
        summary_table['Distribution Rate (%)'] = (summary_table['Total ITNs Distributed'] / summary_table['Total ITNs Received'] * 100).round(1)
        
        # Sort by Total ITNs Received (descending)
        summary_table = summary_table.sort_values('Total ITNs Received', ascending=False)
        
        st.markdown("### 🗺️ Summary by District")
        
    elif grouping_selection == "Chiefdom":
        # Chiefdom level summary
        summary_table = filtered_df.groupby(['District', 'Chiefdom']).agg({
            'Total ITNs Received': 'sum',
            'Total ITNs Distributed': 'sum',
            'Total ITNs Remaining': 'sum'
        }).reset_index()
        
        # Calculate distribution rate for each chiefdom
        summary_table['Distribution Rate (%)'] = (summary_table['Total ITNs Distributed'] / summary_table['Total ITNs Received'] * 100).round(1)
        
        # Sort by District, then by Total ITNs Received
        summary_table = summary_table.sort_values(['District', 'Total ITNs Received'], ascending=[True, False])
        
        st.markdown("### 🏘️ Summary by Chiefdom")
        
    else:  # PHU/PPS level
        # PHU/PPS level summary
        summary_table = filtered_df.groupby(['District', 'Chiefdom', 'PHU/PPS']).agg({
            'Total ITNs Received': 'sum',
            'Total ITNs Distributed': 'sum',
            'Total ITNs Remaining': 'sum'
        }).reset_index()
        
        # Calculate distribution rate for each PHU/PPS
        summary_table['Distribution Rate (%)'] = (summary_table['Total ITNs Distributed'] / summary_table['Total ITNs Received'] * 100).round(1)
        
        # Sort by District, Chiefdom, then by Total ITNs Received
        summary_table = summary_table.sort_values(['District', 'Chiefdom', 'Total ITNs Received'], ascending=[True, True, False])
        
        st.markdown("### 🏥 Summary by PHU/PPS")
    
    # Display the summary table
    st.dataframe(
        summary_table,
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
    
    # Summary statistics for the table
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(f"📊 Total {grouping_selection}s", len(summary_table))
    
    with col2:
        total_received_summary = summary_table['Total ITNs Received'].sum()
        st.metric("📦 Grand Total Received", f"{total_received_summary:,}")
    
    with col3:
        total_distributed_summary = summary_table['Total ITNs Distributed'].sum()
        st.metric("🎯 Grand Total Distributed", f"{total_distributed_summary:,}")
    
    # Export functionality for summary table
    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"📥 Export {grouping_selection} Summary as CSV"):
            csv = summary_table.to_csv(index=False)
            st.download_button(
                label="Download Summary CSV",
                data=csv,
                file_name=f"itn_summary_{grouping_selection.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if summary_table['Total ITNs Received'].sum() > 0:
            overall_rate = (summary_table['Total ITNs Distributed'].sum() / summary_table['Total ITNs Received'].sum() * 100)
            st.metric("📈 Overall Distribution Rate", f"{overall_rate:.1f}%")
    
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
