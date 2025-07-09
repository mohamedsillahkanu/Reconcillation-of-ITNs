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
    Modify the URL below to point to your actual GitHub CSV file
    """
    # Replace this URL with your actual GitHub raw CSV file URL
    github_url = "https://raw.githubusercontent.com/yourusername/yourrepo/main/itn_data.csv"
    
    try:
        # Try to load from GitHub
        df = pd.read_csv(github_url)
        
        # Define your actual column names
        expected_columns = {
            'Total number of ITNs received from PHU after the additional ITNs received': 'Total ITNs Received',
            'Total ITNs distributed': 'Total ITNs Distributed',
            'Total ITNs remaining after distribution': 'Total ITNs Remaining'
        }
        
        # Check if the expected columns exist
        missing_columns = [col for col in expected_columns.keys() if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.info("Expected columns: " + ", ".join(expected_columns.keys()))
            return create_sample_data()
        
        # Rename columns for easier handling
        df = df.rename(columns=expected_columns)
        
        # Ensure administrative columns exist
        admin_columns = ['District', 'Chiefdom', 'PHU/PPS']
        missing_admin = [col for col in admin_columns if col not in df.columns]
        if missing_admin:
            st.error(f"Missing administrative columns: {missing_admin}")
            return create_sample_data()
        
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
        calculated_remaining = df['Total ITNs Received'] - df['Total ITNs Distributed']
        if not calculated_remaining.equals(df['Total ITNs Remaining']):
            st.warning("⚠️ Data validation: Some records show Remaining ≠ (Received - Distributed)")
        
        st.success(f"✅ Successfully loaded {len(df)} records from GitHub")
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading data from GitHub: {str(e)}")
        st.info("Using sample data instead. Please check your GitHub URL and file format.")
        return create_sample_data()

def create_sample_data():
    """Create sample data matching your simplified structure"""
    np.random.seed(42)
    
    districts = ["Western Area Urban", "Western Area Rural", "Bo", "Kenema", "Kailahun", "Kono", "Bombali", "Port Loko", "Moyamba", "Bonthe"]
    data = []
    
    for district in districts:
        for i in range(np.random.randint(3, 6)):
            chiefdom = f"{district} Chiefdom {i+1}"
            for j in range(np.random.randint(2, 4)):
                phu = f"PHU {district[:3]}-{i+1}-{j+1}"
                
                # Generate realistic data
                total_received = np.random.randint(800, 2500)
                distributed = np.random.randint(int(total_received * 0.6), int(total_received * 0.95))
                remaining = total_received - distributed
                
                data.append({
                    'District': district,
                    'Chiefdom': chiefdom,
                    'PHU/PPS': phu,
                    'Total ITNs Received': total_received,
                    'Total ITNs Distributed': distributed,
                    'Total ITNs Remaining': remaining,
                    'Distribution Rate (%)': round((distributed / total_received) * 100, 1),
                    'Last Updated': datetime.now() - timedelta(days=np.random.randint(1, 7))
                })
    
    return pd.DataFrame(data)

# Load data
df = load_data_from_github()

# Main header
st.markdown('<div class="main-header">🏥 ITN Distribution Tracker Dashboard</div>', unsafe_allow_html=True)

# Data source information
with st.expander("📁 Data Source Configuration"):
    st.markdown("""
    **To use your own data from GitHub:**
    
    1. **Upload your CSV file to GitHub** with these exact columns:
       - `District` - Administrative district name
       - `Chiefdom` - Chiefdom within the district  
       - `PHU/PPS` - Primary Health Unit or Peripheral Health Station name
       - `Total number of ITNs received from PHU after the additional ITNs received`
       - `Total ITNs distributed`
       - `Total ITNs remaining after distribution`
    
    2. **Get the raw GitHub URL:**
       - Go to your CSV file on GitHub
       - Click "Raw" button
       - Copy the URL (should start with `https://raw.githubusercontent.com/`)
    
    3. **Update the code:**
       - Replace `github_url` variable in the `load_data_from_github()` function
       - Example: `https://raw.githubusercontent.com/yourusername/yourrepo/main/itn_data.csv`
    
    **Calculated Metric:**
    - `Distribution Rate (%)` - (Distributed/Received)*100
    
    **Data Validation:**
    - ✅ Remaining = Received - Distributed
    
    **Current Status:** ✅ Ready to use your data (update GitHub URL)
    """)

# Sidebar filters
st.sidebar.markdown('<div class="filter-section">', unsafe_allow_html=True)
st.sidebar.header("🔍 Filters")

# District filter
selected_districts = st.sidebar.multiselect(
    "Select Districts",
    options=sorted(df['District'].unique()),
    default=sorted(df['District'].unique())
)

# Chiefdom filter (based on selected districts)
if selected_districts:
    available_chiefdoms = df[df['District'].isin(selected_districts)]['Chiefdom'].unique()
    selected_chiefdoms = st.sidebar.multiselect(
        "Select Chiefdoms",
        options=sorted(available_chiefdoms),
        default=sorted(available_chiefdoms)
    )
else:
    selected_chiefdoms = []

# PHU filter (based on selected chiefdoms)
if selected_chiefdoms:
    available_phus = df[df['Chiefdom'].isin(selected_chiefdoms)]['PHU/PPS'].unique()
    selected_phus = st.sidebar.multiselect(
        "Select PHUs/PPS",
        options=sorted(available_phus),
        default=sorted(available_phus)
    )
else:
    selected_phus = []

# Distribution rate filter
min_rate, max_rate = st.sidebar.slider(
    "Distribution Rate Range (%)",
    min_value=0,
    max_value=100,
    value=(0, 100),
    step=5
)

# ITN quantity filters
st.sidebar.markdown("### 📊 Quantity Filters")

min_received, max_received = st.sidebar.slider(
    "Total ITNs Received Range",
    min_value=int(df['Total ITNs Received'].min()),
    max_value=int(df['Total ITNs Received'].max()),
    value=(int(df['Total ITNs Received'].min()), int(df['Total ITNs Received'].max())),
    step=50
)

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Filter data
filtered_df = df[
    (df['District'].isin(selected_districts)) &
    (df['Chiefdom'].isin(selected_chiefdoms)) &
    (df['PHU/PPS'].isin(selected_phus)) &
    (df['Distribution Rate (%)'] >= min_rate) &
    (df['Distribution Rate (%)'] <= max_rate) &
    (df['Total ITNs Received'] >= min_received) &
    (df['Total ITNs Received'] <= max_received)
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
