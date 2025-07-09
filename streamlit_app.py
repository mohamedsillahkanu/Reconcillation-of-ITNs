import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

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
</style>
""", unsafe_allow_html=True)

# Load data from Excel file
@st.cache_data
def load_data_from_excel():
    """
    Load ITN data from local Excel file
    """
    excel_url = "SBD reconciliation.xlsx"
    
    try:
        # Load the local Excel file
        df = pd.read_excel(excel_url)
        st.success(f"✅ Successfully loaded {len(df)} records from: {excel_url}")
        return df
        
    except FileNotFoundError:
        st.error(f"❌ File not found: {excel_url}")
        st.error("Please ensure 'SBD reconciliation.xlsx' is in the same folder as your script.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading Excel file: {str(e)}")
        st.stop()

# Load data
df = load_data_from_excel()

# Main header
st.markdown('<div class="main-header">🏥 ITN Distribution Summary Dashboard</div>', unsafe_allow_html=True)

# Show available columns for mapping
st.markdown("## 🔧 Column Mapping")
st.info("Map your Excel columns to the required fields:")

# Display available columns
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Available columns in your Excel file:**")
    for i, col in enumerate(df.columns, 1):
        st.write(f"{i}. {col}")

with col2:
    st.markdown("**Required mappings:**")
    
    # Column mapping interface
    district_col = st.selectbox("District column:", df.columns, key="district")
    chiefdom_col = st.selectbox("Chiefdom column:", df.columns, key="chiefdom")
    phu_col = st.selectbox("PHU/PPS column:", df.columns, key="phu")
    received_col = st.selectbox("Total ITNs Received column:", df.columns, key="received")
    distributed_col = st.selectbox("Total ITNs Distributed column:", df.columns, key="distributed")
    remaining_col = st.selectbox("Total ITNs Remaining column:", df.columns, key="remaining")

# Process data with mapping
try:
    # Create mapped dataframe
    mapped_df = pd.DataFrame({
        'District': df[district_col].astype(str),
        'Chiefdom': df[chiefdom_col].astype(str),
        'PHU/PPS': df[phu_col].astype(str),
        'Total ITNs Received': pd.to_numeric(df[received_col], errors='coerce').fillna(0),
        'Total ITNs Distributed': pd.to_numeric(df[distributed_col], errors='coerce').fillna(0),
        'Total ITNs Remaining': pd.to_numeric(df[remaining_col], errors='coerce').fillna(0)
    })
    
    # Clean up data - remove rows where all key columns are empty/null
    mapped_df = mapped_df[
        (mapped_df['District'].str.strip() != '') & 
        (mapped_df['District'] != 'nan') &
        (mapped_df['Chiefdom'].str.strip() != '') & 
        (mapped_df['Chiefdom'] != 'nan') &
        (mapped_df['PHU/PPS'].str.strip() != '') & 
        (mapped_df['PHU/PPS'] != 'nan')
    ]
    
    st.success(f"✅ Data mapped successfully! {len(mapped_df)} valid records processed.")
    
except Exception as e:
    st.error(f"❌ Error mapping columns: {str(e)}")
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Create radio buttons for summary level
summary_level = st.sidebar.radio(
    "Select Summary Level:",
    ["District Summary", "Chiefdom Summary", "PHU/PPS Summary"],
    index=0
)

# Filter by District
districts = sorted(mapped_df['District'].unique())
selected_districts = st.sidebar.multiselect(
    "Filter by District:",
    options=districts,
    default=districts
)

# Filter by Chiefdom (based on selected districts)
if selected_districts:
    available_chiefdoms = sorted(mapped_df[mapped_df['District'].isin(selected_districts)]['Chiefdom'].unique())
    selected_chiefdoms = st.sidebar.multiselect(
        "Filter by Chiefdom:",
        options=available_chiefdoms,
        default=available_chiefdoms
    )
else:
    selected_chiefdoms = []

# Filter by PHU/PPS (based on selected chiefdoms)
if selected_chiefdoms:
    available_phus = sorted(mapped_df[
        (mapped_df['District'].isin(selected_districts)) & 
        (mapped_df['Chiefdom'].isin(selected_chiefdoms))
    ]['PHU/PPS'].unique())
    selected_phus = st.sidebar.multiselect(
        "Filter by PHU/PPS:",
        options=available_phus,
        default=available_phus
    )
else:
    selected_phus = []

# Apply filters
if selected_districts and selected_chiefdoms and selected_phus:
    filtered_df = mapped_df[
        (mapped_df['District'].isin(selected_districts)) &
        (mapped_df['Chiefdom'].isin(selected_chiefdoms)) &
        (mapped_df['PHU/PPS'].isin(selected_phus))
    ]
else:
    filtered_df = mapped_df[mapped_df['District'].isin(selected_districts)] if selected_districts else mapped_df

# Main content
if not filtered_df.empty:
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_received = filtered_df['Total ITNs Received'].sum()
        st.metric("📦 Total ITNs Received", f"{total_received:,.0f}")
    
    with col2:
        total_distributed = filtered_df['Total ITNs Distributed'].sum()
        st.metric("🎯 Total ITNs Distributed", f"{total_distributed:,.0f}")
    
    with col3:
        total_remaining = filtered_df['Total ITNs Remaining'].sum()
        st.metric("📋 Total ITNs Remaining", f"{total_remaining:,.0f}")
    
    with col4:
        if total_received > 0:
            distribution_rate = (total_distributed / total_received * 100)
            st.metric("📈 Distribution Rate", f"{distribution_rate:.1f}%")
        else:
            st.metric("📈 Distribution Rate", "0.0%")
    
    # Generate summary based on selected level
    st.markdown(f"## 📊 {summary_level}")
    
    if summary_level == "District Summary":
        # District level summary
        summary_df = filtered_df.groupby('District').agg({
            'Total ITNs Received': 'sum',
            'Total ITNs Distributed': 'sum',
            'Total ITNs Remaining': 'sum'
        }).reset_index()
        
        # Calculate distribution rate
        summary_df['Distribution Rate (%)'] = (
            summary_df['Total ITNs Distributed'] / summary_df['Total ITNs Received'] * 100
        ).round(1)
        summary_df['Distribution Rate (%)'] = summary_df['Distribution Rate (%)'].fillna(0)
        
        # Sort by Total ITNs Received
        summary_df = summary_df.sort_values('Total ITNs Received', ascending=False)
        
    elif summary_level == "Chiefdom Summary":
        # Chiefdom level summary with District reference
        summary_df = filtered_df.groupby(['District', 'Chiefdom']).agg({
            'Total ITNs Received': 'sum',
            'Total ITNs Distributed': 'sum',
            'Total ITNs Remaining': 'sum'
        }).reset_index()
        
        # Calculate distribution rate
        summary_df['Distribution Rate (%)'] = (
            summary_df['Total ITNs Distributed'] / summary_df['Total ITNs Received'] * 100
        ).round(1)
        summary_df['Distribution Rate (%)'] = summary_df['Distribution Rate (%)'].fillna(0)
        
        # Sort by District, then by Total ITNs Received
        summary_df = summary_df.sort_values(['District', 'Total ITNs Received'], ascending=[True, False])
        
    else:  # PHU/PPS Summary
        # PHU/PPS level summary with District and Chiefdom reference
        summary_df = filtered_df.groupby(['District', 'Chiefdom', 'PHU/PPS']).agg({
            'Total ITNs Received': 'sum',
            'Total ITNs Distributed': 'sum',
            'Total ITNs Remaining': 'sum'
        }).reset_index()
        
        # Calculate distribution rate
        summary_df['Distribution Rate (%)'] = (
            summary_df['Total ITNs Distributed'] / summary_df['Total ITNs Received'] * 100
        ).round(1)
        summary_df['Distribution Rate (%)'] = summary_df['Distribution Rate (%)'].fillna(0)
        
        # Sort by District, Chiefdom, then by Total ITNs Received
        summary_df = summary_df.sort_values(['District', 'Chiefdom', 'Total ITNs Received'], ascending=[True, True, False])
    
    # Display summary table
    st.dataframe(
        summary_df,
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
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(f"📊 Total {summary_level.split()[0]}s", len(summary_df))
    
    with col2:
        avg_distribution_rate = summary_df['Distribution Rate (%)'].mean()
        st.metric("📈 Average Distribution Rate", f"{avg_distribution_rate:.1f}%")
    
    with col3:
        if len(summary_df) > 0:
            max_distribution_rate = summary_df['Distribution Rate (%)'].max()
            st.metric("🏆 Best Distribution Rate", f"{max_distribution_rate:.1f}%")
    
    # Export functionality
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(f"📥 Export {summary_level} as CSV"):
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{summary_level.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📥 Export Filtered Raw Data as CSV"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Raw Data CSV",
                data=csv,
                file_name=f"filtered_raw_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Show sample of filtered raw data
    with st.expander("🔍 View Filtered Raw Data Sample"):
        st.dataframe(filtered_df.head(20), use_container_width=True)

else:
    st.warning("⚠️ No data available for the selected filters. Please adjust your selections.")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    📱 ITN Distribution Dashboard | Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
</div>
""", unsafe_allow_html=True)
