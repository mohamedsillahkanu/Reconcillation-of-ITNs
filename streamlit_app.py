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

# Load data from URL
@st.cache_data
def load_data_from_url():
    """
    Load ITN data from Excel file URL
    """
    # Excel file URL - UPDATE THIS WITH YOUR ACTUAL URL
    excel_url = "SBD reconciliation.xlsx"
    
    try:
        # Try different ways to load the Excel file
        try:
            # If it's a local file
            df = pd.read_excel(excel_url)
        except:
            # If it's a URL
            df = pd.read_excel(f"https://raw.githubusercontent.com/yourusername/yourrepo/main/{excel_url}")
        
        st.success(f"✅ Successfully loaded {len(df)} records from Excel file")
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading Excel file: {str(e)}")
        st.error("Please ensure the file is accessible or update the file path/URL.")
        st.stop()

# Load data
df = load_data_from_url()

# Main header
st.markdown('<div class="main-header">🏥 ITN Distribution Summary Dashboard</div>', unsafe_allow_html=True)

# Data overview
st.markdown("## 📊 Data Overview")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📋 Total Records", len(df))

with col2:
    st.metric("📑 Total Columns", len(df.columns))

with col3:
    st.metric("📅 Last Updated", datetime.now().strftime("%Y-%m-%d %H:%M"))

# Create sidebar filters
st.sidebar.header("🔍 Filter Options")

# Get all text/categorical columns for grouping
text_columns = []
numeric_columns = []

for col in df.columns:
    if df[col].dtype in ['object', 'string']:
        text_columns.append(col)
    elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
        numeric_columns.append(col)

st.sidebar.markdown("### 📊 Available Text Columns for Grouping:")
for col in text_columns:
    st.sidebar.write(f"• {col}")

st.sidebar.markdown("### 🔢 Available Numeric Columns:")
for col in numeric_columns:
    st.sidebar.write(f"• {col}")

# Select grouping column
if text_columns:
    grouping_column = st.sidebar.selectbox(
        "Select column to group by:",
        options=text_columns,
        index=0
    )
    
    # Select columns to sum
    if numeric_columns:
        columns_to_sum = st.sidebar.multiselect(
            "Select numeric columns to sum:",
            options=numeric_columns,
            default=numeric_columns
        )
    else:
        st.warning("No numeric columns found for aggregation")
        columns_to_sum = []
    
    # Filter by unique values in the grouping column
    if grouping_column:
        unique_values = sorted(df[grouping_column].dropna().unique())
        selected_values = st.sidebar.multiselect(
            f"Filter by {grouping_column}:",
            options=unique_values,
            default=unique_values
        )
        
        # Filter dataframe
        if selected_values:
            filtered_df = df[df[grouping_column].isin(selected_values)]
        else:
            filtered_df = df.copy()
    else:
        filtered_df = df.copy()
        selected_values = []

else:
    st.error("No text columns found for grouping")
    st.stop()

# Main content
if not filtered_df.empty and columns_to_sum:
    
    # Summary table
    st.markdown(f"## 📋 Summary by {grouping_column}")
    
    # Create summary
    summary_dict = {'Unique_Values': []}
    
    # Add the grouping column values
    for value in selected_values:
        summary_dict['Unique_Values'].append(value)
    
    # Calculate totals for each numeric column
    for col in columns_to_sum:
        summary_dict[f'Total_{col}'] = []
        for value in selected_values:
            value_data = filtered_df[filtered_df[grouping_column] == value]
            total = value_data[col].sum() if not value_data.empty else 0
            summary_dict[f'Total_{col}'].append(total)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_dict)
    summary_df = summary_df.rename(columns={'Unique_Values': grouping_column})
    
    # Sort by first numeric column (descending)
    if len(columns_to_sum) > 0:
        first_col = f'Total_{columns_to_sum[0]}'
        summary_df = summary_df.sort_values(first_col, ascending=False)
    
    # Display summary table
    st.dataframe(
        summary_df,
        use_container_width=True,
        column_config={
            col: st.column_config.NumberColumn(
                col,
                format="%d" if summary_df[col].dtype in ['int64', 'int32'] else "%.2f"
            ) for col in summary_df.columns if col.startswith('Total_')
        }
    )
    
    # Grand totals
    st.markdown("### 🎯 Grand Totals")
    grand_totals_cols = st.columns(len(columns_to_sum))
    
    for i, col in enumerate(columns_to_sum):
        with grand_totals_cols[i]:
            grand_total = summary_df[f'Total_{col}'].sum()
            st.metric(f"Grand Total {col}", f"{grand_total:,.0f}")
    
    # Export functionality
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export Summary as CSV"):
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="Download Summary CSV",
                data=csv,
                file_name=f"summary_{grouping_column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        st.metric("📊 Unique Values Count", len(summary_df))
    
    # Detailed data table
    st.markdown("## 📋 Detailed Filtered Data")
    
    # Show options
    col1, col2 = st.columns(2)
    with col1:
        show_all_columns = st.checkbox("Show All Columns", value=False)
    with col2:
        max_rows = st.selectbox("Max rows to display", [50, 100, 500, 1000], index=1)
    
    # Display filtered data
    if show_all_columns:
        display_df = filtered_df.head(max_rows)
    else:
        # Show only grouping column and numeric columns
        display_columns = [grouping_column] + columns_to_sum
        available_columns = [col for col in display_columns if col in filtered_df.columns]
        display_df = filtered_df[available_columns].head(max_rows)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Export filtered data
    if st.button("📥 Export Filtered Data as CSV"):
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data CSV",
            data=csv,
            file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

else:
    if filtered_df.empty:
        st.warning("⚠️ No data available for the selected filters.")
    if not columns_to_sum:
        st.warning("⚠️ No numeric columns selected for aggregation.")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    📱 ITN Distribution Summary Dashboard | Data Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
</div>
""", unsafe_allow_html=True)
