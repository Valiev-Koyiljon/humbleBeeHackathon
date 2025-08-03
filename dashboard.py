# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from datetime import datetime, timedelta
# import folium
# from streamlit_folium import folium_static

# # Page config
# st.set_page_config(
#     page_title="NYC 311 Dashboard",
#     page_icon="🗽",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Title and description
# st.title("🗽 NYC 311 Service Requests Dashboard")
# st.markdown("Interactive analysis of service requests across New York City boroughs")

# # File upload
# st.markdown("### 📁 Data Source")
# uploaded_file = st.file_uploader(
#     "Upload your NYC 311 CSV file (any filename accepted)", 
#     type=['csv'],
#     help="Upload any NYC 311 CSV file - filename doesn't matter, we'll auto-detect the columns!"
# )

# # Show file info if uploaded
# if uploaded_file is not None:
#     st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
#     file_details = {
#         "Filename": uploaded_file.name,
#         "File size": f"{uploaded_file.size / 1024:.1f} KB"
#     }
#     st.json(file_details)

# # Sample data function
# @st.cache_data
# def create_sample_data():
#     np.random.seed(42)
#     n_records = 1000
    
#     boroughs = ['MANHATTAN', 'BROOKLYN', 'QUEENS', 'BRONX', 'STATEN ISLAND']
#     agencies = ['NYPD', 'DOT', 'HPD', 'DSNY', 'DEP', 'DOHMH']
#     complaint_types = [
#         'Noise - Residential', 'Street Light Condition', 'Heat/Hot Water', 
#         'Blocked Driveway', 'Water System', 'Illegal Parking', 'Traffic Signal Condition',
#         'Rodent', 'Graffiti', 'Street Condition'
#     ]
#     statuses = ['Open', 'Closed', 'In Progress']
    
#     # Generate dates
#     start_date = datetime(2024, 1, 1)
#     end_date = datetime(2024, 12, 31)
#     date_range = (end_date - start_date).days
    
#     # NYC coordinates ranges
#     lat_range = (40.4774, 40.9176)
#     lon_range = (-74.2591, -73.7004)
    
#     data = []
#     for i in range(n_records):
#         created_date = start_date + timedelta(days=np.random.randint(0, date_range))
#         borough = np.random.choice(boroughs)
#         status = np.random.choice(statuses, p=[0.2, 0.7, 0.1])
        
#         # Closed date logic
#         if status == 'Closed':
#             closed_date = created_date + timedelta(days=np.random.randint(1, 30))
#         else:
#             closed_date = None
            
#         data.append({
#             'unique_key': f"key_{i}",
#             'created_date': created_date,
#             'closed_date': closed_date,
#             'agency': np.random.choice(agencies),
#             'complaint_type': np.random.choice(complaint_types),
#             'borough': borough,
#             'status': status,
#             'latitude': np.random.uniform(*lat_range),
#             'longitude': np.random.uniform(*lon_range),
#             'incident_zip': str(np.random.randint(10001, 11697))
#         })
    
#     return pd.DataFrame(data)

# # Load data
# if uploaded_file is not None:
#     try:
#         df = pd.read_csv(uploaded_file)
#         st.success(f"✅ Successfully loaded {len(df):,} records from {uploaded_file.name}")
        
#         # Show column info
#         st.info(f"📊 Detected columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
        
#         # Auto-detect and convert date columns (flexible column names)
#         date_columns = [col for col in df.columns if 'date' in col.lower()]
#         for col in date_columns:
#             try:
#                 df[col] = pd.to_datetime(df[col], errors='coerce')
#                 st.success(f"✅ Converted '{col}' to datetime")
#             except:
#                 st.warning(f"⚠️ Could not convert '{col}' to datetime")
        
#         # Auto-detect coordinate columns
#         lat_columns = [col for col in df.columns if any(term in col.lower() for term in ['lat', 'latitude'])]
#         lon_columns = [col for col in df.columns if any(term in col.lower() for term in ['lon', 'long', 'longitude'])]
        
#         if lat_columns and lon_columns:
#             st.success(f"🗺️ Map ready! Using {lat_columns[0]} and {lon_columns[0]} for coordinates")
            
#     except Exception as e:
#         st.error(f"❌ Error reading file: {e}")
#         df = create_sample_data()
#         st.info("🔄 Using sample data instead")
# else:
#     df = create_sample_data()
#     st.info("📁 No file uploaded. Using sample data for demonstration.")
#     st.markdown("**👆 Upload any NYC 311 CSV file above to get started!**")

# # Sidebar filters
# st.sidebar.header("🔍 Filters")

# # Date range filter (flexible column detection)
# date_columns = [col for col in df.columns if 'date' in col.lower()]
# if date_columns:
#     primary_date_col = date_columns[0]  # Use first date column found
    
#     min_date = df[primary_date_col].min().date()
#     max_date = df[primary_date_col].max().date()
    
#     date_range = st.sidebar.date_input(
#         f"Select Date Range ({primary_date_col})",
#         value=(min_date, max_date),
#         min_value=min_date,
#         max_value=max_date
#     )
    
#     if len(date_range) == 2:
#         start_date, end_date = date_range
#         df_filtered = df[
#             (df[primary_date_col].dt.date >= start_date) & 
#             (df[primary_date_col].dt.date <= end_date)
#         ]
#     else:
#         df_filtered = df
# else:
#     df_filtered = df
#     st.sidebar.info("No date columns detected")

# # Borough filter (flexible column detection)
# borough_columns = [col for col in df_filtered.columns if 'borough' in col.lower()]
# if borough_columns:
#     borough_col = borough_columns[0]
#     boroughs = st.sidebar.multiselect(
#         f"Select Boroughs ({borough_col})",
#         options=df_filtered[borough_col].unique(),
#         default=df_filtered[borough_col].unique()
#     )
#     df_filtered = df_filtered[df_filtered[borough_col].isin(boroughs)]

# # Complaint type filter (flexible column detection)
# complaint_columns = [col for col in df_filtered.columns if any(term in col.lower() for term in ['complaint', 'type'])]
# if complaint_columns:
#     complaint_col = complaint_columns[0]
#     complaint_types = st.sidebar.multiselect(
#         f"Select Complaint Types ({complaint_col})",
#         options=df_filtered[complaint_col].unique(),
#         default=list(df_filtered[complaint_col].unique())[:5]  # Show top 5 by default
#     )
#     df_filtered = df_filtered[df_filtered[complaint_col].isin(complaint_types)]

# # Status filter (flexible column detection)
# status_columns = [col for col in df_filtered.columns if 'status' in col.lower()]
# if status_columns:
#     status_col = status_columns[0]
#     statuses = st.sidebar.multiselect(
#         f"Select Status ({status_col})",
#         options=df_filtered[status_col].unique(),
#         default=df_filtered[status_col].unique()
#     )
#     df_filtered = df_filtered[df_filtered[status_col].isin(statuses)]

# # Clear filters button
# if st.sidebar.button("🔄 Clear All Filters"):
#     st.experimental_rerun()

# # Key metrics
# col1, col2, col3, col4 = st.columns(4)

# with col1:
#     st.metric("Total Requests", f"{len(df_filtered):,}")

# with col2:
#     closed_requests = len(df_filtered[df_filtered['status'] == 'Closed']) if 'status' in df_filtered.columns else 0
#     st.metric("Closed Requests", f"{closed_requests:,}")

# with col3:
#     open_requests = len(df_filtered[df_filtered['status'] == 'Open']) if 'status' in df_filtered.columns else 0
#     st.metric("Open Requests", f"{open_requests:,}")

# with col4:
#     completion_rate = (closed_requests / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
#     st.metric("Completion Rate", f"{completion_rate:.1f}%")

# st.divider()

# # Main visualizations
# col1, col2 = st.columns([1, 1])

# # 1. NYC MAP - Request Density
# with col1:
#     st.subheader("🗺️ NYC Request Density Map")
    
#     if 'latitude' in df_filtered.columns and 'longitude' in df_filtered.columns:
#         # Create folium map
#         m = folium.Map(
#             location=[40.7128, -73.9854],  # NYC center
#             zoom_start=10,
#             tiles='OpenStreetMap'
#         )
        
#         # Add borough boundaries and counts
#         if 'borough' in df_filtered.columns:
#             borough_counts = df_filtered['borough'].value_counts()
            
#             # Borough coordinates (approximate centers)
#             borough_coords = {
#                 'MANHATTAN': [40.7831, -73.9712],
#                 'BROOKLYN': [40.6782, -73.9442],
#                 'QUEENS': [40.7282, -73.7949],
#                 'BRONX': [40.8448, -73.8648],
#                 'STATEN ISLAND': [40.5795, -74.1502]
#             }
            
#             max_count = borough_counts.max() if len(borough_counts) > 0 else 1
            
#             for borough, count in borough_counts.items():
#                 if borough in borough_coords:
#                     # Circle size based on count
#                     radius = (count / max_count) * 20 + 5
                    
#                     folium.CircleMarker(
#                         location=borough_coords[borough],
#                         radius=radius,
#                         popup=f"{borough}: {count:,} requests",
#                         color='red',
#                         fill=True,
#                         fillColor='red',
#                         fillOpacity=0.6
#                     ).add_to(m)
                    
#                     # Add text label
#                     folium.Marker(
#                         location=borough_coords[borough],
#                         icon=folium.DivIcon(
#                             div_id="label", 
#                             icon_size=(100, 20),
#                             icon_anchor=(50, 10),
#                             html=f'<div style="font-size: 12px; font-weight: bold; text-align: center; color: black; background-color: white; border-radius: 3px; padding: 2px;">{count:,}</div>'
#                         )
#                     ).add_to(m)
        
#         folium_static(m, width=500, height=400)
#     else:
#         st.info("Map requires latitude and longitude columns")

# # 2. Daily Request Volume
# with col2:
#     st.subheader("📈 Daily Request Volume")
    
#     if 'created_date' in df_filtered.columns:
#         daily_counts = df_filtered.groupby(df_filtered['created_date'].dt.date).size().reset_index()
#         daily_counts.columns = ['date', 'requests']
        
#         fig = px.line(
#             daily_counts, 
#             x='date', 
#             y='requests',
#             title="Daily 311 Request Volume",
#             labels={'requests': 'Number of Requests', 'date': 'Date'}
#         )
#         fig.update_layout(height=400)
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.info("Chart requires created_date column")

# # 3. Top Complaint Types
# col1, col2 = st.columns([1, 1])

# with col1:
#     st.subheader("🎯 Top Complaint Types")
    
#     if 'complaint_type' in df_filtered.columns:
#         complaint_counts = df_filtered['complaint_type'].value_counts().head(10)
        
#         fig = px.bar(
#             x=complaint_counts.values,
#             y=complaint_counts.index,
#             orientation='h',
#             title="Most Common Complaint Types",
#             labels={'x': 'Number of Requests', 'y': 'Complaint Type'}
#         )
#         fig.update_layout(height=400)
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.info("Chart requires complaint_type column")

# # 4. Status Distribution
# with col2:
#     st.subheader("📊 Request Status Distribution")
    
#     if 'status' in df_filtered.columns:
#         status_counts = df_filtered['status'].value_counts()
        
#         fig = px.pie(
#             values=status_counts.values,
#             names=status_counts.index,
#             title="Request Status Breakdown"
#         )
#         fig.update_layout(height=400)
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.info("Chart requires status column")

# # 5. Agency Performance (full width)
# st.subheader("🏢 Requests by Agency")

# if 'agency' in df_filtered.columns:
#     agency_counts = df_filtered['agency'].value_counts().head(10)
    
#     fig = px.bar(
#         x=agency_counts.index,
#         y=agency_counts.values,
#         title="Number of Requests by Agency",
#         labels={'x': 'Agency', 'y': 'Number of Requests'}
#     )
#     fig.update_layout(height=400)
#     st.plotly_chart(fig, use_container_width=True)
# else:
#     st.info("Chart requires agency column")

# # Borough comparison
# if 'borough' in df_filtered.columns:
#     st.subheader("🏙️ Borough Comparison")
    
#     # Borough stats table
#     borough_stats = df_filtered.groupby('borough').agg({
#         'unique_key': 'count',
#         'status': lambda x: (x == 'Closed').sum() if 'Closed' in x.values else 0
#     }).round(2)
    
#     borough_stats.columns = ['Total Requests', 'Closed Requests']
#     borough_stats['Completion Rate %'] = (
#         borough_stats['Closed Requests'] / borough_stats['Total Requests'] * 100
#     ).round(1)
    
#     st.dataframe(borough_stats, use_container_width=True)

# # Data preview
# with st.expander("📋 View Raw Data"):
#     st.dataframe(df_filtered.head(100), use_container_width=True)
#     st.caption(f"Showing first 100 rows of {len(df_filtered):,} total filtered records")

# # Download filtered data
# if len(df_filtered) > 0:
#     csv = df_filtered.to_csv(index=False)
#     st.download_button(
#         label="💾 Download Filtered Data as CSV",
#         data=csv,
#         file_name=f"nyc_311_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#         mime="text/csv"
#     )

# # Footer
# st.divider()
# st.markdown(
#     """
#     <div style='text-align: center; color: gray;'>
#         NYC 311 Service Requests Interactive Dashboard | Built with Streamlit
#     </div>
#     """, 
#     unsafe_allow_html=True
# )

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static

# Page config
st.set_page_config(
    page_title="NYC 311 Dashboard",
    page_icon="🗽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🗽 NYC 311 Service Requests Dashboard")
st.markdown("Interactive analysis of service requests across New York City boroughs")

# File upload
st.markdown("### 📁 Data Source")
uploaded_file = st.file_uploader(
    "Upload your NYC 311 CSV file (any filename accepted)", 
    type=['csv'],
    help="Upload any NYC 311 CSV file - filename doesn't matter, we'll auto-detect the columns!"
)

# Show file info if uploaded
if uploaded_file is not None:
    st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
    file_details = {
        "Filename": uploaded_file.name,
        "File size": f"{uploaded_file.size / 1024:.1f} KB"
    }
    st.json(file_details)

# Sample data function
@st.cache_data
def create_sample_data():
    np.random.seed(42)
    n_records = 1000
    
    boroughs = ['MANHATTAN', 'BROOKLYN', 'QUEENS', 'BRONX', 'STATEN ISLAND']
    agencies = ['NYPD', 'DOT', 'HPD', 'DSNY', 'DEP', 'DOHMH']
    complaint_types = [
        'Noise - Residential', 'Street Light Condition', 'Heat/Hot Water', 
        'Blocked Driveway', 'Water System', 'Illegal Parking', 'Traffic Signal Condition',
        'Rodent', 'Graffiti', 'Street Condition'
    ]
    statuses = ['Open', 'Closed', 'In Progress']
    
    # Generate dates
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = (end_date - start_date).days
    
    # NYC coordinates ranges
    lat_range = (40.4774, 40.9176)
    lon_range = (-74.2591, -73.7004)
    
    data = []
    for i in range(n_records):
        created_date = start_date + timedelta(days=np.random.randint(0, date_range))
        borough = np.random.choice(boroughs)
        status = np.random.choice(statuses, p=[0.2, 0.7, 0.1])
        
        # Closed date logic
        if status == 'Closed':
            closed_date = created_date + timedelta(days=np.random.randint(1, 30))
        else:
            closed_date = None
            
        data.append({
            'unique_key': f"key_{i}",
            'created_date': created_date,
            'closed_date': closed_date,
            'agency': np.random.choice(agencies),
            'complaint_type': np.random.choice(complaint_types),
            'borough': borough,
            'status': status,
            'latitude': np.random.uniform(*lat_range),
            'longitude': np.random.uniform(*lon_range),
            'incident_zip': str(np.random.randint(10001, 11697))
        })
    
    return pd.DataFrame(data)

# Load data
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Successfully loaded {len(df):,} records from {uploaded_file.name}")
        
        # Show column info
        st.info(f"📊 Detected columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
        
        # Auto-detect and convert date columns (flexible column names)
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                st.success(f"✅ Converted '{col}' to datetime")
            except:
                st.warning(f"⚠️ Could not convert '{col}' to datetime")
        
        # Auto-detect coordinate columns
        lat_columns = [col for col in df.columns if any(term in col.lower() for term in ['lat', 'latitude'])]
        lon_columns = [col for col in df.columns if any(term in col.lower() for term in ['lon', 'long', 'longitude'])]
        
        if lat_columns and lon_columns:
            st.success(f"🗺️ Map ready! Using {lat_columns[0]} and {lon_columns[0]} for coordinates")
            
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        df = create_sample_data()
        st.info("🔄 Using sample data instead")
else:
    df = create_sample_data()
    st.info("📁 No file uploaded. Using sample data for demonstration.")
    st.markdown("**👆 Upload any NYC 311 CSV file above to get started!**")

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Date range filter (flexible column detection)
date_columns = [col for col in df.columns if 'date' in col.lower()]
if date_columns:
    primary_date_col = date_columns[0]  # Use first date column found
    
    min_date = df[primary_date_col].min().date()
    max_date = df[primary_date_col].max().date()
    
    date_range = st.sidebar.date_input(
        f"Select Date Range ({primary_date_col})",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[
            (df[primary_date_col].dt.date >= start_date) & 
            (df[primary_date_col].dt.date <= end_date)
        ]
    else:
        df_filtered = df
else:
    df_filtered = df
    st.sidebar.info("No date columns detected")

# Borough filter (flexible column detection)
borough_columns = [col for col in df_filtered.columns if 'borough' in col.lower()]
if borough_columns:
    borough_col = borough_columns[0]
    boroughs = st.sidebar.multiselect(
        f"Select Boroughs ({borough_col})",
        options=df_filtered[borough_col].unique(),
        default=df_filtered[borough_col].unique()
    )
    df_filtered = df_filtered[df_filtered[borough_col].isin(boroughs)]

# Complaint type filter (flexible column detection)
complaint_columns = [col for col in df_filtered.columns if any(term in col.lower() for term in ['complaint', 'type'])]
if complaint_columns:
    complaint_col = complaint_columns[0]
    complaint_types = st.sidebar.multiselect(
        f"Select Complaint Types ({complaint_col})",
        options=df_filtered[complaint_col].unique(),
        default=list(df_filtered[complaint_col].unique())[:5]  # Show top 5 by default
    )
    df_filtered = df_filtered[df_filtered[complaint_col].isin(complaint_types)]

# Status filter (flexible column detection)
status_columns = [col for col in df_filtered.columns if 'status' in col.lower()]
if status_columns:
    status_col = status_columns[0]
    statuses = st.sidebar.multiselect(
        f"Select Status ({status_col})",
        options=df_filtered[status_col].unique(),
        default=df_filtered[status_col].unique()
    )
    df_filtered = df_filtered[df_filtered[status_col].isin(statuses)]

# Clear filters button
if st.sidebar.button("🔄 Clear All Filters"):
    st.experimental_rerun()

# Key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Requests", f"{len(df_filtered):,}")

with col2:
    closed_requests = len(df_filtered[df_filtered['status'] == 'Closed']) if 'status' in df_filtered.columns else 0
    st.metric("Closed Requests", f"{closed_requests:,}")

with col3:
    open_requests = len(df_filtered[df_filtered['status'] == 'Open']) if 'status' in df_filtered.columns else 0
    st.metric("Open Requests", f"{open_requests:,}")

with col4:
    completion_rate = (closed_requests / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
    st.metric("Completion Rate", f"{completion_rate:.1f}%")

st.divider()

# Main visualizations
col1, col2 = st.columns([1, 1])

# 1. NYC MAP - Request Density
with col1:
    st.subheader("🗺️ NYC Request Density Map")
    
    if 'latitude' in df_filtered.columns and 'longitude' in df_filtered.columns:
        # Create folium map
        m = folium.Map(
            location=[40.7128, -73.9854],  # NYC center
            zoom_start=10,
            tiles='OpenStreetMap'
        )
        
        # Add borough boundaries and counts
        if 'borough' in df_filtered.columns:
            borough_counts = df_filtered['borough'].value_counts()
            
            # Borough coordinates (approximate centers)
            borough_coords = {
                'MANHATTAN': [40.7831, -73.9712],
                'BROOKLYN': [40.6782, -73.9442],
                'QUEENS': [40.7282, -73.7949],
                'BRONX': [40.8448, -73.8648],
                'STATEN ISLAND': [40.5795, -74.1502]
            }
            
            max_count = borough_counts.max() if len(borough_counts) > 0 else 1
            
            for borough, count in borough_counts.items():
                if borough in borough_coords:
                    # Circle size based on count
                    radius = (count / max_count) * 20 + 5
                    
                    folium.CircleMarker(
                        location=borough_coords[borough],
                        radius=radius,
                        popup=f"{borough}: {count:,} requests",
                        color='red',
                        fill=True,
                        fillColor='red',
                        fillOpacity=0.6
                    ).add_to(m)
                    
                    # Add text label
                    folium.Marker(
                        location=borough_coords[borough],
                        icon=folium.DivIcon(
                            icon_size=(100, 20),
                            icon_anchor=(50, 10),
                            html=f'<div style="font-size: 12px; font-weight: bold; text-align: center; color: black; background-color: white; border-radius: 3px; padding: 2px;">{count:,}</div>'
                        )
                    ).add_to(m)
        
        folium_static(m, width=500, height=400)
    else:
        st.info("Map requires latitude and longitude columns")

# 2. Daily Request Volume
with col2:
    st.subheader("📈 Daily Request Volume")
    
    if 'created_date' in df_filtered.columns:
        daily_counts = df_filtered.groupby(df_filtered['created_date'].dt.date).size().reset_index()
        daily_counts.columns = ['date', 'requests']
        
        fig = px.line(
            daily_counts, 
            x='date', 
            y='requests',
            title="Daily 311 Request Volume",
            labels={'requests': 'Number of Requests', 'date': 'Date'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Chart requires created_date column")

# 3. Top Complaint Types
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎯 Top Complaint Types")
    
    if 'complaint_type' in df_filtered.columns:
        complaint_counts = df_filtered['complaint_type'].value_counts().head(10)
        
        fig = px.bar(
            x=complaint_counts.values,
            y=complaint_counts.index,
            orientation='h',
            title="Most Common Complaint Types",
            labels={'x': 'Number of Requests', 'y': 'Complaint Type'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Chart requires complaint_type column")

# 4. Status Distribution
with col2:
    st.subheader("📊 Request Status Distribution")
    
    if 'status' in df_filtered.columns:
        status_counts = df_filtered['status'].value_counts()
        
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Request Status Breakdown"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Chart requires status column")

# 5. Agency Performance (full width)
st.subheader("🏢 Requests by Agency")

if 'agency' in df_filtered.columns:
    agency_counts = df_filtered['agency'].value_counts().head(10)
    
    fig = px.bar(
        x=agency_counts.index,
        y=agency_counts.values,
        title="Number of Requests by Agency",
        labels={'x': 'Agency', 'y': 'Number of Requests'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Chart requires agency column")

# Borough comparison
if 'borough' in df_filtered.columns:
    st.subheader("🏙️ Borough Comparison")
    
    # Borough stats table
    borough_stats = df_filtered.groupby('borough').agg({
        'unique_key': 'count',
        'status': lambda x: (x == 'Closed').sum() if 'Closed' in x.values else 0
    }).round(2)
    
    borough_stats.columns = ['Total Requests', 'Closed Requests']
    borough_stats['Completion Rate %'] = (
        borough_stats['Closed Requests'] / borough_stats['Total Requests'] * 100
    ).round(1)
    
    st.dataframe(borough_stats, use_container_width=True)

# Data preview
with st.expander("📋 View Raw Data"):
    st.dataframe(df_filtered.head(100), use_container_width=True)
    st.caption(f"Showing first 100 rows of {len(df_filtered):,} total filtered records")

# Download filtered data
if len(df_filtered) > 0:
    csv = df_filtered.to_csv(index=False)
    st.download_button(
        label="💾 Download Filtered Data as CSV",
        data=csv,
        file_name=f"nyc_311_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        NYC 311 Service Requests Interactive Dashboard | Built with Streamlit
    </div>
    """, 
    unsafe_allow_html=True
)