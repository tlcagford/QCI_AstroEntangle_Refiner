# Add new section to QCAUS sidebar
with st.sidebar:
    st.header("🔄 Radar Data Import")
    st.caption("Import StealthPDPRadar detection data")
    
    uploaded_radar = st.file_uploader(
        "Upload Radar JSON Export",
        type=['json'],
        help="Export from StealthPDPRadar as Complete JSON"
    )
    
    if uploaded_radar is not None:
        radar_data = json.load(uploaded_radar)
        st.success(f"Loaded {len(radar_data.get('pdp_detections', []))} detections")
        
        # Convert to optical overlay
        optical_overlay = radar_to_optical_overlay(radar_data)
        
        # Display overlay options
        st.subheader("🎨 Overlay Settings")
        overlay_opacity = st.slider("Opacity", 0.0, 1.0, 0.6)
        show_entanglement = st.checkbox("Show Entanglement (Cyan)", True)
        show_darkmode = st.checkbox("Show Dark Mode (Violet)", True)
        show_stealth = st.checkbox("Show Stealth (IR Red)", True)
