# In Tab 1, replace the comparison section with:

if astro_image is not None:
    enhanced, soliton, pdp = process_qci_astro(astro_image, omega1, fringe1, soliton_scale1)
    
    # Create comparison figure with proper orientation
    fig_comp, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Original (no overlay)
    ax1.imshow(astro_image, cmap='gray', origin='upper')
    ax1.set_title("Original: Standard View\n(Public Data)")
    ax1.axis('off')
    add_scale_bar(ax1, astro_image.shape[1], pixel_scale_kpc=pixel_scale_kpc)
    
    # Right: Enhanced with overlays
    ax2.imshow(enhanced, cmap='gray', origin='upper')
    ax2.set_title("Enhanced: FDM Soliton + PDP Entanglement\n(Quantum Overlays)")
    ax2.axis('off')
    add_scale_bar(ax2, astro_image.shape[1], pixel_scale_kpc=pixel_scale_kpc)
    
    plt.tight_layout()
    st.pyplot(fig_comp)
    
    # Download button
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"qcaus_{image_name}_omega{omega1:.2f}_{timestamp}"
    st.markdown(get_image_download_link(fig_comp, f"{base_filename}_comparison.png", "📸 Download Side-by-Side Comparison"), unsafe_allow_html=True)
    plt.close(fig_comp)
