with tab1:
    st.header("⚛️ Quantum Field Visualization")
    st.markdown("""
    **Scientific Framework:** Quantum superposition of astrophysical signals with dark matter and dark photon fields
    
    | Field | Formula | Physical Interpretation |
    |-------|---------|------------------------|
    | **FDM Soliton** | ρ(r) = ρ₀ [sin(kr)/(kr)]² | Dark matter density wave function |
    | **PDP Field** | ℒ_mix = (ε/2) F_μν F'^μν | Dark photon field from kinetic mixing |
    | **Quantum State** | |Ψ⟩ = |Ψ_astro⟩ + α|Ψ_FDM⟩ + β|Ψ_PDP⟩ | Superposition of quantum fields |
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚛️ Quantum Parameters")
        omega = st.slider("Ω (Entanglement Strength)", 0.0, 1.0, 0.5, 0.01,
                          help="Photon-dark photon coupling strength")
        fringe = st.slider("λ (Fringe Scale)", 0.1, 3.0, 1.5, 0.05,
                           help="Quantum interference wavelength")
        soliton_scale = st.slider("k (Soliton Scale)", 0.5, 3.0, 1.0, 0.05,
                                   help="Dark matter wave number: k = 2π/λ_dm")
        fdm_coupling = st.slider("α (FDM Coupling)", 0.0, 2.0, 0.8, 0.05,
                                  help="FDM field coupling strength")
        pdp_coupling = st.slider("β (PDP Coupling)", 0.0, 2.0, 1.0, 0.05,
                                  help="Dark photon field coupling")
        
        st.markdown("---")
        st.markdown("### 🎨 Visualization")
        cmap_choice = st.selectbox("Color Map", list(SCIENTIFIC_CMAPS.keys()), index=0)
        cmap_value = SCIENTIFIC_CMAPS[cmap_choice]
        
        if astro_image is not None:
            quantum_state, fdm_field, pdp_field = quantum_state_superposition(
                astro_image, omega, fringe, soliton_scale, fdm_coupling, pdp_coupling)
            
            params = {
                'omega': omega, 'fringe': fringe, 'soliton_scale': soliton_scale,
                'fdm_coupling': fdm_coupling, 'pdp_coupling': pdp_coupling
            }
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"qcaus_{image_name}_omega{omega:.2f}_lambda{fringe:.2f}_{timestamp}"
            
            st.markdown("---")
            st.markdown("### 📥 Export")
            
            # Scientific 2x2 grid visualization
            fig_grid = create_quantum_field_visualization(
                astro_image, quantum_state, fdm_field, pdp_field, params, pixel_scale_kpc)
            st.markdown(get_image_download_link(fig_grid, f"{base_filename}_quantum_fields.png", 
                                                "📊 Download Quantum Field Grid"), unsafe_allow_html=True)
            plt.close(fig_grid)
    
    with col2:
        if astro_image is not None:
            quantum_state, fdm_field, pdp_field = quantum_state_superposition(
                astro_image, omega, fringe, soliton_scale, fdm_coupling, pdp_coupling)
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(quantum_state, cmap=cmap_value, origin='upper')
            ax.set_title(f"Quantum Superposition State\n|Ψ⟩ = |Ψ_astro⟩ + {fdm_coupling:.2f}|Ψ_FDM⟩ + {pdp_coupling:.2f}|Ψ_PDP⟩", fontsize=12)
            ax.axis('off')
            add_scale_bar(ax, quantum_state.shape[1], pixel_scale_kpc=pixel_scale_kpc)
            plt.colorbar(ax.images[0], ax=ax, label="Quantum Amplitude")
            st.pyplot(fig)
            st.caption("Quantum interference patterns reveal hidden dark matter and dark photon structures")
            plt.close(fig)
            
            # Download individual fields
            fig_fdm, ax_fdm = plt.subplots(figsize=(6, 6))
            ax_fdm.imshow(fdm_field, cmap='Greens', origin='upper')
            ax_fdm.set_title(f"FDM Soliton Field\nρ(r) = ρ₀ [sin({soliton_scale:.2f}r)/({soliton_scale:.2f}r)]²")
            ax_fdm.axis('off')
            add_scale_bar(ax_fdm, fdm_field.shape[1], pixel_scale_kpc=pixel_scale_kpc)
            st.markdown(get_image_download_link(fig_fdm, f"{base_filename}_fdm_field.png", 
                                                "🌌 Download FDM Field"), unsafe_allow_html=True)
            plt.close(fig_fdm)
            
            fig_pdp, ax_pdp = plt.subplots(figsize=(6, 6))
            ax_pdp.imshow(pdp_field, cmap='Blues', origin='upper')
            ax_pdp.set_title(f"PDP Quantum Field\nℒ_mix = (ε/2) F_μν F'^μν\nΩ={omega:.2f}, λ={fringe:.2f}")
            ax_pdp.axis('off')
            add_scale_bar(ax_pdp, pdp_field.shape[1], pixel_scale_kpc=pixel_scale_kpc)
            st.markdown(get_image_download_link(fig_pdp, f"{base_filename}_pdp_field.png", 
                                                "🌀 Download PDP Field"), unsafe_allow_html=True)
            plt.close(fig_pdp)
            
            # Quantum metrics
            st.markdown("---")
            st.markdown("### 📊 Quantum Field Metrics")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Max FDM Amplitude", f"{np.max(fdm_field):.3f}")
            col_m2.metric("Max PDP Amplitude", f"{np.max(pdp_field):.3f}")
            col_m3.metric("Mean Interference", f"{np.mean(fdm_field * pdp_field):.3e}")
            col_m4.metric("Field Correlation", f"{np.corrcoef(fdm_field.flatten(), pdp_field.flatten())[0,1]:.3f}")
        else:
            st.info("👈 Select or upload an image to begin quantum field analysis")
    
    # ============================================================================
    # BEFORE/AFTER COMPARISON WITH QUANTUM OVERLAYS
    # ============================================================================
    
    if astro_image is not None:
        st.markdown("---")
        st.markdown("### 🔬 Before/After: Quantum Overlay Comparison")
        st.markdown("*Comparing standard astronomical image with quantum superposition state*")
        
        quantum_state, fdm_field, pdp_field = quantum_state_superposition(
            astro_image, omega, fringe, soliton_scale, fdm_coupling, pdp_coupling)
        
        # Create before/after comparison with quantum overlays
        fig_before_after, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # LEFT: Original Image (Before)
        ax1.imshow(astro_image, cmap='gray', origin='upper')
        ax1.set_title("Before: Standard Astrophysical Image\n(Public Data - HST/JWST)", fontsize=12)
        ax1.axis('off')
        add_scale_bar(ax1, astro_image.shape[1], pixel_scale_kpc=pixel_scale_kpc)
        
        # RIGHT: Quantum Superposition State (After)
        ax2.imshow(quantum_state, cmap=cmap_value, origin='upper')
        ax2.set_title(f"After: Quantum Superposition State\n"
                      f"|Ψ⟩ = |Ψ_astro⟩ + α|Ψ_FDM⟩ + β|Ψ_PDP⟩\n"
                      f"α={fdm_coupling:.2f}, β={pdp_coupling:.2f}, Ω={omega:.2f}, λ={fringe:.2f}", fontsize=12)
        ax2.axis('off')
        add_scale_bar(ax2, quantum_state.shape[1], pixel_scale_kpc=pixel_scale_kpc)
        
        plt.tight_layout()
        st.pyplot(fig_before_after)
        
        # Download button
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"qcaus_{image_name}_omega{omega:.2f}_lambda{fringe:.2f}_{timestamp}"
        st.markdown(get_image_download_link(fig_before_after, f"{base_filename}_before_after.png", 
                                            "📸 Download Before/After Comparison"), unsafe_allow_html=True)
        plt.close(fig_before_after)
        
        # Scientific caption
        st.caption("""
        **Scientific Interpretation:**
        - **Left (Before):** Original astronomical image showing only electromagnetic radiation
        - **Right (After):** Quantum superposition state revealing:
          - **Dark Matter Structure**: FDM soliton interference patterns from ρ(r) = ρ₀ [sin(kr)/(kr)]²
          - **Dark Photon Field**: PDP signatures from kinetic mixing ℒ_mix = (ε/2) F_μν F'^μν
          - **Quantum Coherence**: Interference fringes between dark matter and dark photon fields
        """)
